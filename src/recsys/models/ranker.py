"""Ranker de dos etapas: candidatos de ALS + popularidad por género +
popularidad global, reordenados por un `LGBMRanker` (LightGBM,
objetivo `lambdarank`) entrenado para combinar esas tres señales mejor
de lo que cada una hace sola, más features auxiliares de autor/editorial/
año de edición/diversidad de género/recencia/co-lectura ítem-ítem/
similitud de resumen/popularidad y frecuencia de macro-género (ver
`calcular_features_auxiliares`).

Un ranker que usa el score de otros modelos como *features* necesita que
esos scores salgan de datos que el ranker no vio como etiqueta -- si no,
memoriza en vez de aprender a combinar señales. Por eso el entrenamiento
de este módulo espera un split de **tres niveles** (ver
`scripts/evaluate_ranker.py`): un tramo para fitear ALS/popularidad/
género (las fuentes de candidatos), otro tramo con etiquetas conocidas
para entrenar el `LGBMRanker`, y un tercero de hold-out final para medir
NDCG@k del pipeline completo.

Este módulo se evalúa con validación cruzada sobre varios splits/seeds
(`evaluation.evaluar_multisplit`), no un solo split -- después del
episodio en el que un sweep de ALS sobre un único split mejoró el NDCG
local pero empeoró el score real de Kaggle, ver `experiments/bitacora.md`.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from recsys.data import libros_leidos_por_usuario, split_train_val
from recsys.evaluation import evaluar_ndcg_personalizado
from recsys.models.als import fit_als
from recsys.models.als import recomendar_por_usuario as _recomendar_als
from recsys.models.popularity import fit_popularity
from recsys.models.popularity_segmentada import (
    _normalizar_genero,
    fit_popularidad_por_genero_macro,
    fit_popularity_por_genero,
    genero_preferido_por_usuario,
    normalizar_genero_macro,
)

FEATURES = [
    "score_als",
    "rank_als",
    "en_als",
    "score_popularidad",
    "rank_popularidad",
    "en_popularidad",
    "score_genero",
    "rank_genero",
    "en_genero",
    "n_interacciones_libro",
    "n_interacciones_usuario",
    "en_autor_leido",
    "n_libros_autor_leidos",
    "anio_edicion_dif",
    "n_generos_distintos_usuario",
    "dias_desde_ultima_interaccion_usuario",
    "score_coleido",
    "en_editorial_leida",
    "n_libros_editorial_leidos",
    "sim_resumen_historial",
    "popularidad_genero_macro_candidato",
    "frecuencia_genero_macro_usuario",
    "n_libros_editorial_catalogo",
]

N_MAX_FEATURES_TFIDF = 20000
MIN_DF_TFIDF = 2
MAX_DF_TFIDF = 0.8

SENTINEL_DIAS_DESCONOCIDO = 99999


def _calcular_cooccurrencia(matriz_usuario_libro, libros_por_columna: list) -> tuple:
    """Matriz ítem×ítem de co-lectura: `cooc[i, j]` = cantidad de
    usuarios (de `train_candidatos`) que leyeron tanto el libro de la
    columna `i` como el de la columna `j`. Se arma con un solo matmul
    disperso (`X.T @ X`, ~3s para 48k libros / 461k interacciones en este
    dataset -- la razón por la que esta feature se había dejado afuera
    como "cara de calcular bien" ya no aplica calculándola así, en vez de
    con loops por usuario) reusando la matriz binaria que ya arma
    `fit_als` -- no hay leakage nuevo, es la misma fuente que ALS.

    Devuelve `(cooc, columna_por_libro)`, con `columna_por_libro` el
    mismo mapeo `id_libro -> índice de columna` que usa ALS
    (`libros_por_columna`), para poder indexar `cooc` con `id_libro`.
    """
    X = sp.csr_matrix(matriz_usuario_libro > 0, dtype=np.int32)
    cooc = (X.T @ X).tocsr()
    columna_por_libro = {id_libro: idx for idx, id_libro in enumerate(libros_por_columna)}
    return cooc, columna_por_libro


def _calcular_perfil_texto(interacciones: pd.DataFrame, libros: pd.DataFrame, fila_por_usuario: dict) -> dict:
    """Similitud de texto entre el historial de un usuario y un
    candidato, basada en `libros.resumen`.

    El vectorizador TF-IDF se fittea sobre el `resumen` de **todo el
    catálogo** (`libros`, sin filtrar por `interacciones`) -- a
    diferencia de las demás features, esto es metadata estática del
    libro (como `autor`/`anio_edicion`), no depende de qué interacción
    cayó en train o en val, así que no hace falta restringirlo a
    `train_candidatos`. Lo que sí es interacción-derivado, y por lo tanto
    debe calcularse solo con `train_candidatos` (=`interacciones` acá),
    es el *perfil* de cada usuario: el promedio (ponderado por
    co-lectura, vía matmul disperso) de los vectores TF-IDF de los libros
    que leyó.

    Todo se normaliza a norma L2 por fila, así que un producto interno
    entre una fila de `tfidf_norm` (candidato) y una fila de
    `perfil_usuario_norm` (usuario) es una similitud coseno en [0, 1]
    (aproximada para el usuario, porque es el centroide normalizado de
    sus lecturas, no el promedio de sus similitudes individuales -- alcanza
    como señal para el ranker, no hace falta que sea una métrica exacta).

    Las filas de `perfil_usuario_norm` quedan en el mismo orden que
    `matriz_usuario_libro`/`fila_por_usuario` (recibido como parámetro,
    no se recalcula un índice de usuarios nuevo) para poder indexarlo
    igual que la matriz de ALS.
    """
    con_resumen = libros.dropna(subset=["resumen"])
    con_resumen = con_resumen[con_resumen["resumen"].astype(str).str.strip() != ""]

    if con_resumen.empty:
        return {"tfidf_norm": None, "fila_por_libro_texto": {}, "perfil_usuario_norm": None}

    vectorizador = TfidfVectorizer(max_features=N_MAX_FEATURES_TFIDF, min_df=MIN_DF_TFIDF, max_df=MAX_DF_TFIDF)
    tfidf_norm = vectorizador.fit_transform(con_resumen["resumen"].astype(str)).tocsr()  # ya normalizado L2 por default
    fila_por_libro_texto = {id_libro: idx for idx, id_libro in enumerate(con_resumen["id_libro"])}

    n_usuarios = len(fila_por_usuario)

    # Matriz binaria usuario x libro (solo libros con resumen), en el mismo
    # espacio de filas que fila_por_usuario, construida vectorizadamente
    # (no con un loop por interacción) para que escale igual que el resto
    # del pipeline.
    interacciones_con_texto = interacciones.assign(
        _fila_usuario=interacciones["id_lector"].map(fila_por_usuario),
        _fila_texto=interacciones["id_libro"].map(fila_por_libro_texto),
    ).dropna(subset=["_fila_usuario", "_fila_texto"])

    X_bin_texto = sp.csr_matrix(
        (
            np.ones(len(interacciones_con_texto)),
            (interacciones_con_texto["_fila_usuario"].astype(int), interacciones_con_texto["_fila_texto"].astype(int)),
        ),
        shape=(n_usuarios, tfidf_norm.shape[0]),
    )
    perfil_usuario = X_bin_texto @ tfidf_norm
    normas = np.sqrt(perfil_usuario.multiply(perfil_usuario).sum(axis=1)).A1
    normas[normas == 0] = 1.0  # evita división por cero para usuarios sin lecturas con resumen
    perfil_usuario_norm = sp.diags(1.0 / normas) @ perfil_usuario

    return {
        "tfidf_norm": tfidf_norm.tocsr(),
        "fila_por_libro_texto": fila_por_libro_texto,
        "perfil_usuario_norm": perfil_usuario_norm.tocsr(),
    }


def calcular_features_auxiliares(
    interacciones: pd.DataFrame,
    libros: pd.DataFrame,
    matriz_usuario_libro,
    fila_por_usuario: dict,
    libros_por_columna: list,
) -> dict:
    """Precalcula, a partir de `interacciones` (siempre `train_candidatos`,
    nunca datos que el ranker vea como etiqueta), los lookups que necesita
    `generar_candidatos_con_features` para las features de autor/editorial/
    año de edición/diversidad de género/recencia/co-lectura/texto:

    - `autor_por_libro` / `anio_edicion_por_libro`: metadata directa de
      `libros` (`anio_edicion` parseado a numérico, `errors="coerce"`).
    - `n_libros_autor_leidos_por_usuario` / `n_libros_editorial_leidos_por_usuario`:
      `{id_lector: {autor|editorial: n}}` -- cuántos libros de cada
      autor/editorial ya leyó cada usuario.
    - `anio_edicion_promedio_por_usuario`: antigüedad promedio de lo que
      lee cada usuario, para comparar contra la del candidato.
    - `n_generos_distintos_por_usuario`: cuántos géneros distintos
      (normalizados, mismo criterio que `popularity_segmentada.py`) leyó
      cada usuario -- generalista vs especializado.
    - `dias_desde_ultima_interaccion_por_usuario`: recencia general del
      usuario, relativa a la fecha más reciente de todo `interacciones`.
    - `cooc` / `columna_por_libro`: matriz ítem×ítem de co-lectura y el
      índice para consultarla -- ver `_calcular_cooccurrencia`.
    - `tfidf_norm` / `fila_por_libro_texto` / `perfil_usuario_norm`:
      similitud de texto entre historial y candidato -- ver
      `_calcular_perfil_texto`.
    - `genero_macro_por_libro` / `score_por_libro_genero_macro` /
      `frecuencia_genero_macro_por_usuario`: macro-género del candidato
      (10 familias de dominio), popularidad pooleada a ese nivel, y
      frecuencia del macro-género en el historial del usuario -- ver
      `popularity_segmentada.normalizar_genero_macro`/
      `fit_popularidad_por_genero_macro`.
    - `n_libros_por_editorial`: tamaño del catálogo de la editorial del
      candidato (cuántos libros tiene en total `libros`, no solo los
      leídos) -- señal de volumen, no de historial del usuario.

    `matriz_usuario_libro`, `fila_por_usuario` y `libros_por_columna` son
    los que ya devuelve `fit_als` sobre el mismo `train_candidatos` --
    se reusan acá para no re-fitear ALS ni duplicar el índice de usuarios.
    """
    metadata = libros.set_index("id_libro")
    autor_por_libro = metadata["autor"].to_dict()
    editorial_por_libro = metadata["editorial"].to_dict()
    anio_edicion_por_libro = pd.to_numeric(metadata["anio_edicion"], errors="coerce").to_dict()
    genero_por_libro = _normalizar_genero(metadata["genero"]).to_dict()

    con_autor = interacciones.assign(autor=interacciones["id_libro"].map(autor_por_libro))
    n_libros_autor_leidos_por_usuario: dict = {}
    for (id_lector, autor), n in con_autor.dropna(subset=["autor"]).groupby(["id_lector", "autor"]).size().items():
        n_libros_autor_leidos_por_usuario.setdefault(id_lector, {})[autor] = int(n)

    con_editorial = interacciones.assign(editorial=interacciones["id_libro"].map(editorial_por_libro))
    n_libros_editorial_leidos_por_usuario: dict = {}
    for (id_lector, editorial), n in (
        con_editorial.dropna(subset=["editorial"]).groupby(["id_lector", "editorial"]).size().items()
    ):
        n_libros_editorial_leidos_por_usuario.setdefault(id_lector, {})[editorial] = int(n)

    # Tamaño de la editorial (cuántos libros tiene en TODO el catálogo,
    # no solo los leídos por algún usuario) -- a diferencia de
    # `n_libros_editorial_leidos_por_usuario` (depende del historial de
    # cada usuario), esto es una propiedad del libro en sí: 2.762
    # editoriales distintas entre libros con interacción, con una cola
    # muy larga (91% tiene menos de 20 libros, 51% tiene exactamente 1)
    # y sin una agrupación temática natural como la de género -- no tiene
    # sentido una "macro-editorial" categórica, pero sí una señal
    # numérica simple de volumen. Se calcula sobre `libros` completo
    # (metadata estática, no depende del split, mismo criterio que
    # autor/año de edición/resumen).
    n_libros_por_editorial = metadata["editorial"].value_counts().to_dict()

    con_anio = interacciones.assign(anio_edicion=interacciones["id_libro"].map(anio_edicion_por_libro))
    anio_edicion_promedio_por_usuario = (
        con_anio.dropna(subset=["anio_edicion"]).groupby("id_lector")["anio_edicion"].mean().to_dict()
    )

    con_genero = interacciones.assign(genero=interacciones["id_libro"].map(genero_por_libro))
    n_generos_distintos_por_usuario = (
        con_genero.dropna(subset=["genero"]).groupby("id_lector")["genero"].nunique().to_dict()
    )

    # Macro-género (10 familias de dominio, ver `popularity_segmentada.py`):
    # popularidad pooleada a nivel macro (propiedad del libro, no del
    # usuario -- complementa a score_genero/rank_genero, que miran el
    # género *preferido del usuario*) + frecuencia del macro-género en el
    # historial de cada usuario (señal graduada, no solo diversidad
    # binaria/conteo como n_generos_distintos_usuario).
    genero_macro_por_libro = normalizar_genero_macro(metadata["genero"]).to_dict()
    stats_por_genero_macro = fit_popularidad_por_genero_macro(interacciones, libros)
    score_por_libro_genero_macro: dict = {}
    for tabla in stats_por_genero_macro.values():
        score_por_libro_genero_macro.update(tabla.set_index("id_libro")["score"].to_dict())

    con_genero_macro = interacciones.assign(genero_macro=interacciones["id_libro"].map(genero_macro_por_libro))
    con_genero_macro = con_genero_macro.dropna(subset=["genero_macro"])
    conteos_genero_macro = con_genero_macro.groupby(["id_lector", "genero_macro"]).size().rename("n").reset_index()
    total_por_usuario = conteos_genero_macro.groupby("id_lector")["n"].transform("sum")
    conteos_genero_macro["frecuencia"] = conteos_genero_macro["n"] / total_por_usuario
    frecuencia_genero_macro_por_usuario: dict = {}
    for id_lector, genero_macro, frecuencia in zip(
        conteos_genero_macro["id_lector"], conteos_genero_macro["genero_macro"], conteos_genero_macro["frecuencia"]
    ):
        frecuencia_genero_macro_por_usuario.setdefault(id_lector, {})[genero_macro] = frecuencia

    fechas = pd.to_datetime(interacciones["fecha"], format="%d-%m-%Y", errors="coerce")
    fecha_referencia = fechas.max()
    ultima_fecha_por_usuario = (
        interacciones.assign(_fecha=fechas).dropna(subset=["_fecha"]).groupby("id_lector")["_fecha"].max()
    )
    dias_desde_ultima_interaccion_por_usuario = (
        (fecha_referencia - ultima_fecha_por_usuario).dt.days.to_dict()
    )

    cooc, columna_por_libro = _calcular_cooccurrencia(matriz_usuario_libro, libros_por_columna)
    perfil_texto = _calcular_perfil_texto(interacciones, libros, fila_por_usuario)

    return {
        "autor_por_libro": autor_por_libro,
        "anio_edicion_por_libro": anio_edicion_por_libro,
        "n_libros_autor_leidos_por_usuario": n_libros_autor_leidos_por_usuario,
        "editorial_por_libro": editorial_por_libro,
        "n_libros_editorial_leidos_por_usuario": n_libros_editorial_leidos_por_usuario,
        "n_libros_por_editorial": n_libros_por_editorial,
        "anio_edicion_promedio_por_usuario": anio_edicion_promedio_por_usuario,
        "n_generos_distintos_por_usuario": n_generos_distintos_por_usuario,
        "dias_desde_ultima_interaccion_por_usuario": dias_desde_ultima_interaccion_por_usuario,
        "cooc": cooc,
        "columna_por_libro": columna_por_libro,
        "genero_macro_por_libro": genero_macro_por_libro,
        "score_por_libro_genero_macro": score_por_libro_genero_macro,
        "frecuencia_genero_macro_por_usuario": frecuencia_genero_macro_por_usuario,
        **perfil_texto,
    }


def generar_candidatos_con_features(
    usuarios: list,
    modelo_als,
    matriz_usuario_libro,
    fila_por_usuario: dict,
    libros_por_columna: list,
    stats_popularidad: pd.DataFrame,
    stats_por_genero: dict,
    genero_por_usuario: dict,
    libros_leidos: dict,
    n_interacciones_por_usuario: dict,
    features_auxiliares: dict,
    n_por_fuente: int = 150,
) -> pd.DataFrame:
    """Arma, para cada usuario, la unión de candidatos de las tres fuentes
    (ALS, popularidad por género, popularidad global) con sus features.

    Reusa las estructuras que ya arman `fit_popularity` (`stats_popularidad`),
    `fit_popularity_por_genero` (`stats_por_genero`) y `genero_preferido_por_usuario`
    -- no duplica esa lógica. Cada candidato queda con `score_*`/`rank_*`
    (posición real dentro de esa fuente, no posición entre los candidatos
    finalmente elegidos) por fuente que lo propuso, y `en_*` (1/0) indicando
    qué fuentes lo propusieron. Un candidato que no vino de una fuente
    queda con score 0.0 y rank `n_por_fuente` (sentinel: "justo afuera de
    la ventana de esa fuente").

    `features_auxiliares` es el dict que arma `calcular_features_auxiliares`
    (autor, año de edición, diversidad de género, recencia). Un candidato
    sin dato conocido (autor/año de edición ausente, o usuario sin
    historial suficiente) queda con el sentinel correspondiente (0 para
    conteos/diferencias, `SENTINEL_DIAS_DESCONOCIDO` para recencia) -- no
    se imputa a ciegas.

    Devuelve un DataFrame largo con columnas `id_lector`, `id_libro` +
    `FEATURES`.
    """
    autor_por_libro = features_auxiliares["autor_por_libro"]
    anio_edicion_por_libro = features_auxiliares["anio_edicion_por_libro"]
    n_libros_autor_leidos_por_usuario = features_auxiliares["n_libros_autor_leidos_por_usuario"]
    editorial_por_libro = features_auxiliares.get("editorial_por_libro", {})
    n_libros_editorial_leidos_por_usuario = features_auxiliares.get("n_libros_editorial_leidos_por_usuario", {})
    n_libros_por_editorial = features_auxiliares.get("n_libros_por_editorial", {})
    anio_edicion_promedio_por_usuario = features_auxiliares["anio_edicion_promedio_por_usuario"]
    n_generos_distintos_por_usuario = features_auxiliares["n_generos_distintos_por_usuario"]
    dias_desde_ultima_interaccion_por_usuario = features_auxiliares["dias_desde_ultima_interaccion_por_usuario"]
    cooc = features_auxiliares.get("cooc")
    columna_por_libro = features_auxiliares.get("columna_por_libro", {})
    tfidf_norm = features_auxiliares.get("tfidf_norm")
    fila_por_libro_texto = features_auxiliares.get("fila_por_libro_texto", {})
    perfil_usuario_norm = features_auxiliares.get("perfil_usuario_norm")
    genero_macro_por_libro = features_auxiliares.get("genero_macro_por_libro", {})
    score_por_libro_genero_macro = features_auxiliares.get("score_por_libro_genero_macro", {})
    frecuencia_genero_macro_por_usuario = features_auxiliares.get("frecuencia_genero_macro_por_usuario", {})
    n_por_libro = stats_popularidad.set_index("id_libro")["n"].to_dict()
    score_popularidad_por_libro = stats_popularidad.set_index("id_libro")["score"].to_dict()
    ranking_global_ids = stats_popularidad["id_libro"].tolist()
    rank_popularidad_por_libro = {libro: i for i, libro in enumerate(ranking_global_ids)}

    usuarios_con_als = [u for u in usuarios if u in fila_por_usuario]
    ids_items_als: dict = {}
    scores_als: dict = {}
    if usuarios_con_als:
        filas = np.array([fila_por_usuario[u] for u in usuarios_con_als])
        ids_items, scores = modelo_als.recommend(
            filas, matriz_usuario_libro[filas], N=n_por_fuente, filter_already_liked_items=True
        )
        for id_lector, fila_ids, fila_scores in zip(usuarios_con_als, ids_items, scores):
            ids_items_als[id_lector] = [libros_por_columna[idx] for idx in fila_ids]
            scores_als[id_lector] = list(fila_scores)

    # Co-lectura: acotado a los mismos usuarios_con_als (misma fila que ALS)
    # y a un matmul disperso sobre ese batch -- nunca sobre los ~10k
    # usuarios completos, para no explotar en memoria (ver docstring de
    # `_calcular_cooccurrencia`).
    co_scores_por_usuario: dict = {}
    if usuarios_con_als and cooc is not None:
        X_batch = sp.csr_matrix(matriz_usuario_libro[filas] > 0, dtype=np.int32)
        co_scores_batch = (X_batch @ cooc).tocsr()
        for id_lector, fila_row in zip(usuarios_con_als, co_scores_batch):
            co_scores_por_usuario[id_lector] = dict(zip(fila_row.indices, fila_row.data))

    filas_resultado = []
    for id_lector in usuarios:
        vistos = set(libros_leidos.get(id_lector, set()))
        candidatos: dict = {}

        for rank, (id_libro, score) in enumerate(
            zip(ids_items_als.get(id_lector, []), scores_als.get(id_lector, []))
        ):
            if id_libro in vistos:
                continue
            c = candidatos.setdefault(id_libro, {})
            c["score_als"] = float(score)
            c["rank_als"] = rank
            c["en_als"] = 1

        agregados = 0
        for id_libro in ranking_global_ids:
            if agregados >= n_por_fuente:
                break
            if id_libro in vistos:
                continue
            c = candidatos.setdefault(id_libro, {})
            c["score_popularidad"] = score_popularidad_por_libro[id_libro]
            c["rank_popularidad"] = rank_popularidad_por_libro[id_libro]
            c["en_popularidad"] = 1
            agregados += 1

        genero = genero_por_usuario.get(id_lector)
        if genero is not None and genero in stats_por_genero:
            tabla_genero = stats_por_genero[genero]
            agregados = 0
            for rank, (id_libro, score) in enumerate(
                zip(tabla_genero["id_libro"], tabla_genero["score"])
            ):
                if agregados >= n_por_fuente:
                    break
                if id_libro in vistos:
                    continue
                c = candidatos.setdefault(id_libro, {})
                c["score_genero"] = score
                c["rank_genero"] = rank
                c["en_genero"] = 1
                agregados += 1

        n_usuario = n_interacciones_por_usuario.get(id_lector, 0)
        autores_leidos = n_libros_autor_leidos_por_usuario.get(id_lector, {})
        editoriales_leidas = n_libros_editorial_leidos_por_usuario.get(id_lector, {})
        anio_promedio_usuario = anio_edicion_promedio_por_usuario.get(id_lector)
        n_generos_distintos = n_generos_distintos_por_usuario.get(id_lector, 0)
        dias_desde_ultima = dias_desde_ultima_interaccion_por_usuario.get(
            id_lector, SENTINEL_DIAS_DESCONOCIDO
        )
        co_scores_usuario = co_scores_por_usuario.get(id_lector, {})
        frecuencias_genero_macro_usuario = frecuencia_genero_macro_por_usuario.get(id_lector, {})

        # Similitud de texto: un solo matmul chico (como mucho ~3*n_por_fuente
        # filas de tfidf_norm) contra el perfil de ESTE usuario -- nunca un
        # cruce usuario x catálogo completo (ver docstring de `_calcular_perfil_texto`).
        sim_resumen_por_candidato: dict = {}
        fila_usuario_texto = fila_por_usuario.get(id_lector)
        if tfidf_norm is not None and perfil_usuario_norm is not None and fila_usuario_texto is not None:
            ids_con_texto = [id_libro for id_libro in candidatos if id_libro in fila_por_libro_texto]
            if ids_con_texto:
                filas_texto = [fila_por_libro_texto[id_libro] for id_libro in ids_con_texto]
                perfil_usuario = perfil_usuario_norm[fila_usuario_texto]
                similitudes = tfidf_norm[filas_texto].multiply(perfil_usuario).sum(axis=1).A1
                sim_resumen_por_candidato = dict(zip(ids_con_texto, similitudes))

        for id_libro, f in candidatos.items():
            autor = autor_por_libro.get(id_libro)
            n_autor_leidos = autores_leidos.get(autor, 0) if pd.notna(autor) else 0

            editorial = editorial_por_libro.get(id_libro)
            n_editorial_leidos = editoriales_leidas.get(editorial, 0) if pd.notna(editorial) else 0
            n_libros_editorial_catalogo = n_libros_por_editorial.get(editorial, 0) if pd.notna(editorial) else 0

            anio_candidato = anio_edicion_por_libro.get(id_libro)
            if pd.notna(anio_candidato) and anio_promedio_usuario is not None:
                anio_edicion_dif = anio_candidato - anio_promedio_usuario
            else:
                anio_edicion_dif = 0.0

            columna_candidato = columna_por_libro.get(id_libro)
            score_coleido = co_scores_usuario.get(columna_candidato, 0.0) if columna_candidato is not None else 0.0

            genero_macro_candidato = genero_macro_por_libro.get(id_libro)
            popularidad_genero_macro = score_por_libro_genero_macro.get(id_libro, 0.0)
            frecuencia_genero_macro = (
                frecuencias_genero_macro_usuario.get(genero_macro_candidato, 0.0)
                if genero_macro_candidato is not None
                else 0.0
            )

            filas_resultado.append(
                {
                    "id_lector": id_lector,
                    "id_libro": id_libro,
                    "score_als": f.get("score_als", 0.0),
                    "rank_als": f.get("rank_als", n_por_fuente),
                    "en_als": f.get("en_als", 0),
                    "score_popularidad": f.get("score_popularidad", 0.0),
                    "rank_popularidad": f.get("rank_popularidad", n_por_fuente),
                    "en_popularidad": f.get("en_popularidad", 0),
                    "score_genero": f.get("score_genero", 0.0),
                    "rank_genero": f.get("rank_genero", n_por_fuente),
                    "en_genero": f.get("en_genero", 0),
                    "n_interacciones_libro": n_por_libro.get(id_libro, 0),
                    "n_interacciones_usuario": n_usuario,
                    "en_autor_leido": 1 if n_autor_leidos > 0 else 0,
                    "n_libros_autor_leidos": n_autor_leidos,
                    "anio_edicion_dif": anio_edicion_dif,
                    "n_generos_distintos_usuario": n_generos_distintos,
                    "dias_desde_ultima_interaccion_usuario": dias_desde_ultima,
                    "score_coleido": float(score_coleido),
                    "en_editorial_leida": 1 if n_editorial_leidos > 0 else 0,
                    "n_libros_editorial_leidos": n_editorial_leidos,
                    "sim_resumen_historial": float(sim_resumen_por_candidato.get(id_libro, 0.0)),
                    "popularidad_genero_macro_candidato": float(popularidad_genero_macro),
                    "frecuencia_genero_macro_usuario": float(frecuencia_genero_macro),
                    "n_libros_editorial_catalogo": n_libros_editorial_catalogo,
                }
            )

    columnas = ["id_lector", "id_libro"] + FEATURES
    return pd.DataFrame(filas_resultado, columns=columnas)


def armar_dataset_entrenamiento(
    candidatos_df: pd.DataFrame,
    etiquetas_df: pd.DataFrame,
    n_por_fuente: int = 150,
) -> tuple[pd.DataFrame, pd.Series, list]:
    """Arma (X, y, group) para `lightgbm.LGBMRanker.fit` a partir de los
    candidatos generados y las etiquetas reales (el "próximo libro" de
    cada usuario en `train_ranker`, columnas `id_lector`/`id_libro`).

    Si el libro-etiqueta de un usuario no aparece entre sus candidatos
    generados (cobertura incompleta de las 3 fuentes), se lo inyecta
    igual con features "ausente" (mismo sentinel que usa
    `generar_candidatos_con_features`) -- si no, ese usuario no aporta
    ningún positivo y el ranker nunca aprendería de él. `group` es la
    cantidad de filas por usuario, en el mismo orden que `X`/`y`.
    """
    candidatos_por_usuario = {
        id_lector: grupo for id_lector, grupo in candidatos_df.groupby("id_lector", sort=False)
    }

    grupos_filas = []
    tamanos_grupo = []
    for _, fila_etiqueta in etiquetas_df.iterrows():
        id_lector = fila_etiqueta["id_lector"]
        libro_objetivo = fila_etiqueta["id_libro"]

        grupo = candidatos_por_usuario.get(
            id_lector, pd.DataFrame(columns=["id_lector", "id_libro"] + FEATURES)
        ).copy()

        if libro_objetivo not in set(grupo["id_libro"]):
            fila_faltante = dict.fromkeys(FEATURES, 0)
            fila_faltante["rank_als"] = n_por_fuente
            fila_faltante["rank_popularidad"] = n_por_fuente
            fila_faltante["rank_genero"] = n_por_fuente
            fila_faltante["dias_desde_ultima_interaccion_usuario"] = SENTINEL_DIAS_DESCONOCIDO
            fila_faltante["id_lector"] = id_lector
            fila_faltante["id_libro"] = libro_objetivo
            grupo = pd.concat([grupo, pd.DataFrame([fila_faltante])], ignore_index=True)

        grupo = grupo.assign(y=(grupo["id_libro"] == libro_objetivo).astype(int))
        grupos_filas.append(grupo)
        tamanos_grupo.append(len(grupo))

    dataset = pd.concat(grupos_filas, ignore_index=True)
    return dataset[FEATURES], dataset["y"], tamanos_grupo


def fit_ranker(
    X: pd.DataFrame,
    y: pd.Series,
    group: list,
    X_eval: pd.DataFrame | None = None,
    y_eval: pd.Series | None = None,
    group_eval: list | None = None,
    **params,
) -> lgb.LGBMRanker:
    """Entrena un `LGBMRanker` (objetivo `lambdarank`) sobre el dataset
    armado por `armar_dataset_entrenamiento`.

    Hiperparámetros conservadores por default -- sin un sweep agresivo
    tipo optuna en esta primera versión, para no repetir el sobreajuste
    al proxy local que ya se vio con ALS (ver `experiments/bitacora.md`).
    Si se pasa un `X_eval`/`y_eval`/`group_eval`, se usa para early
    stopping (frena el boosting cuando deja de mejorar), no para buscar
    hiperparámetros.
    """
    defaults = dict(
        objective="lambdarank",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        random_state=42,
    )
    defaults.update(params)
    modelo = lgb.LGBMRanker(**defaults)

    fit_kwargs: dict = {}
    callbacks = []
    if X_eval is not None:
        fit_kwargs["eval_set"] = [(X_eval, y_eval)]
        fit_kwargs["eval_group"] = [group_eval]
        callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))

    modelo.fit(X, y, group=group, callbacks=callbacks or None, **fit_kwargs)
    return modelo


def recomendar_por_usuario(
    usuarios: list,
    modelo_ranker: lgb.LGBMRanker,
    candidatos_df: pd.DataFrame,
    ranking_global: list,
    libros_leidos: dict,
    k: int,
) -> dict:
    """Puntúa los candidatos de cada usuario con el ranker entrenado,
    ordena y devuelve el top-k. Fallback a `ranking_global` si a un
    usuario le faltan candidatos para completar k (mismo patrón que el
    resto de los modelos del proyecto)."""
    recomendaciones: dict = {}

    if len(candidatos_df):
        candidatos_df = candidatos_df.copy()
        candidatos_df["_score_ranker"] = modelo_ranker.predict(candidatos_df[FEATURES])
        for id_lector, grupo in candidatos_df.groupby("id_lector", sort=False):
            ordenado = grupo.sort_values("_score_ranker", ascending=False)
            recomendaciones[id_lector] = ordenado["id_libro"].tolist()

    for id_lector in usuarios:
        candidatos = recomendaciones.get(id_lector, [])
        if len(candidatos) < k:
            vistos = set(libros_leidos.get(id_lector, set())) | set(candidatos)
            extra = [libro for libro in ranking_global if libro not in vistos]
            candidatos = candidatos + extra[: k - len(candidatos)]
        recomendaciones[id_lector] = candidatos[:k]

    return recomendaciones


def evaluar_pipeline(
    interacciones: pd.DataFrame,
    libros: pd.DataFrame,
    seed: int,
    n_por_fuente: int = 150,
    lgbm_params: dict | None = None,
    k: int = 20,
) -> dict:
    """Corre el pipeline completo del ranker de dos etapas para un seed:
    split de **tres niveles** (`train_candidatos`/`train_ranker`/`test_final`,
    ver docstring del módulo), fit de las señales solo con
    `train_candidatos`, entrenamiento del `LGBMRanker` con las etiquetas
    de `train_ranker`, y evaluación de NDCG@k en `test_final`. También
    evalúa ALS solo sobre las mismas señales (mismo `train_candidatos`),
    para una comparación controlada -- aísla el efecto de agregar la capa
    de ranking, no el de tener más o menos datos de entrenamiento.

    `lgbm_params` se pasa tal cual a `fit_ranker` (`None` -> hiperparámetros
    conservadores por default). Existe para reusarse tanto desde
    `scripts/evaluate_ranker.py` (hiperparámetros fijos) como desde
    `scripts/tune_ranker.py` (búsqueda con optuna) sin duplicar esta
    lógica en cada script.

    Devuelve `{"ndcg_als": ..., "ndcg_ranker": ..., "modelo_ranker": ...}`
    -- el modelo entrenado se incluye para poder inspeccionar
    `feature_importances_` (ver `scripts/evaluate_ranker.py`) sin
    reentrenar.
    """
    train_candidatos_full, test_final = split_train_val(interacciones, n_val=1, seed=seed)
    train_candidatos, train_ranker = split_train_val(train_candidatos_full, n_val=1, seed=seed + 1000)

    libros_leidos_stage1 = libros_leidos_por_usuario(train_candidatos)
    n_interacciones_por_usuario = train_candidatos.groupby("id_lector").size().to_dict()
    stats_popularidad = fit_popularity(train_candidatos)
    ranking_global = stats_popularidad["id_libro"].tolist()
    stats_por_genero = fit_popularity_por_genero(train_candidatos, libros)
    genero_por_usuario = genero_preferido_por_usuario(train_candidatos, libros)
    modelo_als, matriz, fila_por_usuario, libros_por_columna = fit_als(train_candidatos)
    features_auxiliares = calcular_features_auxiliares(
        train_candidatos, libros, matriz, fila_por_usuario, libros_por_columna
    )

    args_candidatos = dict(
        modelo_als=modelo_als,
        matriz_usuario_libro=matriz,
        fila_por_usuario=fila_por_usuario,
        libros_por_columna=libros_por_columna,
        stats_popularidad=stats_popularidad,
        stats_por_genero=stats_por_genero,
        genero_por_usuario=genero_por_usuario,
        n_interacciones_por_usuario=n_interacciones_por_usuario,
        features_auxiliares=features_auxiliares,
        n_por_fuente=n_por_fuente,
    )

    usuarios_ranker = train_ranker["id_lector"].unique().tolist()
    candidatos_train_ranker = generar_candidatos_con_features(
        usuarios=usuarios_ranker, libros_leidos=libros_leidos_stage1, **args_candidatos
    )
    X, y, group = armar_dataset_entrenamiento(
        candidatos_train_ranker, train_ranker[["id_lector", "id_libro"]], n_por_fuente=n_por_fuente
    )
    modelo_ranker = fit_ranker(X, y, group, **(lgbm_params or {}))

    libros_leidos_hasta_ranker = libros_leidos_por_usuario(train_candidatos_full)
    usuarios_test = test_final["id_lector"].unique().tolist()

    candidatos_test = generar_candidatos_con_features(
        usuarios=usuarios_test, libros_leidos=libros_leidos_hasta_ranker, **args_candidatos
    )
    recs_ranker = recomendar_por_usuario(
        usuarios=usuarios_test,
        modelo_ranker=modelo_ranker,
        candidatos_df=candidatos_test,
        ranking_global=ranking_global,
        libros_leidos=libros_leidos_hasta_ranker,
        k=k,
    )
    ndcg_ranker = evaluar_ndcg_personalizado(test_final, recs_ranker, k)

    recs_als = _recomendar_als(
        usuarios=usuarios_test,
        modelo=modelo_als,
        matriz_usuario_libro=matriz,
        fila_por_usuario=fila_por_usuario,
        libros_por_columna=libros_por_columna,
        ranking_global=ranking_global,
        libros_leidos=libros_leidos_hasta_ranker,
        k=k,
    )
    ndcg_als = evaluar_ndcg_personalizado(test_final, recs_als, k)

    return {"ndcg_als": ndcg_als, "ndcg_ranker": ndcg_ranker, "modelo_ranker": modelo_ranker}
