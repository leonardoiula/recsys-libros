"""Ranker de dos etapas: candidatos de ALS + popularidad por género +
popularidad global + libros de autores ya leídos + similitud de resumen +
co-lectura ítem-ítem (kNN), reordenados por un `LGBMRanker` (LightGBM,
objetivo `lambdarank`) entrenado para combinar esas seis señales mejor
de lo que cada una hace sola, más features auxiliares de autor/
editorial/año de edición/diversidad de género/recencia/co-lectura
ítem-ítem/popularidad y frecuencia de macro-género/señales cruzadas
lector↔libro (género declarado del lector, edad del lector al
publicarse el libro -- ver `calcular_features_auxiliares`).

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

import gc
import hashlib
import heapq
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from recsys.data import libros_leidos_por_usuario, split_train_val
from recsys.evaluation import evaluar_ndcg_personalizado, ndcg_at_k
from recsys.models.als import fit_als
from recsys.models.als import recomendar_por_usuario as _recomendar_als
from recsys.models.popularity import fit_popularity
from recsys.models.popularity_segmentada import (
    NACIMIENTO_SENTINEL,
    _normalizar_genero,
    fit_popularidad_por_genero_macro,
    fit_popularity_por_genero,
    fit_popularity_por_genero_lector,
    genero_lector_por_usuario,
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
    "popularidad_genero_lector_candidato",
    "frecuencia_genero_macro_por_genero_lector",
    "edad_lector_al_publicarse",
    "score_autor_candidato",
    "rank_autor_candidato",
    "en_autor_candidato",
    "score_resumen_candidato",
    "rank_resumen_candidato",
    "en_resumen_candidato",
    "score_coleido_candidato",
    "rank_coleido_candidato",
    "en_coleido_candidato",
    "n_libros_autor_leidos_reciente",
    "n_libros_editorial_leidos_reciente",
    "score_coleido_reciente",
    "sim_resumen_historial_reciente",
]

FUENTES_CANDIDATOS = frozenset({"als", "popularidad", "genero", "autor", "resumen", "coleido"})
"""Nombres válidos de fuente para el parámetro `fuentes_activas` de
`generar_candidatos_con_features`/`preparar_pipeline`/`preparar_pipeline_cacheado`.
Permite apagar una fuente entera (no solo sus features de tracking) para comparar
generadores de candidatos completos -- ver `scripts/comparar_generadores_pareado.py`."""

N_MAX_FEATURES_TFIDF = 20000
MIN_DF_TFIDF = 2
MAX_DF_TFIDF = 0.8

SENTINEL_DIAS_DESCONOCIDO = 99999

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"
"""Directorio default de `preparar_pipeline_cacheado` -- no se commitea (ver
`.gitignore`), mismo criterio que `outputs/`."""

TAMANO_LOTE_RESUMEN = 500
"""Usuarios procesados por lote en `_generar_candidatos_por_resumen`.
Un producto denso `usuarios x libros_con_resumen` (~48.320 libros) para
TODOS los usuarios a la vez sería demasiado grande -- se procesa en
lotes de este tamaño (~500 x 48.320 x 8 bytes ~ 194MB por lote,
liberado antes del siguiente) para acotar el pico de memoria. Mismo
espíritu que el tope de la fuente de autor y el descarte de
`n_por_fuente=500`: no repetir un problema de memoria ya visto dos
veces esta sesión (ver `experiments/bitacora.md`)."""


def _pesos_por_recencia(interacciones: pd.DataFrame) -> pd.Series:
    """Peso de descuento por posición en el historial de cada usuario:
    `peso = 1/log2(rank+2)`, con `rank` = posición de esa interacción
    ordenando por `fecha` DESCENDENTE dentro de su propio usuario (0 = la
    interacción más reciente de ese usuario) -- mismo descuento que ya usa
    `ndcg_at_k` en `evaluation.py`, elegido a propósito para no introducir
    una escala de tiempo nueva (días/vida media de un decaimiento
    exponencial) que habría que barrer/validar aparte (co-diseñado con el
    usuario, ver `experiments/decisiones.md`). Fechas inválidas se tratan
    como las MÁS ANTIGUAS del usuario (mismo criterio que
    `split_train_val`), nunca como las más recientes.

    Usada para dar features "recencia-ponderadas" (autor/editorial/
    co-lectura/similitud de resumen) que priorizan lo que el usuario leyó
    hace poco, en vez de poolear todo el historial parejo -- ver
    `calcular_features_auxiliares`.

    Devuelve una `pd.Series` alineada al índice de `interacciones` (no un
    array por posición), para poder sumarla agrupando por lo que haga
    falta sin depender de que el índice sea contiguo.
    """
    orden = pd.to_datetime(interacciones["fecha"], format="%d-%m-%Y", errors="coerce").fillna(pd.Timestamp.min)
    rank = orden.groupby(interacciones["id_lector"]).rank(method="first", ascending=False) - 1
    return 1.0 / np.log2(rank + 2)


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


def _calcular_perfil_texto(
    interacciones: pd.DataFrame,
    libros: pd.DataFrame,
    fila_por_usuario: dict,
    pesos_recencia: pd.Series | None = None,
) -> dict:
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

    Si se pasa `pesos_recencia` (`_pesos_por_recencia`, alineado al índice
    de `interacciones`), se arma además `perfil_usuario_reciente_norm`:
    el mismo centroide pero pesando cada lectura por qué tan reciente es
    en vez de parejo -- da `sim_resumen_historial_reciente`, la variante
    "gustos recientes" de `sim_resumen_historial` ("gustos de siempre").
    """
    con_resumen = libros.dropna(subset=["resumen"])
    con_resumen = con_resumen[con_resumen["resumen"].astype(str).str.strip() != ""]

    if con_resumen.empty:
        return {
            "tfidf_norm": None,
            "fila_por_libro_texto": {},
            "perfil_usuario_norm": None,
            "perfil_usuario_reciente_norm": None,
        }

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
        _peso_recencia=pesos_recencia if pesos_recencia is not None else 1.0,
    ).dropna(subset=["_fila_usuario", "_fila_texto"])

    filas_usuario = interacciones_con_texto["_fila_usuario"].astype(int)
    filas_texto = interacciones_con_texto["_fila_texto"].astype(int)

    def _perfil_normalizado(valores: np.ndarray):
        X = sp.csr_matrix((valores, (filas_usuario, filas_texto)), shape=(n_usuarios, tfidf_norm.shape[0]))
        perfil = X @ tfidf_norm
        normas = np.sqrt(perfil.multiply(perfil).sum(axis=1)).A1
        normas[normas == 0] = 1.0  # evita división por cero para usuarios sin lecturas con resumen
        return (sp.diags(1.0 / normas) @ perfil).tocsr()

    perfil_usuario_norm = _perfil_normalizado(np.ones(len(interacciones_con_texto)))
    perfil_usuario_reciente_norm = _perfil_normalizado(interacciones_con_texto["_peso_recencia"].to_numpy())

    return {
        "tfidf_norm": tfidf_norm.tocsr(),
        "fila_por_libro_texto": fila_por_libro_texto,
        "perfil_usuario_norm": perfil_usuario_norm,
        "perfil_usuario_reciente_norm": perfil_usuario_reciente_norm,
    }


def _generar_candidatos_por_resumen(
    tfidf_norm,
    perfil_usuario_norm,
    fila_por_libro_texto: dict,
    usuarios_con_als: list,
    filas: np.ndarray,
    n_por_fuente: int,
) -> tuple[dict, dict]:
    """Para cada usuario en `usuarios_con_als`, el top-`n_por_fuente` de
    libros de **todo el catálogo con resumen** (no solo los candidatos
    que ya trajeron otras fuentes) más similares a su perfil de lectura
    (mismo perfil TF-IDF que ya arma `_calcular_perfil_texto` para
    `sim_resumen_historial`).

    A diferencia de esa feature (que solo puntúa candidatos que ya
    llegaron de otra fuente), esto busca en todo el catálogo -- la única
    señal de las 6 fuentes que no depende de cuánta gente más leyó un
    libro, solo de su contenido. Motivada por medir que los libros
    objetivo que las otras 4 fuentes fallan en capturar son ~11x menos
    populares (mediana de interacciones) que los que sí capturan -- ver
    `experiments/modelo_actual.md`.

    Se procesa en lotes de `TAMANO_LOTE_RESUMEN` usuarios (ver docstring
    de esa constante) en vez de un solo producto denso
    `usuarios x libros_con_resumen` -- ya hubo dos problemas de memoria
    reales esta sesión (fuente de autor sin tope, `n_por_fuente=500`)
    por materializar de más.

    Devuelve `(ids_top_por_usuario, scores_top_por_usuario)`, ambos
    `{id_lector: [...]}` en orden de similitud descendente.
    """
    ids_top_por_usuario: dict = {}
    scores_top_por_usuario: dict = {}
    if tfidf_norm is None or perfil_usuario_norm is None or not usuarios_con_als:
        return ids_top_por_usuario, scores_top_por_usuario

    libro_por_columna_texto = [None] * len(fila_por_libro_texto)
    for id_libro, col in fila_por_libro_texto.items():
        libro_por_columna_texto[col] = id_libro

    tfidf_t = tfidf_norm.T.tocsr()
    for inicio in range(0, len(usuarios_con_als), TAMANO_LOTE_RESUMEN):
        usuarios_lote = usuarios_con_als[inicio : inicio + TAMANO_LOTE_RESUMEN]
        filas_lote = filas[inicio : inicio + TAMANO_LOTE_RESUMEN]
        similitudes = (perfil_usuario_norm[filas_lote] @ tfidf_t).toarray()

        for i, id_lector in enumerate(usuarios_lote):
            fila_sim = similitudes[i]
            n_top = min(n_por_fuente, fila_sim.shape[0])
            if n_top <= 0 or not fila_sim.any():
                continue
            idx_top = np.argpartition(-fila_sim, n_top - 1)[:n_top]
            idx_top = idx_top[np.argsort(-fila_sim[idx_top])]
            ids_top_por_usuario[id_lector] = [libro_por_columna_texto[c] for c in idx_top]
            scores_top_por_usuario[id_lector] = fila_sim[idx_top]

    return ids_top_por_usuario, scores_top_por_usuario


def calcular_features_auxiliares(
    interacciones: pd.DataFrame,
    libros: pd.DataFrame,
    lectores: pd.DataFrame,
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
    - `genero_lector_por_lector` / `score_por_libro_por_genero_lector` /
      `afinidad_genero_macro_por_genero_lector`: señales cruzadas
      lector↔libro. `lectores.genero` (Mujer/Hombre/desconocido -- OJO,
      NO es el género literario) segmenta tanto una popularidad bayesiana
      del candidato (mismo patrón que país/franja: el segmento lo define
      el usuario) como una afinidad de *cohorte* por macro-género (qué
      proporción de las interacciones de gente que declaró el mismo
      género que el usuario cae en el macro-género del candidato --
      distinto de `frecuencia_genero_macro_por_usuario`, que mira el
      historial *individual*, no el de la cohorte).
    - `nacimiento_por_lector`: año de nacimiento numérico de cada lector
      (`nacimiento` inválido o igual a `NACIMIENTO_SENTINEL` -- ver
      `popularity_segmentada.py` -- queda como `NaN`), para cruzar contra
      `anio_edicion_por_libro` y estimar la edad del lector cuando se
      publicó el candidato.
    - `n_libros_autor_leidos_reciente_por_usuario` / `n_libros_editorial_leidos_reciente_por_usuario` /
      `matriz_recencia` / `perfil_usuario_reciente_norm` (esta última
      dentro de `perfil_texto`): variantes "recencia-ponderadas" de las
      señales de arriba -- mismo dato, pero cada interacción pesa
      `1/log2(rank+2)` según qué tan reciente es dentro del historial del
      usuario (`_pesos_por_recencia`) en vez de contar/poolear todo el
      historial parejo. Alimentan `n_libros_autor_leidos_reciente`/
      `n_libros_editorial_leidos_reciente`/`score_coleido_reciente`/
      `sim_resumen_historial_reciente` en `generar_candidatos_con_features`.

    `matriz_usuario_libro`, `fila_por_usuario` y `libros_por_columna` son
    los que ya devuelve `fit_als` sobre el mismo `train_candidatos` --
    se reusan acá para no re-fitear ALS ni duplicar el índice de usuarios.
    """
    metadata = libros.set_index("id_libro")
    autor_por_libro = metadata["autor"].to_dict()
    editorial_por_libro = metadata["editorial"].to_dict()
    anio_edicion_por_libro = pd.to_numeric(metadata["anio_edicion"], errors="coerce").to_dict()
    genero_por_libro = _normalizar_genero(metadata["genero"]).to_dict()

    # Pesos de recencia (`_pesos_por_recencia`): 1/log2(rank+2) por
    # interacción, rank=posición desde la más reciente del usuario --
    # alimentan las variantes "recientes" de autor/editorial/co-lectura/
    # resumen (más abajo), que priorizan lo que el usuario leyó hace poco
    # en vez de poolear todo el historial parejo.
    pesos_recencia = _pesos_por_recencia(interacciones)

    con_autor = interacciones.assign(autor=interacciones["id_libro"].map(autor_por_libro), _peso=pesos_recencia)
    n_libros_autor_leidos_por_usuario: dict = {}
    n_libros_autor_leidos_reciente_por_usuario: dict = {}
    agregado_autor = con_autor.dropna(subset=["autor"]).groupby(["id_lector", "autor"]).agg(
        n=("autor", "size"), peso=("_peso", "sum")
    )
    for (id_lector, autor), fila in agregado_autor.iterrows():
        n_libros_autor_leidos_por_usuario.setdefault(id_lector, {})[autor] = int(fila["n"])
        n_libros_autor_leidos_reciente_por_usuario.setdefault(id_lector, {})[autor] = float(fila["peso"])

    con_editorial = interacciones.assign(
        editorial=interacciones["id_libro"].map(editorial_por_libro), _peso=pesos_recencia
    )
    n_libros_editorial_leidos_por_usuario: dict = {}
    n_libros_editorial_leidos_reciente_por_usuario: dict = {}
    agregado_editorial = con_editorial.dropna(subset=["editorial"]).groupby(["id_lector", "editorial"]).agg(
        n=("editorial", "size"), peso=("_peso", "sum")
    )
    for (id_lector, editorial), fila in agregado_editorial.iterrows():
        n_libros_editorial_leidos_por_usuario.setdefault(id_lector, {})[editorial] = int(fila["n"])
        n_libros_editorial_leidos_reciente_por_usuario.setdefault(id_lector, {})[editorial] = float(fila["peso"])

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
    perfil_texto = _calcular_perfil_texto(interacciones, libros, fila_por_usuario, pesos_recencia)

    # Matriz usuario x libro "reciente" (mismo shape/índice que
    # matriz_usuario_libro, pero con `pesos_recencia` como valores en vez
    # de rating/binario) -- da `score_coleido_reciente` en
    # `generar_candidatos_con_features`: mismo cálculo que `score_coleido`
    # (`X_batch @ cooc`) pero pesando más los libros que el usuario leyó
    # hace poco, sin recalcular `cooc` (que sigue siendo una estadística
    # poblacional estable, no algo que tenga sentido "hacer reciente" por
    # usuario).
    con_indices_recencia = interacciones.assign(
        _fila=interacciones["id_lector"].map(fila_por_usuario),
        _columna=interacciones["id_libro"].map(columna_por_libro),
        _peso=pesos_recencia,
    ).dropna(subset=["_fila", "_columna"])
    matriz_recencia = sp.csr_matrix(
        (
            con_indices_recencia["_peso"].to_numpy(),
            (con_indices_recencia["_fila"].astype(int), con_indices_recencia["_columna"].astype(int)),
        ),
        shape=matriz_usuario_libro.shape,
    )

    # Señales cruzadas lector<->libro (ver docstring): género DECLARADO
    # del lector (no confundir con género literario) segmenta tanto una
    # popularidad bayesiana (mismo patrón que país/franja) como una
    # afinidad de cohorte por macro-género.
    genero_lector_por_lector = genero_lector_por_usuario(lectores)
    stats_por_genero_lector = fit_popularity_por_genero_lector(interacciones, lectores)
    score_por_libro_por_genero_lector = {
        genero_lector: tabla.set_index("id_libro")["score"].to_dict()
        for genero_lector, tabla in stats_por_genero_lector.items()
    }

    con_genero_lector = interacciones.assign(
        genero_lector=interacciones["id_lector"].map(genero_lector_por_lector),
        genero_macro=interacciones["id_libro"].map(genero_macro_por_libro),
    ).dropna(subset=["genero_macro"])
    conteos_genero_lector = (
        con_genero_lector.groupby(["genero_lector", "genero_macro"]).size().rename("n").reset_index()
    )
    total_por_genero_lector = conteos_genero_lector.groupby("genero_lector")["n"].transform("sum")
    conteos_genero_lector["frecuencia"] = conteos_genero_lector["n"] / total_por_genero_lector
    afinidad_genero_macro_por_genero_lector: dict = {}
    for genero_lector, genero_macro, frecuencia in zip(
        conteos_genero_lector["genero_lector"], conteos_genero_lector["genero_macro"], conteos_genero_lector["frecuencia"]
    ):
        afinidad_genero_macro_por_genero_lector.setdefault(genero_lector, {})[genero_macro] = frecuencia

    # Edad del lector cuando se publicó el candidato (nacimiento inválido
    # o sentinel -> NaN, mismo criterio que `franja_nacimiento_por_usuario`).
    nacimiento_num = pd.to_numeric(lectores.set_index("id_lector")["nacimiento"], errors="coerce")
    nacimiento_num = nacimiento_num.where(nacimiento_num != NACIMIENTO_SENTINEL)
    nacimiento_por_lector = nacimiento_num.to_dict()

    return {
        "autor_por_libro": autor_por_libro,
        "anio_edicion_por_libro": anio_edicion_por_libro,
        "n_libros_autor_leidos_por_usuario": n_libros_autor_leidos_por_usuario,
        "n_libros_autor_leidos_reciente_por_usuario": n_libros_autor_leidos_reciente_por_usuario,
        "editorial_por_libro": editorial_por_libro,
        "n_libros_editorial_leidos_por_usuario": n_libros_editorial_leidos_por_usuario,
        "n_libros_editorial_leidos_reciente_por_usuario": n_libros_editorial_leidos_reciente_por_usuario,
        "n_libros_por_editorial": n_libros_por_editorial,
        "anio_edicion_promedio_por_usuario": anio_edicion_promedio_por_usuario,
        "n_generos_distintos_por_usuario": n_generos_distintos_por_usuario,
        "dias_desde_ultima_interaccion_por_usuario": dias_desde_ultima_interaccion_por_usuario,
        "cooc": cooc,
        "columna_por_libro": columna_por_libro,
        "matriz_recencia": matriz_recencia,
        "genero_macro_por_libro": genero_macro_por_libro,
        "score_por_libro_genero_macro": score_por_libro_genero_macro,
        "frecuencia_genero_macro_por_usuario": frecuencia_genero_macro_por_usuario,
        "genero_lector_por_lector": genero_lector_por_lector,
        "score_por_libro_por_genero_lector": score_por_libro_por_genero_lector,
        "afinidad_genero_macro_por_genero_lector": afinidad_genero_macro_por_genero_lector,
        "nacimiento_por_lector": nacimiento_por_lector,
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
    n_por_autor: int = 20,
    fuentes_activas: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Arma, para cada usuario, la unión de candidatos de las seis fuentes
    (ALS, popularidad por género, popularidad global, libros de autores ya
    leídos, similitud de resumen, co-lectura ítem-ítem) con sus features.

    Reusa las estructuras que ya arman `fit_popularity` (`stats_popularidad`),
    `fit_popularity_por_genero` (`stats_por_genero`) y `genero_preferido_por_usuario`
    -- no duplica esa lógica. Cada candidato queda con `score_*`/`rank_*`
    (posición real dentro de esa fuente, no posición entre los candidatos
    finalmente elegidos) por fuente que lo propuso, y `en_*` (1/0) indicando
    qué fuentes lo propusieron. Un candidato que no vino de una fuente
    queda con score 0.0 y rank `n_por_fuente` (sentinel: "justo afuera de
    la ventana de esa fuente") -- la fuente de autor usa su propia ventana
    `n_por_autor` como sentinel, no `n_por_fuente`.

    La fuente de autor (`experiments/modelo_actual.md`, sección
    "Recomendación: ¿cambiar de paradigma?" -- 28.6% de los libros
    objetivo son de un autor que el usuario ya leyó, y antes de esto esa
    señal solo existía como *feature*, nunca proponía candidatos nuevos
    por sí sola): para cada autor que el usuario ya leyó
    (`n_libros_autor_leidos_por_usuario`), hasta `n_por_autor` libros sin
    leer de ese autor, rankeados por el score de popularidad GLOBAL (no
    se refittea un score bayesiano por autor -- la mayoría de los
    autores tiene muy pocas interacciones para un shrinkage propio
    confiable). `score_autor_candidato`/`rank_autor_candidato`/
    `en_autor_candidato` son distintos de `en_autor_leido`/
    `n_libros_autor_leidos` (que miden el historial del usuario con ese
    autor, sin importar qué fuente propuso el candidato).

    La fuente de resumen (ver `_generar_candidatos_por_resumen`): top-
    `n_por_fuente` libros de todo el catálogo con resumen más similares
    al perfil de lectura (TF-IDF) del usuario -- a diferencia de las
    otras 4 fuentes, no depende de cuánta gente más leyó un libro, solo
    de su contenido. `score_resumen_candidato`/`rank_resumen_candidato`/
    `en_resumen_candidato` son distintos de `sim_resumen_historial` (que
    solo puntúa candidatos que ya llegaron de otra fuente, nunca
    propone candidatos nuevos por sí sola).

    La fuente de co-lectura ítem-ítem (6ª fuente): el mismo cálculo que
    ya arma `score_coleido` (`co_scores_por_usuario`, el batch
    `X_batch @ cooc` sobre la matriz de co-ocurrencia de
    `_calcular_cooccurrencia`) trae, para cada usuario, un score de
    co-lectura contra **todo el catálogo indexado por ALS** -- hasta
    ahora solo se usaba para puntuar candidatos que ya habían llegado de
    otra fuente. Acá se toma además el top-`n_por_fuente` de ese mismo
    cálculo como candidatos nuevos (excluyendo `vistos`), igual que las
    demás fuentes. `score_coleido_candidato`/`rank_coleido_candidato`/
    `en_coleido_candidato` son distintos de `score_coleido` (que sigue
    puntuando cualquier candidato, sin importar su fuente) -- mismo
    patrón que autor/resumen: la feature de tracking documenta qué
    fuente propuso el candidato, no reemplaza a la feature existente.
    Hereda la misma limitación que ALS y co-lectura hoy: solo alcanza a
    usuarios con fila en la matriz de ALS (`usuarios_con_als`).

    `features_auxiliares` es el dict que arma `calcular_features_auxiliares`
    (autor, año de edición, diversidad de género, recencia). Un candidato
    sin dato conocido (autor/año de edición ausente, o usuario sin
    historial suficiente) queda con el sentinel correspondiente (0 para
    conteos/diferencias, `SENTINEL_DIAS_DESCONOCIDO` para recencia) -- no
    se imputa a ciegas.

    `fuentes_activas` (default `None` = las 6) restringe qué fuentes proponen
    candidatos NUEVOS -- ver `FUENTES_CANDIDATOS`. Apaga solo el bloque que agrega
    candidatos de esa fuente, no las features "de historial" que ya existían antes
    de que esa fuente propusiera candidatos (`score_coleido`/`sim_resumen_historial`
    se siguen calculando para cualquier candidato, venga de donde venga, igual que
    hoy) -- pensado para comparar generadores de candidatos completos manteniendo
    todo lo demás igual (mismo split/seed, mismas features auxiliares). Ver
    `scripts/comparar_generadores_pareado.py`.

    Devuelve un DataFrame largo con columnas `id_lector`, `id_libro` +
    `FEATURES`.
    """
    if fuentes_activas is None:
        fuentes_activas = FUENTES_CANDIDATOS
    elif not fuentes_activas <= FUENTES_CANDIDATOS:
        raise ValueError(
            f"fuentes_activas contiene valores inválidos: {sorted(fuentes_activas - FUENTES_CANDIDATOS)}"
            f" -- válidas: {sorted(FUENTES_CANDIDATOS)}"
        )

    autor_por_libro = features_auxiliares["autor_por_libro"]
    anio_edicion_por_libro = features_auxiliares["anio_edicion_por_libro"]
    n_libros_autor_leidos_por_usuario = features_auxiliares["n_libros_autor_leidos_por_usuario"]
    n_libros_autor_leidos_reciente_por_usuario = features_auxiliares.get("n_libros_autor_leidos_reciente_por_usuario", {})
    editorial_por_libro = features_auxiliares.get("editorial_por_libro", {})
    n_libros_editorial_leidos_por_usuario = features_auxiliares.get("n_libros_editorial_leidos_por_usuario", {})
    n_libros_editorial_leidos_reciente_por_usuario = features_auxiliares.get(
        "n_libros_editorial_leidos_reciente_por_usuario", {}
    )
    n_libros_por_editorial = features_auxiliares.get("n_libros_por_editorial", {})
    anio_edicion_promedio_por_usuario = features_auxiliares["anio_edicion_promedio_por_usuario"]
    n_generos_distintos_por_usuario = features_auxiliares["n_generos_distintos_por_usuario"]
    dias_desde_ultima_interaccion_por_usuario = features_auxiliares["dias_desde_ultima_interaccion_por_usuario"]
    cooc = features_auxiliares.get("cooc")
    columna_por_libro = features_auxiliares.get("columna_por_libro", {})
    matriz_recencia = features_auxiliares.get("matriz_recencia")
    tfidf_norm = features_auxiliares.get("tfidf_norm")
    fila_por_libro_texto = features_auxiliares.get("fila_por_libro_texto", {})
    perfil_usuario_norm = features_auxiliares.get("perfil_usuario_norm")
    perfil_usuario_reciente_norm = features_auxiliares.get("perfil_usuario_reciente_norm")
    genero_macro_por_libro = features_auxiliares.get("genero_macro_por_libro", {})
    score_por_libro_genero_macro = features_auxiliares.get("score_por_libro_genero_macro", {})
    frecuencia_genero_macro_por_usuario = features_auxiliares.get("frecuencia_genero_macro_por_usuario", {})
    genero_lector_por_lector = features_auxiliares.get("genero_lector_por_lector", {})
    score_por_libro_por_genero_lector = features_auxiliares.get("score_por_libro_por_genero_lector", {})
    afinidad_genero_macro_por_genero_lector = features_auxiliares.get("afinidad_genero_macro_por_genero_lector", {})
    nacimiento_por_lector = features_auxiliares.get("nacimiento_por_lector", {})
    n_por_libro = stats_popularidad.set_index("id_libro")["n"].to_dict()
    score_popularidad_por_libro = stats_popularidad.set_index("id_libro")["score"].to_dict()
    ranking_global_ids = stats_popularidad["id_libro"].tolist()
    rank_popularidad_por_libro = {libro: i for i, libro in enumerate(ranking_global_ids)}

    # Fuente de autor: hasta n_por_autor libros por autor, rankeados por
    # popularidad GLOBAL (no un score bayesiano por autor -- la mayoría
    # de los autores tiene muy pocas interacciones). `ranking_global_ids`
    # ya viene ordenado por score descendente (ver `fit_popularity`), así
    # que un solo pase alcanza para quedarse con el top-n_por_autor de
    # cada autor sin ordenar de nuevo.
    libros_por_autor_ordenados: dict = {}
    for id_libro in ranking_global_ids:
        autor = autor_por_libro.get(id_libro)
        if autor is None or pd.isna(autor):
            continue
        lista = libros_por_autor_ordenados.setdefault(autor, [])
        if len(lista) < n_por_autor:
            lista.append(id_libro)

    usuarios_con_als = [u for u in usuarios if u in fila_por_usuario]
    # `filas` se calcula sin importar qué fuentes estén activas -- co-lectura y
    # resumen la reusan igual que ALS (misma indexación de la matriz de ALS).
    filas = np.array([fila_por_usuario[u] for u in usuarios_con_als]) if usuarios_con_als else None

    ids_items_als: dict = {}
    scores_als: dict = {}
    if usuarios_con_als and "als" in fuentes_activas:
        ids_items, scores = modelo_als.recommend(
            filas, matriz_usuario_libro[filas], N=n_por_fuente, filter_already_liked_items=True
        )
        for id_lector, fila_ids, fila_scores in zip(usuarios_con_als, ids_items, scores):
            ids_items_als[id_lector] = [libros_por_columna[idx] for idx in fila_ids]
            scores_als[id_lector] = list(fila_scores)

    # Co-lectura: acotado a los mismos usuarios_con_als (misma fila que ALS)
    # y a un matmul disperso sobre ese batch -- nunca sobre los ~10k
    # usuarios completos, para no explotar en memoria (ver docstring de
    # `_calcular_cooccurrencia`). Se calcula SIEMPRE que haya `cooc` (sin
    # importar si "coleido" está en `fuentes_activas`): alimenta la feature
    # `score_coleido` de candidatos de cualquier fuente, no solo los propios.
    co_scores_por_usuario: dict = {}
    if usuarios_con_als and cooc is not None:
        X_batch = sp.csr_matrix(matriz_usuario_libro[filas] > 0, dtype=np.int32)
        co_scores_batch = (X_batch @ cooc).tocsr()
        for id_lector, fila_row in zip(usuarios_con_als, co_scores_batch):
            co_scores_por_usuario[id_lector] = dict(zip(fila_row.indices, fila_row.data))

    # Variante "reciente" de co-lectura (ver `matriz_recencia` en
    # `calcular_features_auxiliares`): mismo `cooc` poblacional, pero el
    # lado del usuario pesa cada libro que leyó por qué tan reciente es
    # en vez de 1/0 -- dos libros co-leídos hace mucho pesan menos que dos
    # co-leídos hace poco, aunque `cooc[i,j]` en sí sea la misma.
    co_scores_recencia_por_usuario: dict = {}
    if usuarios_con_als and cooc is not None and matriz_recencia is not None:
        co_scores_recencia_batch = (matriz_recencia[filas] @ cooc).tocsr()
        for id_lector, fila_row in zip(usuarios_con_als, co_scores_recencia_batch):
            co_scores_recencia_por_usuario[id_lector] = dict(zip(fila_row.indices, fila_row.data))

    # Candidatos por similitud de resumen (5ª fuente, ver docstring de
    # `_generar_candidatos_por_resumen`): busca en TODO el catálogo con
    # resumen, no solo entre los candidatos que ya trajeron las otras 4
    # fuentes -- a diferencia de `sim_resumen_historial` (más abajo).
    ids_resumen_por_usuario, scores_resumen_por_usuario = ({}, {})
    if usuarios_con_als and "resumen" in fuentes_activas:
        ids_resumen_por_usuario, scores_resumen_por_usuario = _generar_candidatos_por_resumen(
            tfidf_norm,
            perfil_usuario_norm,
            fila_por_libro_texto,
            usuarios_con_als,
            filas,
            n_por_fuente,
        )

    # Acumular por columna (no una lista de dicts por candidato): con
    # ~9k usuarios de test * ~700 candidatos, una lista de millones de
    # dicts de 39 claves cada uno se volvió lo bastante pesada como para
    # que la consolidación final de `pd.DataFrame` fallara por memoria
    # ("unable to allocate ... MiB") pese a tener RAM de sobra en la
    # máquina -- síntoma de fragmentación por acumular millones de objetos
    # Python de vida larga. Construir cada `fila` igual que antes (mismo
    # dict, mismo código) pero volcarla enseguida a listas por columna
    # evita mantener esos millones de dicts vivos a la vez.
    columnas = ["id_lector", "id_libro"] + FEATURES
    columnas_datos: dict = {columna: [] for columna in columnas}
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

        if "popularidad" in fuentes_activas:
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
        if genero is not None and genero in stats_por_genero and "genero" in fuentes_activas:
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

        # Tope total (no solo por autor): un usuario que leyó cientos de
        # autores distintos podía aportar miles de candidatos de esta
        # fuente sola (medido: hasta 5.304 candidatos para un usuario en
        # un solo seed, contra ~450 típicos) -- un problema real de
        # memoria/rendimiento en la corrida completa, no solo teórico.
        # Se prioriza a los autores que MÁS leyó el usuario (no el orden
        # arbitrario del dict) hasta `n_por_fuente` candidatos en total,
        # mismo criterio de ventana que las otras 3 fuentes.
        if "autor" in fuentes_activas:
            autores_leidos_conteo = n_libros_autor_leidos_por_usuario.get(id_lector, {})
            autores_ordenados = sorted(autores_leidos_conteo, key=lambda a: -autores_leidos_conteo[a])
            agregados_autor = 0
            for autor in autores_ordenados:
                if agregados_autor >= n_por_fuente:
                    break
                for rank_autor, id_libro in enumerate(libros_por_autor_ordenados.get(autor, [])):
                    if agregados_autor >= n_por_fuente:
                        break
                    if id_libro in vistos:
                        continue
                    c = candidatos.setdefault(id_libro, {})
                    c["score_autor_candidato"] = score_popularidad_por_libro.get(id_libro, 0.0)
                    c["rank_autor_candidato"] = rank_autor
                    c["en_autor_candidato"] = 1
                    agregados_autor += 1

        for rank_resumen, (id_libro, score) in enumerate(
            zip(ids_resumen_por_usuario.get(id_lector, []), scores_resumen_por_usuario.get(id_lector, []))
        ):
            if id_libro in vistos:
                continue
            c = candidatos.setdefault(id_libro, {})
            c["score_resumen_candidato"] = float(score)
            c["rank_resumen_candidato"] = rank_resumen
            c["en_resumen_candidato"] = 1

        # Candidatos por co-lectura ítem-ítem (6ª fuente, ver docstring):
        # mismo score que ya arma `co_scores_por_usuario` para
        # `score_coleido` (batch `X_batch @ cooc`), pero acá se toma el
        # top-n_por_fuente de TODO ese cálculo como candidatos nuevos, no
        # solo para puntuar lo que ya trajo otra fuente. `heapq.nlargest`
        # evita ordenar el dict completo (puede tener miles de entradas
        # para usuarios con mucho historial) cuando solo hace falta el
        # top-n_por_fuente.
        co_scores_usuario = co_scores_por_usuario.get(id_lector, {})
        if "coleido" in fuentes_activas:
            top_coleido = heapq.nlargest(n_por_fuente, co_scores_usuario.items(), key=lambda kv: kv[1])
            for rank_coleido, (columna_candidato, score) in enumerate(top_coleido):
                id_libro = libros_por_columna[columna_candidato]
                if id_libro in vistos:
                    continue
                c = candidatos.setdefault(id_libro, {})
                c["score_coleido_candidato"] = float(score)
                c["rank_coleido_candidato"] = rank_coleido
                c["en_coleido_candidato"] = 1

        n_usuario = n_interacciones_por_usuario.get(id_lector, 0)
        autores_leidos = n_libros_autor_leidos_por_usuario.get(id_lector, {})
        autores_leidos_reciente = n_libros_autor_leidos_reciente_por_usuario.get(id_lector, {})
        editoriales_leidas = n_libros_editorial_leidos_por_usuario.get(id_lector, {})
        editoriales_leidas_reciente = n_libros_editorial_leidos_reciente_por_usuario.get(id_lector, {})
        co_scores_recencia_usuario = co_scores_recencia_por_usuario.get(id_lector, {})
        anio_promedio_usuario = anio_edicion_promedio_por_usuario.get(id_lector)
        n_generos_distintos = n_generos_distintos_por_usuario.get(id_lector, 0)
        dias_desde_ultima = dias_desde_ultima_interaccion_por_usuario.get(
            id_lector, SENTINEL_DIAS_DESCONOCIDO
        )
        frecuencias_genero_macro_usuario = frecuencia_genero_macro_por_usuario.get(id_lector, {})
        genero_lector_usuario = genero_lector_por_lector.get(id_lector)
        score_por_libro_genero_lector_usuario = score_por_libro_por_genero_lector.get(genero_lector_usuario, {})
        afinidad_genero_macro_usuario = afinidad_genero_macro_por_genero_lector.get(genero_lector_usuario, {})
        nacimiento_usuario = nacimiento_por_lector.get(id_lector)

        # Similitud de texto: un solo matmul chico (como mucho ~3*n_por_fuente
        # filas de tfidf_norm) contra el perfil de ESTE usuario -- nunca un
        # cruce usuario x catálogo completo (ver docstring de `_calcular_perfil_texto`).
        sim_resumen_por_candidato: dict = {}
        sim_resumen_reciente_por_candidato: dict = {}
        fila_usuario_texto = fila_por_usuario.get(id_lector)
        if tfidf_norm is not None and perfil_usuario_norm is not None and fila_usuario_texto is not None:
            ids_con_texto = [id_libro for id_libro in candidatos if id_libro in fila_por_libro_texto]
            if ids_con_texto:
                # Un solo matmul disperso (`@`, no `.multiply(...).sum(axis=1)`)
                # apilando el perfil "de siempre" y el "reciente" en una sola
                # matriz de 2 filas -- da las dos similitudes de un saque en
                # vez de dos productos dispersos intermedios por usuario
                # (con ~9k usuarios de test, duplicar esa asignación en un
                # loop así de caliente llegó a agotar la memoria del proceso
                # -- "unable to allocate 378 KiB" pese a tener RAM de sobra,
                # síntoma de fragmentación por muchas asignaciones chicas).
                filas_texto = [fila_por_libro_texto[id_libro] for id_libro in ids_con_texto]
                tfidf_candidatos = tfidf_norm[filas_texto]
                perfil_usuario = perfil_usuario_norm[fila_usuario_texto]

                if perfil_usuario_reciente_norm is not None:
                    perfiles = sp.vstack([perfil_usuario, perfil_usuario_reciente_norm[fila_usuario_texto]])
                    similitudes = (tfidf_candidatos @ perfiles.T).toarray()
                    sim_resumen_por_candidato = dict(zip(ids_con_texto, similitudes[:, 0]))
                    sim_resumen_reciente_por_candidato = dict(zip(ids_con_texto, similitudes[:, 1]))
                else:
                    similitudes = (tfidf_candidatos @ perfil_usuario.T).toarray().ravel()
                    sim_resumen_por_candidato = dict(zip(ids_con_texto, similitudes))

        for id_libro, f in candidatos.items():
            autor = autor_por_libro.get(id_libro)
            n_autor_leidos = autores_leidos.get(autor, 0) if pd.notna(autor) else 0
            n_autor_leidos_reciente = autores_leidos_reciente.get(autor, 0.0) if pd.notna(autor) else 0.0

            editorial = editorial_por_libro.get(id_libro)
            n_editorial_leidos = editoriales_leidas.get(editorial, 0) if pd.notna(editorial) else 0
            n_editorial_leidos_reciente = (
                editoriales_leidas_reciente.get(editorial, 0.0) if pd.notna(editorial) else 0.0
            )
            n_libros_editorial_catalogo = n_libros_por_editorial.get(editorial, 0) if pd.notna(editorial) else 0

            anio_candidato = anio_edicion_por_libro.get(id_libro)
            if pd.notna(anio_candidato) and anio_promedio_usuario is not None:
                anio_edicion_dif = anio_candidato - anio_promedio_usuario
            else:
                anio_edicion_dif = 0.0

            columna_candidato = columna_por_libro.get(id_libro)
            score_coleido = co_scores_usuario.get(columna_candidato, 0.0) if columna_candidato is not None else 0.0
            score_coleido_reciente = (
                co_scores_recencia_usuario.get(columna_candidato, 0.0) if columna_candidato is not None else 0.0
            )

            genero_macro_candidato = genero_macro_por_libro.get(id_libro)
            popularidad_genero_macro = score_por_libro_genero_macro.get(id_libro, 0.0)
            frecuencia_genero_macro = (
                frecuencias_genero_macro_usuario.get(genero_macro_candidato, 0.0)
                if genero_macro_candidato is not None
                else 0.0
            )

            popularidad_genero_lector = score_por_libro_genero_lector_usuario.get(id_libro, 0.0)
            frecuencia_genero_macro_genero_lector = (
                afinidad_genero_macro_usuario.get(genero_macro_candidato, 0.0)
                if genero_macro_candidato is not None
                else 0.0
            )
            if pd.notna(anio_candidato) and pd.notna(nacimiento_usuario):
                edad_lector_al_publicarse = anio_candidato - nacimiento_usuario
            else:
                edad_lector_al_publicarse = 0.0

            fila = {
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
                "popularidad_genero_lector_candidato": float(popularidad_genero_lector),
                "frecuencia_genero_macro_por_genero_lector": float(frecuencia_genero_macro_genero_lector),
                "edad_lector_al_publicarse": float(edad_lector_al_publicarse),
                "score_autor_candidato": f.get("score_autor_candidato", 0.0),
                "rank_autor_candidato": f.get("rank_autor_candidato", n_por_autor),
                "en_autor_candidato": f.get("en_autor_candidato", 0),
                "score_resumen_candidato": f.get("score_resumen_candidato", 0.0),
                "rank_resumen_candidato": f.get("rank_resumen_candidato", n_por_fuente),
                "en_resumen_candidato": f.get("en_resumen_candidato", 0),
                "score_coleido_candidato": f.get("score_coleido_candidato", 0.0),
                "rank_coleido_candidato": f.get("rank_coleido_candidato", n_por_fuente),
                "en_coleido_candidato": f.get("en_coleido_candidato", 0),
                "n_libros_autor_leidos_reciente": float(n_autor_leidos_reciente),
                "n_libros_editorial_leidos_reciente": float(n_editorial_leidos_reciente),
                "score_coleido_reciente": float(score_coleido_reciente),
                "sim_resumen_historial_reciente": float(sim_resumen_reciente_por_candidato.get(id_libro, 0.0)),
            }
            for columna, valor in fila.items():
                columnas_datos[columna].append(valor)

    # Cada columna de FEATURES se convierte a un array de numpy ANTES de
    # armar el DataFrame (no se deja que pandas infiera el dtype de una
    # lista de Python) y todas a `float32` en vez del `float64`/`int64`
    # mixto que salía antes: reduce a la mitad el tamaño del bloque
    # numérico consolidado y evita que pandas necesite dos bloques
    # separados (uno por familia de dtype) -- con ~5.5M filas x 39
    # columnas, la consolidación mixta llegó a fallar por
    # `ArrayMemoryError` pese a tener RAM de sobra en la máquina (síntoma
    # de fragmentación, no de falta de memoria real). LightGBM no pierde
    # nada de precisión útil con float32 (ranks/conteos/scores de este
    # proyecto están lejos del límite de representación exacta de ese
    # tipo).
    for columna in FEATURES:
        columnas_datos[columna] = np.asarray(columnas_datos[columna], dtype=np.float32)
    return pd.DataFrame(columnas_datos)


def armar_dataset_entrenamiento(
    candidatos_df: pd.DataFrame,
    etiquetas_df: pd.DataFrame,
    n_por_fuente: int = 150,
    n_por_autor: int = 20,
) -> tuple[pd.DataFrame, pd.Series, list]:
    """Arma (X, y, group) para `lightgbm.LGBMRanker.fit` a partir de los
    candidatos generados y las etiquetas reales (el "próximo libro" de
    cada usuario en `train_ranker`, columnas `id_lector`/`id_libro`).

    Si el libro-etiqueta de un usuario no aparece entre sus candidatos
    generados (cobertura incompleta de las 6 fuentes), se lo inyecta
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
            fila_faltante["rank_autor_candidato"] = n_por_autor
            fila_faltante["rank_resumen_candidato"] = n_por_fuente
            fila_faltante["rank_coleido_candidato"] = n_por_fuente
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


def preparar_pipeline(
    interacciones: pd.DataFrame,
    libros: pd.DataFrame,
    lectores: pd.DataFrame,
    seed: int,
    n_por_fuente: int = 150,
    n_por_autor: int = 20,
    k: int = 20,
    fuentes_activas: frozenset[str] | None = None,
    refit_para_test: bool = False,
) -> dict:
    """Arma todo lo que necesita el pipeline del ranker para un seed,
    **excepto** entrenar el `LGBMRanker` en sí -- eso queda para
    `evaluar_con_params`, que recibe el dict que devuelve esta función
    (el "contexto") y lo reusa.

    Por qué está separado de `evaluar_con_params`: medido con el set de
    23 features (`experiments/bitacora.md`, sección "Separar armado de
    candidatos de tuneo de LightGBM"), una corrida de esta función tarda
    ~280-300s (fit de ALS/popularidad/género, `calcular_features_auxiliares`
    -- TF-IDF/co-lectura/macro-género --, y sobre todo las dos llamadas a
    `generar_candidatos_con_features`, la parte más cara con ~130s cada
    una), mientras que entrenar el `LGBMRanker` (`fit_ranker`) tarda
    ~22s. Nada de lo que arma esta función depende de los hiperparámetros
    de LightGBM -- son las mismas señales/candidatos/dataset sin importar
    qué configuración se vaya a probar. `scripts/tune_ranker.py` explota
    justo eso: arma el contexto **una sola vez por seed** y prueba muchas
    configuraciones de LightGBM contra el mismo contexto vía
    `evaluar_con_params`, en vez de repetir los ~280-300s en cada trial.

    Reproduce el split de **tres niveles** (`train_candidatos`/
    `train_ranker`/`test_final`, ver docstring del módulo) y fitea las
    señales solo con `train_candidatos`, igual que antes.

    Devuelve un dict con todo lo que `evaluar_con_params` necesita:
    `X`/`y`/`group` (dataset de entrenamiento del ranker), `candidatos_test`
    (candidatos de `test_final` ya con features), `ranking_global`,
    `libros_leidos_hasta_ranker`, `usuarios_test`, `test_final` (para
    calcular NDCG@k), `ndcg_als` (score de ALS solo -- tampoco depende de
    `lgbm_params`, se calcula acá una sola vez) y `k`.

    `fuentes_activas` (default `None` = las 6) se pasa tal cual a las dos
    llamadas de `generar_candidatos_con_features` -- ver docstring de esa
    función y `FUENTES_CANDIDATOS`.

    `refit_para_test` (default `False`, sin cambios de comportamiento):
    con `True`, después de entrenar el ranker sobre las señales fiteadas
    en `train_candidatos` (eso no cambia -- evita que el ranker vea, como
    features, scores calculados con la misma etiqueta que tiene que
    predecir), se REFITEAN ALS/popularidad/género/`calcular_features_auxiliares`
    sobre `train_candidatos_full` (=`train_candidatos`+`train_ranker`,
    todo menos `test_final`) y ese refit se usa para generar
    `candidatos_test`/`ndcg_als` -- mismo patrón que tendría producción
    (`submit.py`, que no tiene un `test_final` que reservar y podría
    refitear sobre absolutamente todos los datos antes de generar la
    entrega real). Ver `scripts/comparar_refit_etapa1.py`.
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
        train_candidatos, libros, lectores, matriz, fila_por_usuario, libros_por_columna
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
        n_por_autor=n_por_autor,
        fuentes_activas=fuentes_activas,
    )

    usuarios_ranker = train_ranker["id_lector"].unique().tolist()
    candidatos_train_ranker = generar_candidatos_con_features(
        usuarios=usuarios_ranker, libros_leidos=libros_leidos_stage1, **args_candidatos
    )
    X, y, group = armar_dataset_entrenamiento(
        candidatos_train_ranker,
        train_ranker[["id_lector", "id_libro"]],
        n_por_fuente=n_por_fuente,
        n_por_autor=n_por_autor,
    )
    # `candidatos_train_ranker` (millones de filas) ya no hace falta -- lo
    # que importa de acá en más es `X`/`y`/`group` (mucho más chico, solo
    # columnas numéricas). Liberarlo antes de la segunda llamada, tan
    # pesada como la primera, a `generar_candidatos_con_features` reduce
    # el pico de memoria del proceso (encontrado en la práctica: la
    # segunda llamada llegó a fallar por `ArrayMemoryError` con RAM de
    # sobra en la máquina -- síntoma de fragmentación, no de falta de
    # memoria real).
    del candidatos_train_ranker
    gc.collect()

    libros_leidos_hasta_ranker = libros_leidos_por_usuario(train_candidatos_full)
    usuarios_test = test_final["id_lector"].unique().tolist()

    if refit_para_test:
        # Ya no hace falta el fit sobre train_candidatos (candidatos_train_ranker
        # ya se usó para armar X/y/group) -- liberarlo antes de refitear
        # evita tener las dos versiones de ALS/features_auxiliares (cada
        # una con su propia matriz de co-ocurrencia/TF-IDF) vivas a la vez.
        del args_candidatos, features_auxiliares, matriz, modelo_als, fila_por_usuario, libros_por_columna
        gc.collect()

        stats_popularidad_test = fit_popularity(train_candidatos_full)
        ranking_global_test = stats_popularidad_test["id_libro"].tolist()
        stats_por_genero_test = fit_popularity_por_genero(train_candidatos_full, libros)
        genero_por_usuario_test = genero_preferido_por_usuario(train_candidatos_full, libros)
        modelo_als_test, matriz_test, fila_por_usuario_test, libros_por_columna_test = fit_als(
            train_candidatos_full
        )
        features_auxiliares_test = calcular_features_auxiliares(
            train_candidatos_full, libros, lectores, matriz_test, fila_por_usuario_test, libros_por_columna_test
        )
        args_candidatos_test = dict(
            modelo_als=modelo_als_test,
            matriz_usuario_libro=matriz_test,
            fila_por_usuario=fila_por_usuario_test,
            libros_por_columna=libros_por_columna_test,
            stats_popularidad=stats_popularidad_test,
            stats_por_genero=stats_por_genero_test,
            genero_por_usuario=genero_por_usuario_test,
            n_interacciones_por_usuario=train_candidatos_full.groupby("id_lector").size().to_dict(),
            features_auxiliares=features_auxiliares_test,
            n_por_fuente=n_por_fuente,
            n_por_autor=n_por_autor,
            fuentes_activas=fuentes_activas,
        )
    else:
        ranking_global_test = ranking_global
        modelo_als_test, matriz_test, fila_por_usuario_test, libros_por_columna_test = (
            modelo_als,
            matriz,
            fila_por_usuario,
            libros_por_columna,
        )
        args_candidatos_test = args_candidatos

    candidatos_test = generar_candidatos_con_features(
        usuarios=usuarios_test, libros_leidos=libros_leidos_hasta_ranker, **args_candidatos_test
    )

    recs_als = _recomendar_als(
        usuarios=usuarios_test,
        modelo=modelo_als_test,
        matriz_usuario_libro=matriz_test,
        fila_por_usuario=fila_por_usuario_test,
        libros_por_columna=libros_por_columna_test,
        ranking_global=ranking_global_test,
        libros_leidos=libros_leidos_hasta_ranker,
        k=k,
    )
    ndcg_als = evaluar_ndcg_personalizado(test_final, recs_als, k)

    return {
        "X": X,
        "y": y,
        "group": group,
        "candidatos_test": candidatos_test,
        "ranking_global": ranking_global_test,
        "libros_leidos_hasta_ranker": libros_leidos_hasta_ranker,
        "usuarios_test": usuarios_test,
        "test_final": test_final,
        "ndcg_als": ndcg_als,
        "recs_als": recs_als,
        "k": k,
    }


def evaluar_con_params(contexto: dict, lgbm_params: dict | None = None) -> dict:
    """Entrena un `LGBMRanker` con `lgbm_params` sobre el `contexto` que
    arma `preparar_pipeline` (ver ese docstring para el porqué de la
    separación) y evalúa NDCG@k en `test_final` -- la parte barata del
    pipeline (~22s medido con 23 features), pensada para llamarse muchas
    veces con distintos `lgbm_params` contra el mismo `contexto`/seed sin
    repetir el armado de candidatos.

    Devuelve `{"ndcg_als": ..., "ndcg_ranker": ..., "modelo_ranker": ...,
    "recs_ranker": ...}` -- `recs_ranker` (recomendaciones por usuario, ya
    filtradas/completadas a `k`) se expone además del NDCG agregado para
    poder recalcular diagnósticos por usuario (ej.
    `evaluation.evaluar_ndcg_ponderado_por_actividad`) sin tener que
    recomputar el ranking.
    """
    modelo_ranker = fit_ranker(contexto["X"], contexto["y"], contexto["group"], **(lgbm_params or {}))

    recs_ranker = recomendar_por_usuario(
        usuarios=contexto["usuarios_test"],
        modelo_ranker=modelo_ranker,
        candidatos_df=contexto["candidatos_test"],
        ranking_global=contexto["ranking_global"],
        libros_leidos=contexto["libros_leidos_hasta_ranker"],
        k=contexto["k"],
    )
    ndcg_ranker = evaluar_ndcg_personalizado(contexto["test_final"], recs_ranker, contexto["k"])

    return {
        "ndcg_als": contexto["ndcg_als"],
        "ndcg_ranker": ndcg_ranker,
        "modelo_ranker": modelo_ranker,
        "recs_ranker": recs_ranker,
    }


def evaluar_pipeline(
    interacciones: pd.DataFrame,
    libros: pd.DataFrame,
    lectores: pd.DataFrame,
    seed: int,
    n_por_fuente: int = 150,
    n_por_autor: int = 20,
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

    Es un atajo de conveniencia (`preparar_pipeline` + `evaluar_con_params`)
    para el caso de **una sola configuración por seed** -- así la usa
    `scripts/evaluate_ranker.py`. Si hace falta probar *varias*
    configuraciones de LightGBM contra el mismo seed (`scripts/tune_ranker.py`),
    llamar a `preparar_pipeline` una vez y reusar ese contexto en varias
    llamadas a `evaluar_con_params` es mucho más barato que llamar a esta
    función una vez por configuración -- ver el docstring de
    `preparar_pipeline` para el detalle de costos.

    `lgbm_params` se pasa tal cual a `fit_ranker` (`None` -> hiperparámetros
    conservadores por default).

    Devuelve `{"ndcg_als": ..., "ndcg_ranker": ..., "modelo_ranker": ...}`
    -- el modelo entrenado se incluye para poder inspeccionar
    `feature_importances_` (ver `scripts/evaluate_ranker.py`) sin
    reentrenar.
    """
    contexto = preparar_pipeline(
        interacciones, libros, lectores, seed, n_por_fuente=n_por_fuente, n_por_autor=n_por_autor, k=k
    )
    return evaluar_con_params(contexto, lgbm_params)


def preparar_pipeline_cacheado(
    interacciones: pd.DataFrame,
    libros: pd.DataFrame,
    lectores: pd.DataFrame,
    seed: int,
    n_por_fuente: int = 150,
    n_por_autor: int = 20,
    k: int = 20,
    fuentes_activas: frozenset[str] | None = None,
    refit_para_test: bool = False,
    cache_dir: str | Path | None = None,
) -> dict:
    """Wrapper de `preparar_pipeline` que cachea el contexto resultante a
    disco (`pickle`), para no repetir los ~280-300s de fit de ALS/
    popularidad/género + `calcular_features_auxiliares` (TF-IDF/co-lectura/
    macro-género) + `generar_candidatos_con_features` cada vez que se
    quiere repetir o ajustar una comparación sobre el mismo seed/config --
    ver `scripts/comparar_features_pareado.py`, `scripts/recall_candidatos.py`,
    `scripts/comparar_generadores_pareado.py`.

    La clave de caché combina `seed`/`n_por_fuente`/`n_por_autor`/`k`/
    `fuentes_activas` con un hash corto del *código fuente* de este módulo
    (`ranker.py`): cualquier cambio en la lógica de generación de
    candidatos/features invalida el caché automáticamente, sin depender de
    acordarse de bumpear una versión a mano. También incluye la cantidad de
    filas de `interacciones`/`libros`/`lectores` como red de seguridad
    barata -- OJO: esto NO detecta un reemplazo de `data/raw/data.db` con
    datos de igual tamaño pero contenido distinto; en ese caso hay que
    borrar `cache_dir` a mano.

    Antes de picklear, las columnas `id_lector`/`id_libro` de
    `candidatos_test`/`test_final` se convierten a `category` (ver
    `_comprimir_para_cache`): sin esto, un `pickle.dump` directo del
    contexto pesaba **3-4 GB** (medido con `n_por_fuente=150` -- millones de
    filas de candidatos con esos dos strings repetidos sin deduplicar) y
    tardaba más en recargarse de lo que tardaba recalcular. `category`
    deduplica cada string único una sola vez; el contexto que devuelve esta
    función (en cache-hit o cache-miss) queda idéntico al que devolvería
    `preparar_pipeline` directamente -- la conversión es un detalle interno
    del cacheo, no algo que le importe al resto del pipeline.

    `cache_dir` default `CACHE_DIR` (`data/cache/`, gitignored).
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    hash_codigo = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    fuentes_label = "todas" if fuentes_activas is None else "+".join(sorted(fuentes_activas))
    refit_label = "refit" if refit_para_test else "sinrefit"
    nombre = (
        f"ranker_ctx_seed{seed}_nf{n_por_fuente}_na{n_por_autor}_k{k}"
        f"_fuentes-{fuentes_label}_{refit_label}"
        f"_n{len(interacciones)}-{len(libros)}-{len(lectores)}"
        f"_{hash_codigo}.pkl"
    )
    ruta = cache_dir / nombre

    if ruta.exists():
        with open(ruta, "rb") as f:
            return _descomprimir_de_cache(pickle.load(f))

    contexto = preparar_pipeline(
        interacciones,
        libros,
        lectores,
        seed,
        n_por_fuente=n_por_fuente,
        n_por_autor=n_por_autor,
        k=k,
        fuentes_activas=fuentes_activas,
        refit_para_test=refit_para_test,
    )
    with open(ruta, "wb") as f:
        pickle.dump(_comprimir_para_cache(contexto), f, protocol=pickle.HIGHEST_PROTOCOL)
    return contexto


_COLUMNAS_ID_CACHE = ("id_lector", "id_libro")
_CLAVES_DATAFRAME_CACHE = ("candidatos_test", "test_final")


def _comprimir_para_cache(contexto: dict) -> dict:
    """Copia `contexto` convirtiendo `id_lector`/`id_libro` de
    `candidatos_test`/`test_final` a `category` -- ver docstring de
    `preparar_pipeline_cacheado` para el porqué. No modifica el `contexto`
    original (el que se devuelve al llamador en un cache-miss sigue con
    los dtypes normales)."""
    comprimido = dict(contexto)
    for clave in _CLAVES_DATAFRAME_CACHE:
        df = comprimido.get(clave)
        if df is None:
            continue
        columnas = [c for c in _COLUMNAS_ID_CACHE if c in df.columns]
        if columnas:
            comprimido[clave] = df.astype({c: "category" for c in columnas})
    return comprimido


def _descomprimir_de_cache(contexto: dict) -> dict:
    """Inverso de `_comprimir_para_cache`, aplicado tras `pickle.load` --
    devuelve `id_lector`/`id_libro` a su dtype normal (string) para que el
    contexto se comporte igual venga de un cache-hit o de un cálculo
    fresco."""
    for clave in _CLAVES_DATAFRAME_CACHE:
        df = contexto.get(clave)
        if df is None:
            continue
        columnas = [c for c in _COLUMNAS_ID_CACHE if c in df.columns and isinstance(df[c].dtype, pd.CategoricalDtype)]
        if columnas:
            contexto[clave] = df.astype({c: "str" for c in columnas})
    return contexto


def ndcg_por_usuario(ctx: dict, features: list[str] | None = None) -> dict:
    """Entrena un `LGBMRanker` sobre `features` (subconjunto de columnas de
    `ctx["X"]`, default: todas) y devuelve NDCG@k por usuario de
    `ctx["usuarios_test"]` -- SIN promediar, para poder comparar dos
    configuraciones (dos subconjuntos de `FEATURES`, o dos generadores de
    candidatos con distinto `fuentes_activas`) con un test PAREADO por
    usuario en vez de comparar promedios independientes -- mucho más poder
    estadístico (ver `scripts/comparar_features_pareado.py`,
    `scripts/comparar_generadores_pareado.py`).

    `ctx` es el dict que arma `preparar_pipeline`/`preparar_pipeline_cacheado`.
    """
    features = list(ctx["X"].columns) if features is None else features
    modelo = fit_ranker(ctx["X"][features], ctx["y"], ctx["group"])
    candidatos_por_usuario = {u: g for u, g in ctx["candidatos_test"].groupby("id_lector", sort=False)}
    relevantes_por_usuario = ctx["test_final"].groupby("id_lector")["id_libro"].agg(set).to_dict()
    libros_leidos = ctx["libros_leidos_hasta_ranker"]
    ranking_global = ctx["ranking_global"]
    k = ctx["k"]

    resultado = {}
    for id_lector in ctx["usuarios_test"]:
        grupo = candidatos_por_usuario.get(id_lector)
        if grupo is None or len(grupo) == 0:
            recomendados = []
        else:
            scores = modelo.predict(grupo[features])
            recomendados = list(grupo["id_libro"].to_numpy()[np.argsort(-scores)][:k])
        if len(recomendados) < k:
            vistos = set(libros_leidos.get(id_lector, set())) | set(recomendados)
            extra = [libro for libro in ranking_global if libro not in vistos]
            recomendados = recomendados + extra[: k - len(recomendados)]
        resultado[id_lector] = ndcg_at_k(recomendados, relevantes_por_usuario.get(id_lector, set()), k)
    return resultado


def recall_de_candidatos(ctx: dict) -> float:
    """Fracción de usuarios de `ctx["test_final"]` cuyo libro objetivo está
    entre los candidatos de `ctx["candidatos_test"]` -- el TECHO absoluto
    de NDCG@k que puede lograr el reranking con ese set de candidatos
    (ver `scripts/recall_candidatos.py`, que tenía esta misma lógica
    inline)."""
    candidatos_por_usuario = ctx["candidatos_test"].groupby("id_lector")["id_libro"].agg(set).to_dict()
    test_final = ctx["test_final"]
    hits = sum(
        1
        for _, fila in test_final.iterrows()
        if fila["id_libro"] in candidatos_por_usuario.get(fila["id_lector"], set())
    )
    return hits / len(test_final)
