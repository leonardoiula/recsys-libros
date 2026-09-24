"""Carga de datos y utilidades de split para el sistema de recomendación de libros."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "data.db"


def _read_table(table: str, db_path: Path | str = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(f"SELECT * FROM {table}", con)


def load_interacciones(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Carga la tabla interacciones(id_lector, id_libro, fecha, rating)."""
    return _read_table("interacciones", db_path)


def load_lectores(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Carga la tabla lectores(id_lector, nombre, genero, vive_en, nacimiento)."""
    return _read_table("lectores", db_path)


def load_libros(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Carga la tabla libros(id_libro, titulo, autor, genero, editorial, anio_edicion, isbn, resumen, img_src)."""
    return _read_table("libros", db_path)


def normalizar_texto(valor) -> str | None:
    """Normaliza un campo de texto libre (autor, editorial, título, isbn) para poder
    agrupar variantes que son la misma entidad real pero difieren en acentuación,
    mayúsculas, espacios o puntuación -- ej. `"GARCÍA MÁRQUEZ, GABRIEL"` /
    `"GARCIA MARQUEZ, GABRIEL"` / `"Garc a M rquez, Gabriel "` deben agrupar juntos.

    Minúsculas + `NFKD` sin acentos (mismo mecanismo que `_quitar_acentos` de
    `popularity_segmentada.py`, pero acá además se sacan signos de puntuación y se
    colapsa espacio interno repetido -- confirmado con los datos reales que hace
    falta para este caso: hay variantes que solo difieren en un espacio doble
    (`"EDICIONES  B"` vs `"EDICIONES B"`) o en un símbolo (`"ARROBA@BOOKS"` vs
    `"ARROBABOOKS"`) que acentos+mayúsculas por sí solos no unen.

    Devuelve `None` para nulos o strings que normalizan a vacío (no hay entidad que
    agrupar) -- nunca `""`, para que `dropna`/`pd.isna` sigan funcionando igual que con
    el dato crudo en el resto del pipeline.
    """
    if pd.isna(valor):
        return None
    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    texto = re.sub(r"[^a-z0-9 ]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto or None


def canonicalizar_libros_duplicados(
    libros: pd.DataFrame, interacciones: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fusiona filas de `libros` que son ediciones duplicadas del mismo libro bajo
    `id_libro` distintos: mismo `isbn` normalizado, o mismo `titulo`+`autor`
    normalizados (`normalizar_texto`) -- encadenado transitivamente vía union-find
    (si A comparte isbn con B y B comparte título+autor con C, A/B/C quedan en un
    solo grupo aunque A y C no compartan ninguna clave entre sí directamente).

    Confirmado contra los datos reales (auditoría de esta sesión, ver
    `experiments/estado_del_arte.md`): son duplicados genuinos -- reimpresiones o
    ediciones distintas del mismo libro (ej. "EL PRINCIPITO" con 3 ISBN distintos,
    "A 33.000 PIES" con el mismo ISBN pero id_libro repetido) -- 293 grupos, 0.46%
    de los libros del catálogo pero 6.79% de las interacciones (los libros
    populares son justamente los que acumulan reimpresiones). Ninguna muestra
    inspeccionada mezcla libros realmente distintos bajo esta regla.

    Por grupo, el id_libro CANÓNICO es el de mayor cantidad de interacciones en
    `interacciones` (desempate: menor `id_libro` en orden lexicográfico, para que
    el resultado sea determinístico) -- ahí es donde conviene concentrar la señal
    de popularidad/co-lectura/ALS que hoy queda repartida entre 2-3 ids
    equivalentes. Los metadatos del canónico se completan con los del resto del
    grupo cuando el canónico los tiene nulos (ej. un `resumen` presente en una
    edición y ausente en otra) -- nunca al revés.

    Devuelve `(libros, interacciones)`: `libros` con una fila por grupo (se
    descartan las filas no-canónicas) e `interacciones` con `id_libro` remapeado
    al canónico de su grupo. Si tras el remapeo un mismo usuario queda con dos
    interacciones sobre el mismo id_libro (leyó dos ediciones distintas del mismo
    libro), se deduplica quedándose con la fecha MÁS TEMPRANA -- mismo criterio que
    `split_train_val`/`split_temporal_global`: nunca inventarle al usuario una
    interacción más reciente de la que realmente se puede sostener con el dato.

    No muta `data/raw/data.db` ni los DataFrames recibidos -- transformación
    explícita que el caller aplica después de `load_libros`/`load_interacciones`,
    no un default de esas funciones (mismo criterio "opt-in" que `fuentes_activas`/
    `BM25_ALS` en `ranker.py`: se valida con un test pareado antes de volverla
    default de ningún script).
    """
    libros = libros.copy()
    libros["_isbn_norm"] = libros["isbn"].apply(normalizar_texto)
    libros["_titulo_norm"] = libros["titulo"].apply(normalizar_texto)
    libros["_autor_norm"] = libros["autor"].apply(normalizar_texto)
    libros["_ta_key"] = libros["_titulo_norm"].fillna("") + "|" + libros["_autor_norm"].fillna("")

    parent = {id_libro: id_libro for id_libro in libros["id_libro"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for _, grupo in libros[libros["_isbn_norm"].notna()].groupby("_isbn_norm"):
        ids = grupo["id_libro"].tolist()
        for otro in ids[1:]:
            union(ids[0], otro)

    con_titulo_autor = libros[libros["_titulo_norm"].notna() & libros["_autor_norm"].notna()]
    for _, grupo in con_titulo_autor.groupby("_ta_key"):
        ids = grupo["id_libro"].tolist()
        if len(ids) > 1:
            for otro in ids[1:]:
                union(ids[0], otro)

    libros["_grupo"] = libros["id_libro"].map(find)

    n_interacciones_por_libro = interacciones["id_libro"].value_counts()
    libros["_n_interacciones"] = libros["id_libro"].map(n_interacciones_por_libro).fillna(0)
    # Orden dentro de cada grupo: canónico primero (más interacciones, desempate por
    # id_libro) -- el GroupBy.first() de abajo toma el primer valor no nulo por
    # columna en ESTE orden, así que el resultado siempre prioriza el valor del
    # canónico.
    libros = libros.sort_values(["_grupo", "_n_interacciones", "id_libro"], ascending=[True, False, True])

    columnas_meta = [c for c in libros.columns if not c.startswith("_") and c != "id_libro"]
    # GroupBy.first(): toma, por columna, el primer valor NO NULO de cada grupo
    # (skipna=True es el default) -- con las filas ya ordenadas canónico-primero,
    # esto ES "el dato del canónico, completado desde el resto del grupo si el
    # canónico lo tiene nulo" en una sola operación vectorizada (evita el
    # `.apply(lambda ...)` por grupo, ~100x más lento sobre el catálogo real y,
    # peor, sin garantía de quedar alineado por posición con un `.first()`
    # calculado aparte -- versión anterior de esta función tenía ese bug).
    agrupado = libros.groupby("_grupo", sort=False)
    libros_canon = agrupado[["id_libro"] + columnas_meta].first().reset_index(drop=True)

    id_canonico_por_id_original = libros.set_index("id_libro")["_grupo"].map(agrupado["id_libro"].first())

    interacciones = interacciones.copy()
    # .fillna(id_libro original): un id_libro de interacciones que no tiene fila
    # correspondiente en libros (gap de integridad referencial preexistente y
    # ajeno a esta función -- confirmado contra los datos reales, 70 ids/212 filas,
    # ver experiments/estado_del_arte.md) no tiene grupo que asignarle. Sin este
    # fallback, `.map()` lo dejaría en NaN -- lo correcto es dejarlo tal cual
    # (mismo comportamiento que tiene HOY sin canonicalizar), no inventarle un
    # id_libro nuevo ni convertir un id "huérfano pero válido" en NaN (peor: NaN
    # colisiona con cualquier otro NaN como si fuera el mismo libro en el dedup
    # de abajo).
    interacciones["id_libro"] = interacciones["id_libro"].map(id_canonico_por_id_original).fillna(
        interacciones["id_libro"]
    )
    fechas = pd.to_datetime(interacciones["fecha"], format="%d-%m-%Y", errors="coerce")
    # na_position="last": entre dos filas que se fusionan en la misma (id_lector,
    # id_libro), preferir una fecha real (la más temprana) por sobre una fecha no
    # parseable -- una fila con fecha real es siempre más informativa para las
    # features de recencia que una que solo se sabe "vieja por convención" (ver
    # split_train_val), incluso si ese criterio de "NaT = más antigua" es el
    # correcto para desempatar DENTRO del historial normal de un usuario.
    interacciones = (
        interacciones.assign(_fecha=fechas)
        .sort_values("_fecha", na_position="last")
        .drop_duplicates(subset=["id_lector", "id_libro"], keep="first")
        .drop(columns="_fecha")
        .reset_index(drop=True)
    )

    return libros_canon, interacciones


def split_train_val(
    interacciones: pd.DataFrame,
    n_val: int = 1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split leave-one-out **temporal** por usuario.

    Agrupa por id_lector y retiene para validación las `n_val`
    interacciones más *recientes* de cada usuario (según `fecha`),
    dejando el resto en train. Nunca deja a un usuario sin interacciones
    en train: si `n_val` implicaría vaciar el train de un usuario, se
    limita a dejarle al menos una interacción.

    Se ordena por fecha en vez de retener una muestra aleatoria porque se
    confirmó empíricamente que un split aleatorio filtra información del
    futuro del usuario hacia train: sobre el mismo modelo (ALS), pasar de
    split aleatorio a split temporal bajó el NDCG@20 local de 0.260068 a
    0.122789 sin cambiar nada más, evidenciando ese leakage (ver
    `experiments/legacy/bitacora.md`). Además se usa un `n_val` fijo en vez de
    una fracción proporcional a la actividad de cada usuario: con
    `frac_val` proporcional, un usuario con mucho historial terminaba con
    muchos más libros "relevantes" simultáneos en validación que uno
    liviano, inflando su NDCG de forma dispareja e irrepresentativa del
    escenario real de "predecir la próxima lectura" (un libro a la vez).

    Un porcentaje mínimo de filas (~0.0002%, ver EDA) tiene `fecha` no
    parseable; esas quedan con fecha nula y se tratan como las más
    antiguas del usuario (nunca se las manda a val "a ciegas" como si
    fueran recientes).
    """
    if n_val < 1:
        raise ValueError("n_val debe ser >= 1")

    rng = np.random.default_rng(seed)
    fechas = pd.to_datetime(interacciones["fecha"], format="%d-%m-%Y", errors="coerce")

    train_idx: list[int] = []
    val_idx: list[int] = []

    for _, grupo in interacciones.groupby("id_lector", sort=False):
        idx = grupo.index.to_numpy().copy()
        rng.shuffle(idx)  # desempata determinísticamente fechas iguales o nulas
        idx_ordenado = fechas.loc[idx].sort_values(kind="stable", na_position="first").index.to_numpy()

        n_val_efectivo = min(n_val, len(idx_ordenado) - 1)
        if n_val_efectivo == 0:
            train_idx.extend(idx_ordenado)
        else:
            train_idx.extend(idx_ordenado[:-n_val_efectivo])
            val_idx.extend(idx_ordenado[-n_val_efectivo:])

    train = interacciones.loc[train_idx].reset_index(drop=True)
    val = interacciones.loc[val_idx].reset_index(drop=True)
    return train, val


def split_temporal_global(
    interacciones: pd.DataFrame,
    fecha_corte: str | pd.Timestamp,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal **GLOBAL**: un solo corte de calendario para todo el
    dataset, a diferencia de `split_train_val` (leave-one-out **por
    usuario**, cada uno con su propio punto de corte).

    `fecha_corte`, si es un string, se parsea `DD-MM-YYYY` -- el mismo
    formato que la columna `fecha` en todo el resto del proyecto
    (`interacciones["fecha"]`, siempre `%d-%m-%Y`). OJO: NO pasar un string
    a `pd.Timestamp(...)`/`pd.to_datetime(...)` sin ese formato explícito
    acá -- su parser genérico asume `MM-DD-YYYY` (convención US) para
    fechas ambiguas, así que p.ej. `"01-07-2024"` se leería como 7 de enero
    en vez de 1 de julio. Este bug ya se cometió una vez armando el primer
    script que usa esta función (`scripts/evaluate_global_cutoff.py`) --
    ver `experiments/estado_del_arte.md`.

    `train` = todas las interacciones con `fecha <= fecha_corte`. `val` = la
    interacción MÁS TEMPRANA de cada usuario con `fecha > fecha_corte` (si
    tiene alguna) -- su primer "próximo libro" después del corte, análogo a
    un solo objetivo por usuario igual que `split_train_val(n_val=1)`, pero
    fijando el corte en el calendario global en vez de en el historial
    propio de cada usuario.

    Existe para medir cuánto de la brecha NDCG local-vs-Kaggle se explica
    por una diferencia de PROTOCOLO, no de modelo: `split_train_val` deja
    fluir señal de "futuro" hacia `train_candidatos` de validación local
    (ALS/popularidad/co-lectura se fitean sobre toda la población sin
    ningún corte de fecha, así que para un usuario mediano -- cuyo propio
    corte cae ~2018 -- casi un tercio de TODO el dataset es posterior a su
    punto de validación). Kaggle evalúa específicamente a usuarios activos
    hasta el final del rango de fechas del dataset (mediana de última
    interacción 2024-10-25 en `ejemplo.csv` vs 2018-01-01 en la población
    general), donde ese tipo de señal casi no existe. Ver
    `experiments/estado_del_arte.md`, sección "Validación con corte
    temporal global".

    A diferencia de `split_train_val`, acá NO hay garantía de que todo
    usuario aparezca en `val` -- usuarios sin ninguna interacción posterior
    a `fecha_corte` no tienen un "próximo libro" que predecir bajo este
    corte y quedan enteros en `train`. Es intencional: Kaggle tampoco evalúa
    a todo el mundo, solo a quien sigue activo cerca del corte real.

    Mismo criterio que `split_train_val` para fechas no parseables (quedan
    en `train`, nunca en `val` -- no se puede saber si son posteriores al
    corte) y para desempatar fechas iguales dentro de un mismo usuario
    (shuffle determinístico por `seed` antes de un sort estable).
    """
    fechas = pd.to_datetime(interacciones["fecha"], format="%d-%m-%Y", errors="coerce")
    fecha_corte = (
        pd.to_datetime(fecha_corte, format="%d-%m-%Y") if isinstance(fecha_corte, str) else pd.Timestamp(fecha_corte)
    )
    es_posterior = fechas > fecha_corte  # NaT -> False: nunca va a val

    train = interacciones.loc[~es_posterior].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    val_idx: list[int] = []
    posteriores = interacciones.loc[es_posterior]
    for _, grupo in posteriores.groupby("id_lector", sort=False):
        idx = grupo.index.to_numpy().copy()
        rng.shuffle(idx)  # desempata determinísticamente fechas iguales dentro del usuario
        idx_ordenado = fechas.loc[idx].sort_values(kind="stable").index.to_numpy()
        val_idx.append(idx_ordenado[0])

    val = interacciones.loc[val_idx].reset_index(drop=True)
    return train, val


def libros_leidos_por_usuario(interacciones: pd.DataFrame) -> dict:
    """Devuelve {id_lector: {id_libro, ...}} con los libros leídos por cada usuario."""
    return interacciones.groupby("id_lector")["id_libro"].agg(set).to_dict()
