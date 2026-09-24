"""Carga de datos y utilidades de split para el sistema de recomendación de libros."""

from __future__ import annotations

import sqlite3
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
