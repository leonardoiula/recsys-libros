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
    `experiments/bitacora.md`). Además se usa un `n_val` fijo en vez de
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


def libros_leidos_por_usuario(interacciones: pd.DataFrame) -> dict:
    """Devuelve {id_lector: {id_libro, ...}} con los libros leídos por cada usuario."""
    return interacciones.groupby("id_lector")["id_libro"].agg(set).to_dict()
