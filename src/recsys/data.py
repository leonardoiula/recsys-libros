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
    frac_val: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split leave-one-out por usuario.

    Agrupa por id_lector, mezcla sus interacciones y separa `frac_val` de
    ellas para validación, dejando el resto en train. Nunca deja a un
    usuario sin interacciones en train: si `frac_val` implicaría vaciar el
    train de un usuario, se limita a dejarle al menos una interacción.
    """
    if not 0 <= frac_val < 1:
        raise ValueError("frac_val debe estar en el rango [0, 1)")

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for _, grupo in interacciones.groupby("id_lector", sort=False):
        idx = grupo.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = min(int(len(idx) * frac_val), len(idx) - 1)
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])

    train = interacciones.loc[train_idx].reset_index(drop=True)
    val = interacciones.loc[val_idx].reset_index(drop=True)
    return train, val


def libros_leidos_por_usuario(interacciones: pd.DataFrame) -> dict:
    """Devuelve {id_lector: {id_libro, ...}} con los libros leídos por cada usuario."""
    return interacciones.groupby("id_lector")["id_libro"].agg(set).to_dict()
