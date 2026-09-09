"""Carga de datos y split para el recomendador de comics.

Mismo enfoque que src/recsys/data.py del proyecto de libros (sqlite3 + pandas,
sin ORM), pero con los nombres de columna propios de este dominio
(id_usuario/id_comic en vez de id_lector/id_libro) y un formato de fecha
distinto: acá `fecha` ya viene normalizada a ISO (`db._normalizar_fecha`,
"YYYY-MM-DD"), no al formato "%d-%m-%Y" que usa el dataset de libros.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "comics.db"


def _read_table(table: str, db_path: Path | str = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(f"SELECT * FROM {table}", con)


def load_interacciones(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Carga la tabla interacciones(id_usuario, id_comic, fecha, rating, texto)."""
    return _read_table("interacciones", db_path)


def load_usuarios(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Carga la tabla usuarios(id_usuario, nombre)."""
    return _read_table("usuarios", db_path)


def load_comics(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Carga la tabla comics(id_comic, titulo, serie, numero, editorial,
    anio_edicion, escritor, dibujante, precio_tapa, img_src, url)."""
    return _read_table("comics", db_path)


def load_critic_reviews(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Carga critic_reviews(id_comic, outlet, reviewer, fecha, rating, texto, url_externa).
    No son interacciones de usuario -- ver comics/CLAUDE.md, sección de decisiones de diseño."""
    return _read_table("critic_reviews", db_path)


def split_train_val(
    interacciones: pd.DataFrame,
    n_val: int = 1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split leave-one-out **temporal** por usuario -- idéntica lógica y
    motivo que `recsys.data.split_train_val` (ver ese docstring para el
    detalle completo de por qué temporal y no aleatorio, y por qué `n_val`
    fijo y no proporcional): agrupa por `id_usuario` y retiene a validación
    las `n_val` interacciones más recientes de cada usuario, nunca vacía el
    train de nadie, y trata fechas no parseables como las más antiguas.

    Se reimplementa acá (en vez de importar la de libros) solo porque
    cambian el nombre de columna de usuario y el formato de fecha (acá ya
    viene en ISO, ver `db._normalizar_fecha`).
    """
    if n_val < 1:
        raise ValueError("n_val debe ser >= 1")

    rng = np.random.default_rng(seed)
    fechas = pd.to_datetime(interacciones["fecha"], format="%Y-%m-%d", errors="coerce")

    train_idx: list[int] = []
    val_idx: list[int] = []

    for _, grupo in interacciones.groupby("id_usuario", sort=False):
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


def comics_calificados_por_usuario(interacciones: pd.DataFrame) -> dict:
    """Devuelve {id_usuario: {id_comic, ...}} con los comics ya calificados por cada usuario."""
    return interacciones.groupby("id_usuario")["id_comic"].agg(set).to_dict()
