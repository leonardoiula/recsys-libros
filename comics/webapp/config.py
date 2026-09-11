"""Configuración del webapp. Rutas resueltas con `Path(__file__)`, nunca cwd
-- PythonAnywhere ejecuta el WSGI con un cwd que no necesariamente es la raíz
del proyecto (ver comics/docs/webapp/guia.md, sección deployment)."""

from __future__ import annotations

import os
from pathlib import Path

COMICS_DIR = Path(__file__).resolve().parents[1]


class Config:
    SECRET_KEY = os.environ.get("COMICS_SECRET_KEY", "dev-insecure-key-no-usar-en-produccion")
    DB_PATH = Path(os.environ.get("COMICS_DB_PATH", COMICS_DIR / "data" / "raw" / "comics.db"))
    RECS_CACHE_PATH = Path(
        os.environ.get("COMICS_RECS_CACHE_PATH", COMICS_DIR / "data" / "cache" / "item_item_top50.npz")
    )
