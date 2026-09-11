"""Conexión sqlite por request, vía `flask.g` -- sqlite no tolera bien
compartir una conexión entre requests concurrentes, y PythonAnywhere free
corre single-worker igual, así que este patrón (una conexión abierta al
principio del request, cerrada al final) es tanto correcto como simple. Ver
comics/docs/webapp/guia.md, sección "Capa de datos"."""

from __future__ import annotations

import sqlite3

from comics_recsys import db as core_db
from flask import current_app, g


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = core_db.conectar(current_app.config["DB_PATH"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_exception=None) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_app(app) -> None:
    app.teardown_appcontext(close_db)
