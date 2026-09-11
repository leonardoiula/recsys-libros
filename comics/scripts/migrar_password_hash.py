"""
Agrega la columna `usuarios.password_hash` (nullable) a una comics.db ya
poblada -- necesaria para el login/registro del webapp (ver comics/webapp/).
`_ESQUEMA` en comics_recsys/db.py ya la declara para instalaciones nuevas,
pero `CREATE TABLE IF NOT EXISTS` no altera una tabla que ya existe, así que
una comics.db scrapeada antes de este cambio necesita este ALTER TABLE aparte.

Idempotente: correrlo de nuevo sobre una BD ya migrada no falla ni duplica nada.

Uso: uv run python comics/scripts/migrar_password_hash.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comics_recsys import db

COMICS_DIR = Path(__file__).resolve().parents[1]


def columna_existe(conn: sqlite3.Connection, tabla: str, columna: str) -> bool:
    filas = conn.execute(f"PRAGMA table_info({tabla})").fetchall()
    return any(fila[1] == columna for fila in filas)


def migrar(conn: sqlite3.Connection) -> bool:
    """Devuelve True si aplicó el ALTER TABLE, False si ya existía la columna."""
    if columna_existe(conn, "usuarios", "password_hash"):
        return False
    conn.execute("ALTER TABLE usuarios ADD COLUMN password_hash TEXT")
    conn.commit()
    return True


def main() -> None:
    conn = db.conectar(COMICS_DIR / "data" / "raw" / "comics.db")
    if migrar(conn):
        print("Columna password_hash agregada a usuarios.")
    else:
        print("password_hash ya existía, nada que hacer.")


if __name__ == "__main__":
    main()
