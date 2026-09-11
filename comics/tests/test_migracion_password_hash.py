import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from migrar_password_hash import columna_existe, migrar


def _bd_sin_password_hash() -> sqlite3.Connection:
    """Simula una comics.db scrapeada ANTES de agregar password_hash al schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE usuarios (id_usuario TEXT PRIMARY KEY, nombre TEXT)")
    conn.execute("INSERT INTO usuarios (id_usuario, nombre) VALUES ('8994', 'motorik')")
    conn.commit()
    return conn


def test_columna_existe_detecta_antes_y_despues():
    conn = _bd_sin_password_hash()
    assert not columna_existe(conn, "usuarios", "password_hash")
    migrar(conn)
    assert columna_existe(conn, "usuarios", "password_hash")


def test_migrar_agrega_la_columna_sin_perder_datos():
    conn = _bd_sin_password_hash()
    aplico = migrar(conn)
    assert aplico is True

    fila = conn.execute(
        "SELECT nombre, password_hash FROM usuarios WHERE id_usuario = '8994'"
    ).fetchone()
    assert fila == ("motorik", None)


def test_migrar_es_idempotente():
    conn = _bd_sin_password_hash()
    migrar(conn)
    aplico_de_nuevo = migrar(conn)
    assert aplico_de_nuevo is False
