"""
Arma una COPIA liviana y limpia de comics.db para subir al webapp en
PythonAnywhere. La BD local no se toca.

Qué saca y por qué:
- `critic_reviews` entera (~38 MB de texto): el webapp no la usa, el
  recomendador colaborativo solo mira `interacciones` (ver comics/CLAUDE.md).
- `interacciones.texto` (~26 MB): el webapp tampoco lo usa, y
  `load_interacciones` lo cargaría a RAM igual -- RAM y disco son justo lo que
  escasea en el plan free. El upload web de PythonAnywhere, además, tiene un
  límite por archivo que la BD completa (~144 MB) supera.
- Todo lo que generó el uso LOCAL del webapp (pruebas de desarrollo): cuentas
  `local-*` y sus interacciones, `password_hash` de usuarios scrapeados
  "reclamados" en local (si no, en producción esas identidades quedarían
  bloqueadas por una password de prueba), `pila_por_leer`, y los ratings
  rápidos del botón "Leído" (se reconocen por `texto IS NULL`, ver
  `webapp/repo.py::marcar_como_leido`: una review scrapeada sin texto queda
  guardada como cadena vacía, nunca NULL).

Uso: uv run python comics/scripts/armar_db_produccion.py [--destino RUTA]
(default: comics/data/deploy/comics.db, gitignored)
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

COMICS_DIR = Path(__file__).resolve().parents[1]
ORIGEN = COMICS_DIR / "data" / "raw" / "comics.db"
DESTINO_DEFAULT = COMICS_DIR / "data" / "deploy" / "comics.db"


def limpiar(conn: sqlite3.Connection) -> dict:
    stats = {}
    # Primero lo generado por el webapp -- después de anular `texto` ya no se
    # podrían distinguir los ratings rápidos de las reviews scrapeadas.
    stats["interacciones_webapp"] = conn.execute(
        "DELETE FROM interacciones WHERE texto IS NULL OR id_usuario LIKE 'local-%'"
    ).rowcount
    stats["usuarios_local"] = conn.execute("DELETE FROM usuarios WHERE id_usuario LIKE 'local-%'").rowcount
    stats["passwords_reseteadas"] = conn.execute(
        "UPDATE usuarios SET password_hash = NULL WHERE password_hash IS NOT NULL"
    ).rowcount
    stats["pila"] = conn.execute("DELETE FROM pila_por_leer").rowcount
    conn.execute("DROP TABLE IF EXISTS critic_reviews")
    conn.execute("UPDATE interacciones SET texto = NULL")
    conn.commit()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destino", type=Path, default=DESTINO_DEFAULT)
    args = parser.parse_args()

    args.destino.parent.mkdir(parents=True, exist_ok=True)
    args.destino.unlink(missing_ok=True)

    # VACUUM INTO copia de forma consistente (aunque el webapp local esté
    # abierto) y compacta en el mismo paso.
    with sqlite3.connect(ORIGEN) as origen:
        origen.execute("VACUUM INTO ?", (str(args.destino),))

    conn = sqlite3.connect(args.destino)
    stats = limpiar(conn)
    conn.execute("VACUUM")  # recupera el espacio de lo borrado
    conn.close()

    print(f"Limpieza: {stats}")
    print(f"{args.destino}: {args.destino.stat().st_size / 1e6:.1f} MB (origen: {ORIGEN.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
