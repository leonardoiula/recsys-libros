"""
Descarga las imágenes de tapa (comics.img_src) a comics/static/covers/, para
usarlas después en el sitio Flask sin depender de hotlinkear al CDN del sitio
original en cada request.

Uso:
    uv run python comics/scripts/descargar_tapas.py [--espera 0.5]

Reanudable: si la imagen ya existe en disco, se saltea (no vuelve a descargar).
Importante: son tapas de comics con copyright de sus editoriales -- este uso es
para un TP académico no comercial; no redistribuir el dataset de imágenes fuera
de ese contexto.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comics_recsys import db  # noqa: E402
from comics_recsys.scraping.fetch import crear_session  # noqa: E402

COMICS_DIR = Path(__file__).resolve().parents[1]


def nombre_archivo(id_comic: str, img_src: str) -> str:
    extension = img_src.rsplit(".", 1)[-1] if "." in img_src.rsplit("/", 1)[-1] else "webp"
    return id_comic.replace("/", "_") + "." + extension


def descargar_tapas(conn, session, destino: Path, espera: float) -> dict:
    destino.mkdir(parents=True, exist_ok=True)
    stats = {"descargadas": 0, "ya_existian": 0, "sin_imagen": 0, "con_error": 0}

    filas = conn.execute("SELECT id_comic, img_src FROM comics").fetchall()
    for id_comic, img_src in filas:
        if not img_src:
            stats["sin_imagen"] += 1
            continue

        ruta = destino / nombre_archivo(id_comic, img_src)
        if ruta.exists():
            stats["ya_existian"] += 1
            continue

        try:
            r = session.get(img_src, timeout=10)
            r.raise_for_status()
            ruta.write_bytes(r.content)
            stats["descargadas"] += 1
        except Exception:
            logging.exception("Error descargando tapa de %s (%s)", id_comic, img_src)
            stats["con_error"] += 1

        time.sleep(espera)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--espera", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    conn = db.conectar(COMICS_DIR / "data" / "raw" / "comics.db")
    session = crear_session()
    destino = COMICS_DIR / "static" / "covers"

    stats = descargar_tapas(conn, session, destino, args.espera)
    logging.info("Resultado: %s", stats)


if __name__ == "__main__":
    main()
