"""
Descarga las imágenes de tapa (comics.img_src) a comics/static/covers/, ya
redimensionadas a un ancho chico (default 200px, ver --ancho), para usarlas
después en el sitio Flask sin depender de hotlinkear al CDN del sitio
original en cada request. Se redimensiona ACÁ (no con CSS al mostrarlas)
porque el navegador igual descargaría el archivo a resolución original si no
-- con ~14k comics eso pesa varios cientos de MB en disco, contra la cuota de
512MB del plan free de PythonAnywhere (ver comics/docs/webapp/guia.md).

Uso:
    uv run python comics/scripts/descargar_tapas.py [--espera 0.5] [--ancho 200]

Reanudable: si la imagen ya existe en disco, se saltea (no vuelve a descargar).
Importante: son tapas de comics con copyright de sus editoriales -- este uso es
para un TP académico no comercial; no redistribuir el dataset de imágenes fuera
de ese contexto.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comics_recsys import db
from comics_recsys.covers import nombre_archivo
from comics_recsys.scraping.fetch import crear_session

COMICS_DIR = Path(__file__).resolve().parents[1]
ANCHO_DEFAULT = 200


def redimensionar_y_guardar(contenido: bytes, ruta: Path, ancho: int) -> None:
    with Image.open(io.BytesIO(contenido)) as img:
        if img.width > ancho:
            alto = round(img.height * ancho / img.width)
            img = img.resize((ancho, alto), Image.LANCZOS)
        if ruta.suffix.lower() in (".jpg", ".jpeg") and img.mode != "RGB":
            img = img.convert("RGB")
        img.save(ruta)


def descargar_tapas(conn, session, destino: Path, espera: float, ancho: int = ANCHO_DEFAULT) -> dict:
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
            redimensionar_y_guardar(r.content, ruta, ancho)
            stats["descargadas"] += 1
        except Exception:
            logging.exception("Error descargando tapa de %s (%s)", id_comic, img_src)
            stats["con_error"] += 1

        time.sleep(espera)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--espera", type=float, default=0.5)
    parser.add_argument("--ancho", type=int, default=ANCHO_DEFAULT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    conn = db.conectar(COMICS_DIR / "data" / "raw" / "comics.db")
    session = crear_session()
    destino = COMICS_DIR / "static" / "covers"

    stats = descargar_tapas(conn, session, destino, args.espera, args.ancho)
    logging.info("Resultado: %s", stats)


if __name__ == "__main__":
    main()
