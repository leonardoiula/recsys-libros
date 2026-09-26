"""
Rellena `comics.img_src` donde quedó NULL, re-parseando el HTML crudo que el
scraper ya dejó en `comics/data/cache/` -- sin ningún request al sitio.

Por qué existe: hasta el 2026-09-26 `parse.py` solo reconocía tapas `.webp`,
y los issues anteriores a ~2022 las tienen en `.jpg` -- ~17 mil comics
quedaron sin tapa. El parser ya está corregido; esto arregla lo ya guardado.

Solo toca filas con `img_src IS NULL`. Idempotente. Los comics cuyo HTML no
esté en el cache se reportan y quedan como están (el webapp muestra
sintapa.jpg).

Uso: uv run python comics/scripts/rellenar_img_src.py
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

COMICS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMICS_DIR / "src"))

from bs4 import BeautifulSoup

from comics_recsys.scraping.cache import ruta_cache
from comics_recsys.scraping.parse import _parsear_metadata

DB_PATH = COMICS_DIR / "data" / "raw" / "comics.db"
CACHE_DIR = COMICS_DIR / "data" / "cache"


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    filas = conn.execute("SELECT id_comic, url FROM comics WHERE img_src IS NULL").fetchall()
    stats = {"pendientes": len(filas), "rellenados": 0, "sin_tapa_en_html": 0, "sin_cache": 0}
    inicio = time.time()

    for i, (id_comic, url) in enumerate(filas, 1):
        ruta = ruta_cache(url, CACHE_DIR)
        if not ruta.exists():
            stats["sin_cache"] += 1
            continue
        soup = BeautifulSoup(ruta.read_text(encoding="utf-8"), "lxml")
        img_src = _parsear_metadata(soup, url).img_src
        if img_src is None:
            stats["sin_tapa_en_html"] += 1
            continue
        conn.execute("UPDATE comics SET img_src = ? WHERE id_comic = ?", (img_src, id_comic))
        stats["rellenados"] += 1
        if i % 1000 == 0:
            conn.commit()
            print(f"{i}/{len(filas)} ({time.time() - inicio:.0f}s) {stats}", flush=True)

    conn.commit()
    conn.close()
    print(f"Listo en {time.time() - inicio:.0f}s: {stats}")


if __name__ == "__main__":
    main()
