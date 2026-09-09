"""
CLI del scraper. Uso:

    uv run python comics/scripts/scrape_comics.py --semanas 12 --desde 2026-09-02

Guarda en comics/data/raw/comics.db, cachea el HTML crudo en comics/data/cache/.
Correrlo de nuevo con más --semanas es seguro: los issues ya guardados se saltan.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comics_recsys import db  # noqa: E402
from comics_recsys.scraping.crawl import scrapear_semanas  # noqa: E402
from comics_recsys.scraping.fetch import crear_session  # noqa: E402

COMICS_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semanas", type=int, default=4, help="Cuántas semanas hacia atrás recorrer")
    parser.add_argument(
        "--desde", type=date.fromisoformat, default=date.today(), help="Semana más reciente (AAAA-MM-DD)"
    )
    parser.add_argument("--espera", type=float, default=1.5, help="Segundos de espera entre requests a issues")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    conn = db.conectar(COMICS_DIR / "data" / "raw" / "comics.db")
    db.crear_esquema(conn)
    session = crear_session()
    cache_dir = COMICS_DIR / "data" / "cache"

    stats = scrapear_semanas(
        conn, session, cache_dir, semana_inicial=args.desde, n_semanas=args.semanas, espera=args.espera
    )
    logging.info("Resultado: %s", stats)


if __name__ == "__main__":
    main()
