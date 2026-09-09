"""
Recorrido del catálogo: semana por semana hacia atrás en el tiempo.

Elegimos ir por fecha (no seguir el link "semana anterior" del sitio) porque el
sitio ya nos mostró que esas URLs son predecibles: `/comic-books/release-dates/AAAA-MM-DD`.
Restarle 7 días a la fecha da la semana anterior sin necesidad de parsear el link
de navegación -- una dependencia menos del HTML de esa página.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

from comics_recsys import db
from comics_recsys.scraping.cache import obtener_html_cacheado
from comics_recsys.scraping.fetch import PaginaNoEncontrada
from comics_recsys.scraping.parse import parsear_pagina_issue, urls_de_issues_en_semana

logger = logging.getLogger(__name__)

BASE_URL = "https://comicbookroundup.com"
ESPERA_ENTRE_REQUESTS_SEGUNDOS = 1.5


def url_semana(fecha: date) -> str:
    return f"{BASE_URL}/comic-books/release-dates/{fecha.isoformat()}"


def scrapear_semanas(
    conn,
    session,
    cache_dir,
    semana_inicial: date,
    n_semanas: int,
    espera: float = ESPERA_ENTRE_REQUESTS_SEGUNDOS,
) -> dict:
    """
    Recorre `n_semanas` hacia atrás desde `semana_inicial`. Por cada semana:
    obtiene la lista de issues, y por cada issue que todavía no está en la BD,
    lo descarga, parsea y guarda (issue por issue, no en batch).

    Un error en UN issue no aborta la corrida completa -- se loguea y se sigue
    con el próximo. Devuelve estadísticas simples para saber qué pasó.
    """
    stats = {"semanas_procesadas": 0, "issues_nuevos": 0, "issues_ya_conocidos": 0, "issues_con_error": 0}

    for i in range(n_semanas):
        fecha = semana_inicial - timedelta(weeks=i)
        url = url_semana(fecha)
        try:
            html_semana = obtener_html_cacheado(url, session, cache_dir)
        except PaginaNoEncontrada:
            logger.warning("Semana %s no encontrada, se salta", fecha)
            continue

        urls_issues = urls_de_issues_en_semana(html_semana)
        logger.info("Semana %s: %d issues listados", fecha, len(urls_issues))

        for url_issue in urls_issues:
            id_comic_tentativo = "/".join(url_issue.rstrip("/").split("/")[-3:])
            if db.existe_comic(conn, id_comic_tentativo):
                stats["issues_ya_conocidos"] += 1
                continue

            time.sleep(espera)
            try:
                html_issue = obtener_html_cacheado(url_issue, session, cache_dir)
                comic, reviews_usuario, reviews_critico = parsear_pagina_issue(html_issue, url_issue)
                db.guardar_pagina_issue(conn, comic, reviews_usuario, reviews_critico)
                stats["issues_nuevos"] += 1
            except PaginaNoEncontrada:
                logger.warning("Issue %s no encontrado, se salta", url_issue)
            except Exception:
                # a propósito, catch-all: un issue con HTML inesperado no puede tirar
                # abajo una corrida de horas. Se loguea con traceback y se sigue.
                logger.exception("Error procesando %s", url_issue)
                stats["issues_con_error"] += 1

        stats["semanas_procesadas"] += 1

    return stats
