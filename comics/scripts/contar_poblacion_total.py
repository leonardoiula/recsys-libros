"""
Script de medición (no de scraping definitivo): cuenta cuántos issues aparecen
listados en TODAS las páginas semanales de release-dates, sin abrir cada issue.
Sirve para tener un denominador real de "cuánto del sitio ya scrapeamos", no para
poblar la BD (para eso está scrape_comics.py).

Se detiene cuando encuentra `n_semanas_vacias_seguidas` semanas sin datos --
señal de que llegamos al principio del archivo del sitio.
"""
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comics_recsys.scraping.fetch import crear_session, obtener_html, PaginaNoEncontrada
from comics_recsys.scraping.parse import urls_de_issues_en_semana

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
session = crear_session()

semana = date(2026, 9, 2)
total_issues = 0
total_semanas_con_datos = 0
semanas_vacias_seguidas = 0
MAX_SEMANAS_VACIAS_SEGUIDAS = 6  # ~1.5 meses sin datos => asumimos que es el principio del archivo
MAX_SEMANAS_A_REVISAR = 26 * 52  # tope de seguridad: 26 anios

for i in range(MAX_SEMANAS_A_REVISAR):
    url = f"https://comicbookroundup.com/comic-books/release-dates/{semana.isoformat()}"
    try:
        html = obtener_html(url, session)
        n = len(urls_de_issues_en_semana(html))
    except PaginaNoEncontrada:
        n = 0
    except Exception:
        logging.exception("Error en %s, se cuenta como 0 y se sigue", url)
        n = 0

    if n == 0:
        semanas_vacias_seguidas += 1
    else:
        semanas_vacias_seguidas = 0
        total_semanas_con_datos += 1
        total_issues += n

    if i % 20 == 0 or n == 0:
        logging.info("semana=%s issues=%d | acumulado=%d issues en %d semanas", semana, n, total_issues, total_semanas_con_datos)

    if semanas_vacias_seguidas >= MAX_SEMANAS_VACIAS_SEGUIDAS:
        logging.info("Corte: %d semanas seguidas sin datos, asumimos que llegamos al inicio del archivo (semana=%s)", semanas_vacias_seguidas, semana)
        break

    semana -= timedelta(weeks=1)
    time.sleep(0.8)

logging.info("RESULTADO FINAL: total_issues_catalogados=%d total_semanas_con_datos=%d semana_mas_antigua=%s", total_issues, total_semanas_con_datos, semana)
