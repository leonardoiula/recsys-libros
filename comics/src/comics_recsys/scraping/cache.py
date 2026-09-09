"""
Cache en disco del HTML crudo descargado.

Dos razones para esto, no una sola:
1. Reanudable: si el scraping se corta a la mitad (o lo cortás vos a propósito),
   la próxima corrida no vuelve a pedirle al sitio las páginas que ya tiene.
2. Barato para iterar: mientras se ajusta el parser (parse.py) conviene poder
   reparsear el mismo HTML mil veces sin generar tráfico nuevo -- ya lo usamos
   así en esta misma sesión para diseñar los selectores.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from comics_recsys.scraping.fetch import obtener_html


def ruta_cache(url: str, cache_dir: Path) -> Path:
    # hash y no el path "legible" de la URL a propósito: nos ahorramos lidiar con
    # caracteres raros de filesystem (los paréntesis de "batman-(2025)", "?" de
    # querystrings, límites de longitud de path en Windows, etc.)
    nombre = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24] + ".html"
    return cache_dir / nombre


def obtener_html_cacheado(url: str, session, cache_dir: Path, **kwargs) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ruta = ruta_cache(url, cache_dir)
    if ruta.exists():
        return ruta.read_text(encoding="utf-8")

    html = obtener_html(url, session, **kwargs)
    ruta.write_text(html, encoding="utf-8")
    return html
