"""
Capa HTTP para el scraper de comicbookroundup.com.

Decisiones que salen de probar contra el sitio real, no de suposiciones:

- No hace falta un User-Agent especial para que el sitio responda: probamos el UA
  por default de `requests`, uno identificado y uno de navegador común, y los tres
  devolvieron 200. Igual mandamos uno identificado -- es la práctica correcta aunque
  acá no sea estrictamente necesaria (y puede dejar de serlo si el sitio cambia).
- Los issues que no existen NO devuelven 404: el sitio responde con un 302 a la
  home (`https://comicbookroundup.com/`). Si dejáramos que `requests` siguiera el
  redirect a ciegas (lo hace por default), terminaríamos "parseando" la home como
  si fuera un comic real. Por eso comparamos la URL final contra la pedida.
"""

from __future__ import annotations

import time

import requests

USER_AGENT = "TP-RecSys-Comics/0.1 (proyecto academico universitario; sin fines comerciales)"
TIMEOUT_DEFAULT = 10.0
MAX_REINTENTOS_DEFAULT = 3
ESPERA_BASE_SEGUNDOS = 1.0


class PaginaNoEncontrada(Exception):
    """El sitio redirigió la URL pedida a una página genérica (ej. la home) -> no existe."""


def crear_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def es_redirect_a_pagina_generica(url_pedida: str, url_final: str) -> bool:
    """
    True si el sitio nos mandó a una página que claramente no es la que pedimos
    (p.ej. la home), señal de que el recurso pedido no existe.

    Es una función chica y pura a propósito: se puede testear sin red ni mocks,
    pasándole strings nomás (ver tests/test_fetch.py).
    """
    if url_pedida == url_final:
        return False
    # cualquier redirect que nos deje en la raíz del dominio (con o sin barra final)
    # es sospechoso: ninguna página de issue real vive ahí.
    return url_final.rstrip("/").endswith("comicbookroundup.com")


def obtener_html(
    url: str,
    session: requests.Session,
    timeout: float = TIMEOUT_DEFAULT,
    max_reintentos: int = MAX_REINTENTOS_DEFAULT,
) -> str:
    """
    Descarga el HTML de `url`. Reintenta con backoff exponencial ante errores
    transitorios (429 / 5xx). Errores permanentes (4xx salvo 429) no se reintentan.
    Lanza `PaginaNoEncontrada` si el sitio redirige a una página genérica.
    """
    ultimo_error: Exception | None = None
    for intento in range(max_reintentos + 1):
        try:
            respuesta = session.get(url, timeout=timeout)
        except requests.exceptions.RequestException as e:
            ultimo_error = e
        else:
            if es_redirect_a_pagina_generica(url, respuesta.url):
                raise PaginaNoEncontrada(url)
            if respuesta.status_code == 429 or respuesta.status_code >= 500:
                ultimo_error = requests.exceptions.HTTPError(
                    f"status={respuesta.status_code} en intento {intento + 1} para {url}"
                )
            elif not respuesta.ok:
                respuesta.raise_for_status()  # 4xx permanente: no tiene sentido reintentar
            else:
                return respuesta.text

        if intento < max_reintentos:
            time.sleep(ESPERA_BASE_SEGUNDOS * (2**intento))

    assert ultimo_error is not None
    raise ultimo_error
