"""
Parsing de las páginas de comicbookroundup.com.

Todos los selectores de acá salieron de inspeccionar HTML real guardado en
comics/tests/fixtures/ (no son suposiciones). Dos páginas nos importan:

- Página semanal de lanzamientos (`/comic-books/release-dates/AAAA-MM-DD`):
  solo nos interesa la lista de URLs de issues que contiene.
- Página de un issue (`/comic-books/reviews/<editorial>/<serie>/<numero>`):
  trae metadata del comic + reviews de usuario + reviews de crítica, todo
  mezclado en el mismo HTML sin separación clara por contenedor, así que
  distinguimos cada `<li>` de review por su contenido, no por su ubicación.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin

from bs4 import BeautifulSoup

RE_ISSUE_HREF = re.compile(r"^/comic-books/reviews/[^/]+/[^/]+/\d+$")
RE_ANIO = re.compile(r"\((\d{4})\)")

# Nombres de editorial legibles para las que vimos en el menú de "Publishers" del sitio.
# Si aparece una que no está acá, nos quedamos con el slug tal cual (mejor eso que
# reventar el scraping por una editorial nueva).
EDITORIALES_CONOCIDAS = {
    "marvel-comics": "Marvel",
    "dc-comics": "DC",
    "image-comics": "Image",
    "dark-horse-comics": "Dark Horse",
    "idw-publishing": "IDW",
    "boom-studios": "Boom!",
    "dynamite": "Dynamite",
}


@dataclass
class ComicMetadata:
    id_comic: str  # ej: "dc-comics/batman-(2025)/1" -- derivado de la URL, estable
    titulo: str
    serie: str | None
    numero: str | None
    editorial: str | None
    anio_edicion: int | None
    escritor: str | None
    dibujante: str | None
    precio_tapa: str | None
    img_src: str | None
    url: str


@dataclass
class ReviewUsuario:
    id_usuario: str  # id numérico estable de /user/profile/<id>, no el nombre
    nombre_usuario: str
    rating: float
    fecha_texto: str  # tal cual aparece en el sitio, ej "Sep 02, 2025"
    texto: str


@dataclass
class ReviewCritico:
    outlet: str
    reviewer: str | None
    rating: float
    fecha_texto: str
    texto: str
    url_externa: str | None


def urls_de_issues_en_semana(html: str) -> list[str]:
    """Extrae las URLs (absolutas) de todos los issues linkeados en una página semanal."""
    soup = BeautifulSoup(html, "lxml")
    hrefs = {
        a["href"] for a in soup.find_all("a", href=RE_ISSUE_HREF) if a.get("href")
    }
    return sorted(urljoin("https://comicbookroundup.com", h) for h in hrefs)


def _id_comic_desde_url(url: str) -> str:
    # "https://.../comic-books/reviews/dc-comics/batman-(2025)/1" -> "dc-comics/batman-(2025)/1"
    return "/".join(url.rstrip("/").split("/")[-3:])


def _parsear_metadata(soup: BeautifulSoup, url: str) -> ComicMetadata:
    id_comic = _id_comic_desde_url(url)
    editorial_slug, serie_slug, numero = id_comic.split("/")

    h1 = soup.find("h1")
    titulo = h1.get_text(" ", strip=True) if h1 else id_comic

    m_anio = RE_ANIO.search(serie_slug)
    anio_edicion = int(m_anio.group(1)) if m_anio else None
    serie = RE_ANIO.sub("", serie_slug).replace("-", " ").strip().title() or None

    escritor = dibujante = precio_tapa = None
    tabla = soup.find("table", class_="issue-info")
    if tabla:
        for fila in tabla.find_all("tr"):
            celdas = fila.find_all("td")
            if len(celdas) != 2:
                continue
            etiqueta = celdas[0].get_text(strip=True).lower()
            valor = celdas[1].get_text(" ", strip=True)
            if etiqueta == "writer":
                escritor = valor
            elif etiqueta == "artist":
                dibujante = valor
            elif etiqueta == "cover price":
                precio_tapa = valor

    img_src = None
    for img in soup.find_all("img", src=True):
        if img["src"].rstrip("/").endswith(f"/{numero}.webp") and "/covers/" in img["src"]:
            img_src = img["src"]
            break

    return ComicMetadata(
        id_comic=id_comic,
        titulo=titulo,
        serie=serie,
        numero=numero,
        editorial=EDITORIALES_CONOCIDAS.get(editorial_slug, editorial_slug),
        anio_edicion=anio_edicion,
        escritor=escritor,
        dibujante=dibujante,
        precio_tapa=precio_tapa,
        img_src=img_src,
        url=url,
    )


def _texto_de_p_sin_links(p) -> str:
    """Texto de un <p>, sacando anchors (ej. 'Read Full Review') para no ensuciar el texto."""
    copia = BeautifulSoup(str(p), "lxml")
    for a in copia.find_all("a"):
        a.decompose()
    return copia.get_text(" ", strip=True)


def _parsear_reviews(soup: BeautifulSoup) -> tuple[list[ReviewUsuario], list[ReviewCritico]]:
    reviews_usuario: list[ReviewUsuario] = []
    reviews_critico: list[ReviewCritico] = []

    for li in soup.find_all("li"):
        div_score = li.find("div", class_="list-review")
        p = li.find("p")
        h3 = li.find("h3")
        if div_score is None or p is None or h3 is None:
            continue  # no tiene la forma de una review, es otro <li> cualquiera de la página

        span_score = div_score.find("span")
        if span_score is None or not span_score.get_text(strip=True):
            continue
        try:
            rating = float(span_score.get_text(strip=True))
        except ValueError:
            continue

        detalles_usuario = li.find(class_="user-review-details")
        span_fecha = li.find("span", class_="date")
        fecha_texto = span_fecha.get_text(strip=True) if span_fecha else ""
        texto = _texto_de_p_sin_links(p)

        if detalles_usuario is not None:
            link_perfil = detalles_usuario.find("a", href=re.compile(r"^/user/profile/\d+$"))
            if link_perfil is None:
                continue
            id_usuario = link_perfil["href"].rsplit("/", 1)[-1]
            nombre_usuario = link_perfil.get_text(strip=True)
            reviews_usuario.append(
                ReviewUsuario(
                    id_usuario=id_usuario,
                    nombre_usuario=nombre_usuario,
                    rating=rating,
                    fecha_texto=fecha_texto,
                    texto=texto,
                )
            )
        else:
            # review de crítica: "<h3>Outlet - <a>Reviewer</a></h3>"
            link_reviewer = h3.find("a")
            reviewer = link_reviewer.get_text(strip=True) if link_reviewer else None
            outlet = h3.get_text(" ", strip=True)
            if reviewer and outlet.endswith(reviewer):
                outlet = outlet[: -len(reviewer)].rstrip(" -").strip()
            link_externo = p.find("a", href=True)
            url_externa = link_externo["href"] if link_externo else None
            reviews_critico.append(
                ReviewCritico(
                    outlet=outlet,
                    reviewer=reviewer,
                    rating=rating,
                    fecha_texto=fecha_texto,
                    texto=texto,
                    url_externa=url_externa,
                )
            )

    return reviews_usuario, reviews_critico


def parsear_pagina_issue(
    html: str, url: str
) -> tuple[ComicMetadata, list[ReviewUsuario], list[ReviewCritico]]:
    soup = BeautifulSoup(html, "lxml")
    comic = _parsear_metadata(soup, url)
    reviews_usuario, reviews_critico = _parsear_reviews(soup)
    return comic, reviews_usuario, reviews_critico
