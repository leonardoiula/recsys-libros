from pathlib import Path

from comics_recsys.scraping.parse import parsear_pagina_issue, urls_de_issues_en_semana

FIXTURES = Path(__file__).parent / "fixtures"


def test_parsear_pagina_issue_extrae_metadata_y_reviews():
    html = (FIXTURES / "batman_2025_1.html").read_text(encoding="utf-8")
    url = "https://comicbookroundup.com/comic-books/reviews/dc-comics/batman-(2025)/1"

    comic, reviews_usuario, reviews_critico = parsear_pagina_issue(html, url)

    assert comic.id_comic == "dc-comics/batman-(2025)/1"
    assert comic.titulo == "Batman #1"
    assert comic.numero == "1"
    assert comic.editorial == "DC"
    assert comic.anio_edicion == 2025
    assert comic.escritor == "Matt Fraction"
    assert comic.dibujante == "Jorge Jimenez"
    assert comic.img_src and comic.img_src.endswith("/1.webp")

    assert len(reviews_usuario) > 0
    primera = next(r for r in reviews_usuario if r.id_usuario == "8994")
    assert primera.nombre_usuario == "motorik"
    assert primera.rating == 9.0
    assert primera.fecha_texto == "Sep 02, 2025"
    assert "Loving the art" in primera.texto

    assert len(reviews_critico) > 0
    critica = next(c for c in reviews_critico if c.reviewer == "William Tucker")
    assert critica.outlet == "But Why Tho?"
    assert critica.rating == 10.0
    assert critica.url_externa and critica.url_externa.startswith("https://butwhytho.net")


def test_urls_de_issues_en_semana_devuelve_urls_absolutas_unicas():
    html = (FIXTURES / "semana_2026-09-02.html").read_text(encoding="utf-8")

    urls = urls_de_issues_en_semana(html)

    assert len(urls) > 0
    assert len(urls) == len(set(urls))
    assert all(u.startswith("https://comicbookroundup.com/comic-books/reviews/") for u in urls)
    assert any("batman-(2025)/13" in u for u in urls)
