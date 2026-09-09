from comics_recsys.scraping.fetch import es_redirect_a_pagina_generica


def test_misma_url_no_es_pagina_generica():
    url = "https://comicbookroundup.com/comic-books/reviews/dc-comics/batman-(2025)/1"
    assert not es_redirect_a_pagina_generica(url, url)


def test_redirect_a_la_home_es_pagina_generica():
    pedida = "https://comicbookroundup.com/comic-books/reviews/dc-comics/batman-(2025)/9999"
    final = "https://comicbookroundup.com/"
    assert es_redirect_a_pagina_generica(pedida, final)


def test_redirect_a_la_home_sin_barra_final_tambien_cuenta():
    pedida = "https://comicbookroundup.com/comic-books/reviews/dc-comics/batman-(2025)/9999"
    final = "https://comicbookroundup.com"
    assert es_redirect_a_pagina_generica(pedida, final)


def test_redirect_a_otra_pagina_real_no_es_generico():
    # ej: un slug viejo que el sitio actualizó a uno nuevo -- sigue siendo un comic real
    pedida = "https://comicbookroundup.com/comic-books/reviews/dc-comics/batman/1"
    final = "https://comicbookroundup.com/comic-books/reviews/dc-comics/batman-(2025)/1"
    assert not es_redirect_a_pagina_generica(pedida, final)
