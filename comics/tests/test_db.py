import sqlite3

from comics_recsys import db
from comics_recsys.scraping.parse import ComicMetadata, ReviewCritico, ReviewUsuario


def _comic_de_prueba(id_comic="dc-comics/batman-(2025)/1"):
    return ComicMetadata(
        id_comic=id_comic,
        titulo="Batman #1",
        serie="Batman",
        numero="1",
        editorial="DC",
        anio_edicion=2025,
        escritor="Matt Fraction",
        dibujante="Jorge Jimenez",
        precio_tapa="$4.99",
        img_src="https://images.comicbookroundup.com/img/covers/b/batman-(2025)/1.webp",
        url="https://comicbookroundup.com/comic-books/reviews/dc-comics/batman-(2025)/1",
    )


def _conexion_en_memoria():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    db.crear_esquema(conn)
    return conn


def test_guardar_pagina_issue_inserta_comic_usuario_e_interaccion():
    conn = _conexion_en_memoria()
    comic = _comic_de_prueba()
    review = ReviewUsuario(
        id_usuario="8994", nombre_usuario="motorik", rating=9.0, fecha_texto="Sep 02, 2025", texto="Muy bueno."
    )

    db.guardar_pagina_issue(conn, comic, [review], [])

    assert db.existe_comic(conn, comic.id_comic)
    fila_usuario = conn.execute("SELECT nombre FROM usuarios WHERE id_usuario = ?", ("8994",)).fetchone()
    assert fila_usuario == ("motorik",)
    fila_interaccion = conn.execute(
        "SELECT fecha, rating FROM interacciones WHERE id_usuario = ? AND id_comic = ?",
        ("8994", comic.id_comic),
    ).fetchone()
    assert fila_interaccion == ("2025-09-02", 9.0)


def test_guardar_pagina_issue_es_idempotente():
    conn = _conexion_en_memoria()
    comic = _comic_de_prueba()
    review = ReviewUsuario(
        id_usuario="8994", nombre_usuario="motorik", rating=9.0, fecha_texto="Sep 02, 2025", texto="Muy bueno."
    )

    db.guardar_pagina_issue(conn, comic, [review], [])
    db.guardar_pagina_issue(conn, comic, [review], [])  # correrlo dos veces no duplica nada

    total_comics = conn.execute("SELECT COUNT(*) FROM comics").fetchone()[0]
    total_interacciones = conn.execute("SELECT COUNT(*) FROM interacciones").fetchone()[0]
    assert total_comics == 1
    assert total_interacciones == 1


def test_guardar_pagina_issue_con_critic_review():
    conn = _conexion_en_memoria()
    comic = _comic_de_prueba()
    critica = ReviewCritico(
        outlet="But Why Tho?",
        reviewer="William Tucker",
        rating=10.0,
        fecha_texto="Jul 31, 2025",
        texto="Excelente numero.",
        url_externa="https://butwhytho.net/2025/07/batman-issue-1-review/",
    )

    db.guardar_pagina_issue(conn, comic, [], [critica])

    fila = conn.execute(
        "SELECT outlet, rating FROM critic_reviews WHERE id_comic = ?", (comic.id_comic,)
    ).fetchone()
    assert fila == ("But Why Tho?", 10.0)
