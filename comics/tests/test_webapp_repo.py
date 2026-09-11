import sqlite3

import pytest
from comics_recsys import db as core_db
from webapp import repo
from werkzeug.security import generate_password_hash


def _conexion_en_memoria() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    core_db.crear_esquema(conn)
    return conn


def _insertar_usuario(conn, id_usuario, nombre, password=None):
    conn.execute(
        "INSERT INTO usuarios (id_usuario, nombre, password_hash) VALUES (?, ?, ?)",
        (id_usuario, nombre, generate_password_hash(password) if password else None),
    )
    conn.commit()


def _insertar_comic(conn, id_comic="dc-comics/batman-(2025)/1", titulo="Batman #1"):
    conn.execute(
        """
        INSERT INTO comics (id_comic, titulo, serie, numero, editorial, anio_edicion,
                             escritor, dibujante, precio_tapa, img_src, url)
        VALUES (?, ?, 'Batman', '1', 'DC', 2025, 'Matt Fraction', 'Jorge Jimenez', '$4.99',
                'https://images.comicbookroundup.com/img/covers/b/batman.webp', 'https://x')
        """,
        (id_comic, titulo),
    )
    conn.commit()


def test_crear_cuenta_nueva_usa_prefijo_local_y_no_choca_con_ids_scrapeados():
    conn = _conexion_en_memoria()
    _insertar_usuario(conn, "8994", "motorik", password="algo")

    id_nuevo = repo.crear_cuenta_nueva(conn, "Lea", "unapassword")

    assert id_nuevo.startswith("local-")
    assert id_nuevo != "8994"
    assert repo.verificar_password(conn, id_nuevo, "unapassword")


def test_reclamar_identidad_sobre_usuario_sin_reclamar():
    conn = _conexion_en_memoria()
    _insertar_usuario(conn, "8994", "motorik")  # sin password -> scrapeado, sin reclamar

    repo.reclamar_identidad(conn, "8994", "unapassword")

    assert repo.verificar_password(conn, "8994", "unapassword")


def test_reclamar_identidad_ya_reclamada_lanza_value_error():
    conn = _conexion_en_memoria()
    _insertar_usuario(conn, "8994", "motorik", password="primera")

    with pytest.raises(ValueError):
        repo.reclamar_identidad(conn, "8994", "segunda")

    # el password original no se pisó
    assert repo.verificar_password(conn, "8994", "primera")
    assert not repo.verificar_password(conn, "8994", "segunda")


def test_verificar_password_false_si_no_reclamado_o_no_existe():
    conn = _conexion_en_memoria()
    _insertar_usuario(conn, "8994", "motorik")  # sin password

    assert not repo.verificar_password(conn, "8994", "cualquiera")
    assert not repo.verificar_password(conn, "no-existe", "cualquiera")


def test_buscar_usuarios_sin_reclamar_excluye_ya_reclamados():
    conn = _conexion_en_memoria()
    _insertar_usuario(conn, "8994", "motorik")
    _insertar_usuario(conn, "1234", "motorik_reclamado", password="algo")

    resultados = repo.buscar_usuarios_sin_reclamar(conn, "motorik")

    ids = [fila["id_usuario"] for fila in resultados]
    assert "8994" in ids
    assert "1234" not in ids


def test_comics_leidos_trae_metadata_via_join():
    conn = _conexion_en_memoria()
    _insertar_usuario(conn, "8994", "motorik", password="algo")
    _insertar_comic(conn)
    conn.execute(
        "INSERT INTO interacciones (id_usuario, id_comic, fecha, rating, texto) "
        "VALUES ('8994', 'dc-comics/batman-(2025)/1', '2025-09-02', 9.0, 'bueno')"
    )
    conn.commit()

    leidos = repo.comics_leidos(conn, "8994")

    assert len(leidos) == 1
    assert leidos[0]["titulo"] == "Batman #1"
    assert leidos[0]["rating"] == 9.0


def test_comics_por_id_preserva_el_orden_de_entrada():
    conn = _conexion_en_memoria()
    _insertar_comic(conn, id_comic="a", titulo="A")
    _insertar_comic(conn, id_comic="b", titulo="B")

    resultado = repo.comics_por_id(conn, ["b", "a"])

    assert [c["id_comic"] for c in resultado] == ["b", "a"]


def test_comics_leidos_admite_orden_y_paginado():
    conn = _conexion_en_memoria()
    _insertar_usuario(conn, "8994", "motorik", password="algo")
    _insertar_comic(conn, id_comic="a", titulo="A")
    _insertar_comic(conn, id_comic="b", titulo="B")
    conn.execute(
        "INSERT INTO interacciones (id_usuario, id_comic, fecha, rating, texto) VALUES "
        "('8994', 'a', '2020-01-01', 5.0, ''), ('8994', 'b', '2020-06-01', 9.0, '')"
    )
    conn.commit()

    assert repo.contar_comics_leidos(conn, "8994") == 2

    por_rating = repo.comics_leidos(conn, "8994", orden="rating")
    assert [c["id_comic"] for c in por_rating] == ["b", "a"]

    solo_una_por_pagina = repo.comics_leidos(conn, "8994", orden="fecha", pagina=1, por_pagina=1)
    assert [c["id_comic"] for c in solo_una_por_pagina] == ["b"]  # fecha desc -> la más reciente


def test_agregar_y_quitar_de_pila_por_leer():
    conn = _conexion_en_memoria()
    _insertar_usuario(conn, "8994", "motorik", password="algo")
    _insertar_comic(conn, id_comic="a", titulo="A")

    repo.agregar_a_pila(conn, "8994", "a")
    assert [c["id_comic"] for c in repo.pila_por_leer(conn, "8994")] == ["a"]

    repo.quitar_de_pila(conn, "8994", "a")
    assert repo.pila_por_leer(conn, "8994") == []


def test_editoriales_leidas_y_filtro_por_editorial():
    conn = _conexion_en_memoria()
    _insertar_usuario(conn, "8994", "motorik", password="algo")
    _insertar_comic(conn, id_comic="a", titulo="A")  # editorial DC (ver _insertar_comic)
    conn.execute(
        """
        INSERT INTO comics (id_comic, titulo, serie, numero, editorial, anio_edicion,
                             escritor, dibujante, precio_tapa, img_src, url)
        VALUES ('b', 'B', 'Serie', '1', 'Marvel', 2025, 'X', 'Y', '$4.99', NULL, 'https://x')
        """
    )
    conn.execute(
        "INSERT INTO interacciones (id_usuario, id_comic, fecha, rating, texto) VALUES "
        "('8994', 'a', '2020-01-01', 5.0, ''), ('8994', 'b', '2020-01-01', 5.0, '')"
    )
    conn.commit()

    assert repo.editoriales_leidas(conn, "8994") == ["DC", "Marvel"]
    assert repo.contar_comics_leidos(conn, "8994", editorial="Marvel") == 1

    solo_marvel = repo.comics_leidos(conn, "8994", editorial="Marvel")
    assert [c["id_comic"] for c in solo_marvel] == ["b"]


def test_marcar_como_leido_crea_interaccion_y_saca_de_la_pila():
    conn = _conexion_en_memoria()
    _insertar_usuario(conn, "8994", "motorik", password="algo")
    _insertar_comic(conn, id_comic="a", titulo="A")
    repo.agregar_a_pila(conn, "8994", "a")

    repo.marcar_como_leido(conn, "8994", "a", 8.5)

    leidos = repo.comics_leidos(conn, "8994")
    assert len(leidos) == 1
    assert leidos[0]["rating"] == 8.5
    assert repo.pila_por_leer(conn, "8994") == []
