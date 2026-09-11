import numpy as np
import pytest
import scipy.sparse as sp
from comics_recsys import db as core_db
from webapp import create_app


def _preparar_bd(db_path):
    conn = core_db.conectar(db_path)
    core_db.crear_esquema(conn)
    with conn:
        conn.execute(
            "INSERT INTO comics (id_comic, titulo, serie, numero) VALUES ('a', 'Comic A', 'Serie', '1')"
        )
        conn.execute(
            "INSERT INTO comics (id_comic, titulo, serie, numero) VALUES ('b', 'Comic B', 'Serie', '2')"
        )
        conn.execute("INSERT INTO usuarios (id_usuario, nombre) VALUES ('target', 'target')")
        conn.execute(
            "INSERT INTO interacciones (id_usuario, id_comic, fecha, rating, texto) "
            "VALUES ('target', 'a', '2020-01-01', 9.0, '')"
        )
    conn.close()


def _preparar_cache_similitud(cache_path):
    vacia = sp.csr_matrix((1, 1))
    np.savez_compressed(
        cache_path,
        data=vacia.data,
        indices=vacia.indices,
        indptr=vacia.indptr,
        shape=np.array(vacia.shape),
        comic_ids=np.array(["a"]),
    )


@pytest.fixture
def app(tmp_path):
    db_path = tmp_path / "comics.db"
    cache_path = tmp_path / "item_item.npz"
    _preparar_bd(db_path)
    _preparar_cache_similitud(cache_path)
    return create_app(TESTING=True, DB_PATH=db_path, RECS_CACHE_PATH=cache_path)


def test_registro_y_login_llevan_al_dashboard(app):
    client = app.test_client()

    respuesta = client.post(
        "/registro", data={"nombre": "Lea", "password": "unapassword"}, follow_redirects=True
    )

    assert respuesta.status_code == 200
    assert b"Hola, Lea" in respuesta.data
    assert b"Comic A" in respuesta.data  # recomendado por fallback de popularidad


def test_login_con_password_incorrecta_no_ingresa(app):
    client = app.test_client()
    client.post("/registro", data={"nombre": "Lea", "password": "correcta"})
    client.post("/logout")

    respuesta = client.post(
        "/login", data={"identificador": "Lea", "password": "incorrecta"}, follow_redirects=True
    )

    assert respuesta.status_code == 200
    assert b"incorrectos" in respuesta.data
    assert b"Hola, Lea" not in respuesta.data


def test_dashboard_sin_sesion_redirige_a_login(app):
    client = app.test_client()

    respuesta = client.get("/", follow_redirects=False)

    assert respuesta.status_code == 302
    assert "/login" in respuesta.headers["Location"]


def test_agregar_y_quitar_de_la_pila_por_leer(app):
    client = app.test_client()
    client.post("/registro", data={"nombre": "Lea", "password": "unapassword"})

    respuesta = client.post("/comics/b/pila", follow_redirects=True)
    assert b"Comic B" in respuesta.data  # aparece en "Tu pila por leer"

    respuesta = client.post("/comics/b/pila/quitar", follow_redirects=True)
    assert b"Todav\xc3\xada no guardaste nada para despu\xc3\xa9s" in respuesta.data


def test_marcar_leido_lo_agrega_a_la_comiteca_y_lo_saca_de_la_pila(app):
    client = app.test_client()
    client.post("/registro", data={"nombre": "Lea", "password": "unapassword"})
    client.post("/comics/b/pila", follow_redirects=True)

    respuesta = client.post("/comics/b/leido", data={"rating": "7.5"}, follow_redirects=True)

    assert b"Agregado a tu comiteca" in respuesta.data
    assert b"Tu rating: 7.5" in respuesta.data
    assert b"Todav\xc3\xada no guardaste nada para despu\xc3\xa9s" in respuesta.data  # ya no está en la pila
