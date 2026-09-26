import numpy as np
import pytest
import scipy.sparse as sp
from comics_recsys import db as core_db
from webapp import create_app
from webapp import onboarding_datos as datos

# (id_comic, editorial, [(usuario, fecha, rating), ...])
# "hoy" del dataset = 2026-09-01 (la review más reciente).
COMICS = [
    ("marvel-comics/serie-a/1", "Marvel", [("u1", "2026-09-01", 9), ("u2", "2026-08-01", 8), ("u3", "2026-08-02", 8)]),
    ("marvel-comics/serie-a/2", "Marvel", [("u1", "2026-08-15", 9), ("u2", "2026-08-16", 9)]),
    ("dc-comics/serie-b/1", "DC", [("u1", "2026-07-01", 7)]),
    ("marvel-comics/serie-c/1", "Marvel", [("u1", "2023-01-01", 9)]),
    ("image-comics/serie-d/1", "Image", [(f"v{i}", "2019-01-01", 10) for i in range(25)]),
    ("image-comics/serie-e/1", "Image", [(f"w{i}", "2024-01-01", 8) for i in range(12)]),
]
# dos recientes más (1 review c/u), para que el carrusel del último año tenga
# más comics que el mínimo de interacciones cuando se eligen 4 editoriales
COMICS += [
    ("image-comics/serie-g/1", "Image", [("u4", "2026-06-01", 8)]),
    ("boom-studios/serie-h/1", "Boom!", [("u5", "2026-05-01", 7)]),
]
SIN_TAPA = ("marvel-comics/serie-f/1", "Marvel", [(f"z{i}", "2026-08-20", 9) for i in range(30)])


def _preparar(tmp_path):
    db_path = tmp_path / "comics.db"
    conn = core_db.conectar(db_path)
    core_db.crear_esquema(conn)
    with conn:
        for id_comic, editorial, reviews in COMICS + [SIN_TAPA]:
            img_src = None if id_comic == SIN_TAPA[0] else f"https://img/{id_comic}.jpg"
            conn.execute(
                "INSERT INTO comics (id_comic, titulo, editorial, img_src) VALUES (?, ?, ?, ?)",
                (id_comic, id_comic, editorial, img_src),
            )
            for usuario, fecha, rating in reviews:
                conn.execute("INSERT OR IGNORE INTO usuarios (id_usuario, nombre) VALUES (?, ?)", (usuario, usuario))
                conn.execute(
                    "INSERT INTO interacciones (id_usuario, id_comic, fecha, rating, texto) VALUES (?, ?, ?, ?, '')",
                    (usuario, id_comic, fecha, rating),
                )
    conn.close()

    # similitud: serie-d/1 ~ serie-e/1 (para ver que una curiosidad mueve la recomendación)
    ids = np.array([c[0] for c in COMICS])
    i_d, i_e = 4, 5
    sim = sp.csr_matrix(([0.8, 0.8], ([i_d, i_e], [i_e, i_d])), shape=(len(ids), len(ids)))
    cache_path = tmp_path / "sim.npz"
    np.savez_compressed(
        cache_path, data=sim.data, indices=sim.indices, indptr=sim.indptr, shape=np.array(sim.shape), comic_ids=ids
    )
    return db_path, cache_path


@pytest.fixture
def rutas(tmp_path):
    return _preparar(tmp_path)


@pytest.fixture
def app(rutas):
    db_path, cache_path = rutas
    return create_app(TESTING=True, DB_PATH=db_path, RECS_CACHE_PATH=cache_path)


# --- selección de comics por fase ---------------------------------------------


def test_fase_reciente_muestra_un_issue_por_serie_e_intercala_editoriales(rutas):
    catalogo = datos.CatalogoOnboarding(rutas[0])

    ids = [c["id_comic"] for c in catalogo.comics_de_fase(2, "x", {"Marvel", "DC"})]

    # serie-a/1 (3 reviews) le gana a serie-a/2 y la serie aparece una sola vez;
    # serie-c/1 es de 2023, fuera del último año; serie-f/1 es el más reseñado
    # pero no tiene tapa -> no se muestra
    assert ids == ["marvel-comics/serie-a/1", "dc-comics/serie-b/1"]


def test_fase_ecos_usa_la_ventana_de_1_a_5_anios(rutas):
    catalogo = datos.CatalogoOnboarding(rutas[0])

    ids = [c["id_comic"] for c in catalogo.comics_de_fase(3, "x", {"Marvel", "Image"})]

    assert set(ids) == {"marvel-comics/serie-c/1", "image-comics/serie-e/1"}


def test_fundacionales_son_viejos_y_con_suficientes_reviews(rutas):
    catalogo = datos.CatalogoOnboarding(rutas[0])

    ids = [c["id_comic"] for c in catalogo.comics_de_fase(4, "x", set())]

    assert ids == ["image-comics/serie-d/1"]


def test_anomalias_solo_de_editoriales_no_elegidas_y_estables_por_usuario(rutas):
    catalogo = datos.CatalogoOnboarding(rutas[0])

    primera = catalogo.comics_de_fase(5, "neo", {"Marvel", "DC"})
    segunda = catalogo.comics_de_fase(5, "neo", {"Marvel", "DC"})

    assert {c["editorial"] for c in primera} == {"Image"}
    assert [c["id_comic"] for c in primera] == [c["id_comic"] for c in segunda]


def test_excluir_saca_lo_ya_respondido(rutas):
    catalogo = datos.CatalogoOnboarding(rutas[0])

    ids = [
        c["id_comic"]
        for c in catalogo.comics_de_fase(2, "x", {"Marvel"}, excluir={"marvel-comics/serie-a/1"})
    ]

    assert ids == ["marvel-comics/serie-a/2"]


# --- flujo completo por HTTP -----------------------------------------------------


def _registrar(client):
    return client.post("/registro", data={"nombre": "Neo", "password": "unapassword"})


def test_cuenta_nueva_pasa_por_la_bienvenida_que_ofrece_jugar_o_ir_al_sitio(app):
    client = app.test_client()

    respuesta = _registrar(client)

    assert respuesta.headers["Location"].endswith("/onboarding/bienvenida")
    html = client.get("/onboarding/bienvenida").get_data(as_text=True)
    assert 'href="/onboarding/"' in html  # jugar
    assert 'href="/"' in html  # ir directo al sitio


def _hasta_fase_2(client, prefs):
    client.post("/onboarding/0")
    return client.post("/onboarding/1", data={f"pref-{e}": p for e, p in prefs.items()})


def _uid(app, nombre="Neo"):
    conn = core_db.conectar(app.config["DB_PATH"])
    uid = conn.execute("SELECT id_usuario FROM usuarios WHERE nombre = ?", (nombre,)).fetchone()[0]
    return conn, uid


def test_fase_1_exige_al_menos_tres_editoriales(app):
    client = app.test_client()
    _registrar(client)
    client.post("/onboarding/0")

    respuesta = client.post("/onboarding/1", data={"pref-Marvel": "encanta"})

    html = respuesta.get_data(as_text=True)
    assert "Señal insuficiente" in html
    # la elección hecha no se pierde al volver a mostrar la pantalla
    assert 'name="pref-Marvel" value="encanta" checked' in " ".join(html.split())


def test_fases_de_comics_exigen_tres_interacciones_y_conservan_lo_cargado(app):
    client = app.test_client()
    _registrar(client)
    _hasta_fase_2(client, {"Marvel": "gusta", "DC": "gusta", "Image": "gusta", "Boom!": "gusta"})
    ids = ["marvel-comics/serie-a/1", "dc-comics/serie-b/1", "image-comics/serie-g/1", "boom-studios/serie-h/1"]

    respuesta = client.post(
        "/onboarding/2",
        data={"comic": ids, f"rating-{ids[0]}": "8", f"curiosidad-{ids[1]}": "1"},
    )

    html = respuesta.get_data(as_text=True)
    assert respuesta.status_code == 200 and "Señal insuficiente" in html
    # la nota elegida no se pierde: esa estrella vuelve marcada
    assert f'name="rating-{ids[0]}" value="8" checked' in " ".join(html.split())
    conn, uid = _uid(app)
    assert conn.execute("SELECT COUNT(*) FROM interacciones WHERE id_usuario = ?", (uid,)).fetchone()[0] == 0
    assert client.get("/onboarding/3").headers["Location"].endswith("/onboarding/2")  # no avanzó

    respuesta = client.post(
        "/onboarding/2",
        data={"comic": ids, f"rating-{ids[0]}": "8", f"curiosidad-{ids[1]}": "1", f"curiosidad-{ids[2]}": "1"},
    )

    assert respuesta.headers["Location"].endswith("/onboarding/3")


def test_si_el_carrusel_tiene_menos_comics_que_el_minimo_alcanza_con_responder_todos(app):
    client = app.test_client()
    _registrar(client)
    # Marvel + Image -> el último año muestra solo serie-a/1 y serie-g/1
    _hasta_fase_2(client, {"Marvel": "encanta", "DC": "no_atrae", "Image": "curiosidad"})

    respuesta = client.post(
        "/onboarding/2",
        data={
            "comic": ["marvel-comics/serie-a/1", "image-comics/serie-g/1", "comic/inventado/1"],
            "rating-marvel-comics/serie-a/1": "8.5",
            "curiosidad-image-comics/serie-g/1": "1",
            "rating-comic/inventado/1": "10",
        },
    )

    assert respuesta.headers["Location"].endswith("/onboarding/3")
    conn, uid = _uid(app)
    # "lo leí" -> interacciones; curiosidad -> tabla aparte; el id inventado, a ningún lado
    assert conn.execute("SELECT id_comic, rating FROM interacciones WHERE id_usuario = ?", (uid,)).fetchall() == [
        ("marvel-comics/serie-a/1", 8.5)
    ]
    assert conn.execute("SELECT id_comic FROM onboarding_curiosidad WHERE id_usuario = ?", (uid,)).fetchall() == [
        ("image-comics/serie-g/1",)
    ]


def test_no_se_puede_saltar_fases_escribiendo_la_url(app):
    client = app.test_client()
    _registrar(client)

    assert client.get("/onboarding/6").headers["Location"].endswith("/onboarding/0")
    assert client.post("/onboarding/6").headers["Location"].endswith("/onboarding/0")
    conn, uid = _uid(app)
    assert datos.progreso(conn, uid)["completado_en"] is None


def test_una_curiosidad_mueve_las_recomendaciones_del_dashboard(app):
    client = app.test_client()
    _registrar(client)
    conn, uid = _uid(app)
    datos.guardar_curiosidad(conn, uid, ["image-comics/serie-d/1"], 4)
    datos.avanzar(conn, uid, 6, completado=True)

    html = client.get("/").get_data(as_text=True)

    # serie-e/1 es el único vecino de serie-d/1 -> primer recomendado.
    # (serie-d/1 no se recomienda: ya está en el perfil)
    recomendados = html.split("Tu pila por leer")[0]
    assert recomendados.index("image-comics/serie-e/1") < recomendados.index("marvel-comics/serie-a/1")
    assert "Reconstruí tu Códice" not in html  # completado: no se vuelve a ofrecer


def test_dashboard_ofrece_el_onboarding_si_no_hay_historial(app):
    client = app.test_client()
    _registrar(client)

    html = client.get("/").get_data(as_text=True)

    assert "Reconstruí tu Códice" in html


# --- recomendador: orden del relleno por editorial --------------------------------


def test_relleno_por_popularidad_respeta_preferidas_y_evitadas(app):
    recomendador = app.extensions["recomendador"]

    ids = recomendador.recomendar(
        "nadie", k=20, perfil_coldstart={}, editoriales_preferidas={"Image"}, editoriales_evitadas={"Marvel"}
    )

    editoriales = [dict((c[0], c[1]) for c in COMICS + [SIN_TAPA])[i] for i in ids]
    assert len(ids) == 9
    assert editoriales[:3] == ["Image"] * 3
    assert editoriales[-4:] == ["Marvel"] * 4
