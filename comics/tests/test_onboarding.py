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


def test_cuenta_nueva_arranca_el_onboarding(app):
    respuesta = _registrar(app.test_client())

    assert respuesta.headers["Location"].endswith("/onboarding/")


def test_fase_1_exige_al_menos_tres_editoriales(app):
    client = app.test_client()
    _registrar(client)

    respuesta = client.post("/onboarding/1", data={"pref-Marvel": "encanta"}, follow_redirects=True)

    assert "Señal insuficiente" in respuesta.get_data(as_text=True)


def test_onboarding_guarda_lecturas_como_interacciones_y_curiosidad_aparte(app):
    client = app.test_client()
    _registrar(client)
    client.post("/onboarding/1", data={"pref-Marvel": "encanta", "pref-DC": "no_atrae", "pref-Image": "curiosidad"})

    client.post(
        "/onboarding/2",
        data={
            "comic": ["marvel-comics/serie-a/1", "dc-comics/serie-b/1", "comic/inventado/1"],
            "rating-marvel-comics/serie-a/1": "8.5",
            "curiosidad-dc-comics/serie-b/1": "1",
            "rating-comic/inventado/1": "10",
        },
    )

    conn = core_db.conectar(app.config["DB_PATH"])
    uid = conn.execute("SELECT id_usuario FROM usuarios WHERE nombre = 'Neo'").fetchone()[0]
    assert conn.execute("SELECT id_comic, rating FROM interacciones WHERE id_usuario = ?", (uid,)).fetchall() == [
        ("marvel-comics/serie-a/1", 8.5)
    ]
    assert conn.execute("SELECT id_comic FROM onboarding_curiosidad WHERE id_usuario = ?", (uid,)).fetchall() == [
        ("dc-comics/serie-b/1",)
    ]


def test_una_curiosidad_mueve_las_recomendaciones_del_dashboard(app):
    client = app.test_client()
    _registrar(client)
    client.post("/onboarding/1", data={"pref-Marvel": "gusta", "pref-DC": "gusta", "pref-Image": "gusta"})
    client.post("/onboarding/4", data={"comic": ["image-comics/serie-d/1"], "curiosidad-image-comics/serie-d/1": "1"})
    client.post("/onboarding/6")

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
        "nadie", k=6, perfil_coldstart={}, editoriales_preferidas={"Image"}, editoriales_evitadas={"Marvel"}
    )

    editoriales = [dict((c[0], c[1]) for c in COMICS + [SIN_TAPA])[i] for i in ids]
    assert editoriales[:2] == ["Image", "Image"]
    assert editoriales[-3:] == ["Marvel", "Marvel", "Marvel"]
