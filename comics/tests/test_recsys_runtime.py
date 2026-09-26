import numpy as np
import pytest
import scipy.sparse as sp
from comics_recsys import db as core_db
from webapp.recsys_runtime import Recomendador

COMIC_IDS = np.array(["A", "B", "C", "D"])


def _guardar_similitud_de_prueba(ruta):
    """Vecinos sintéticos: A~B(0.9), A~C(0.1), B~A(0.9), D~C(0.5). C sin vecinos."""
    filas = [0, 0, 1, 3]
    columnas = [1, 2, 0, 2]
    valores = [0.9, 0.1, 0.9, 0.5]
    sim = sp.csr_matrix((valores, (filas, columnas)), shape=(4, 4))

    np.savez_compressed(
        ruta,
        data=sim.data,
        indices=sim.indices,
        indptr=sim.indptr,
        shape=np.array(sim.shape),
        comic_ids=COMIC_IDS,
    )


def _crear_comics_db(ruta):
    conn = core_db.conectar(ruta)
    core_db.crear_esquema(conn)
    with conn:
        for id_comic in COMIC_IDS:
            conn.execute(
                "INSERT INTO comics (id_comic, titulo) VALUES (?, ?)", (id_comic, id_comic)
            )
        # target leyó solo A; el resto de usuarios genera el ranking de popularidad
        filas = [
            ("target", "A", "2020-01-01", 9.0),
            ("u2", "B", "2020-01-01", 8.0),
            ("u3", "B", "2020-01-01", 8.0),
            ("u4", "B", "2020-01-01", 8.0),
            ("u2", "C", "2020-01-01", 5.0),
            ("u5", "D", "2020-01-01", 5.0),
        ]
        for id_usuario, id_comic, fecha, rating in filas:
            conn.execute(
                "INSERT INTO usuarios (id_usuario, nombre) VALUES (?, ?) "
                "ON CONFLICT (id_usuario) DO NOTHING",
                (id_usuario, id_usuario),
            )
            conn.execute(
                "INSERT INTO interacciones (id_usuario, id_comic, fecha, rating, texto) "
                "VALUES (?, ?, ?, ?, '')",
                (id_usuario, id_comic, fecha, rating),
            )
    conn.close()


@pytest.fixture
def recomendador(tmp_path):
    cache_path = tmp_path / "item_item.npz"
    db_path = tmp_path / "comics.db"
    _guardar_similitud_de_prueba(cache_path)
    _crear_comics_db(db_path)
    return Recomendador(cache_path, db_path)


def test_usuario_con_historial_usa_item_item(recomendador):
    # target leyó A -> vecinos de A son B(0.9) y C(0.1), en ese orden de score
    assert recomendador.recomendar("target", k=2) == ["B", "C"]


def test_completa_con_popularidad_si_faltan_candidatos(recomendador):
    # solo hay 2 vecinos de A (B, C); para llegar a k=4 completa con lo que
    # falte del ranking de popularidad (excluyendo A/B/C ya usados) -> D
    resultado = recomendador.recomendar("target", k=4)
    assert resultado == ["B", "C", "D"]


def test_perfil_coldstart_no_toca_historial_real(recomendador):
    # 'newbie' no tiene filas en interacciones; el perfil_coldstart lo reemplaza
    resultado = recomendador.recomendar("newbie", k=1, perfil_coldstart={"D": 1.0})
    assert resultado == ["C"]  # único vecino de D


def test_perfil_vacio_devuelve_fallback_de_popularidad(recomendador):
    resultado = recomendador.recomendar("ghost", k=3)
    assert resultado == recomendador._ranking_popularidad[:3]


def test_registrar_interaccion_se_refleja_sin_reconsultar_la_bd(recomendador):
    # antes de registrar nada, 'nuevo' no tiene historial -> fallback de popularidad
    assert recomendador.recomendar("nuevo", k=1) == recomendador._ranking_popularidad[:1]

    # simula "marcar como leído" un comic recomendado, sin tocar el archivo .npz
    # ni la comics.db en disco -- solo el estado en memoria del proceso
    recomendador.registrar_interaccion("nuevo", "A", 9.0)

    # ahora 'nuevo' tiene a A en su perfil -> usa sus vecinos item-item (B, C)
    assert recomendador.recomendar("nuevo", k=2) == ["B", "C"]


def test_limitar_por_serie_saltea_el_exceso_sin_reordenar():
    from webapp.recsys_runtime import _limitar_por_serie

    ordenados = ["m/usm/2", "m/usm/3", "m/usm/4", "d/batman/1", "m/usm/5", "i/saga/1"]

    assert _limitar_por_serie(ordenados, k=4, max_por_serie=2) == ["m/usm/2", "m/usm/3", "d/batman/1", "i/saga/1"]
    assert _limitar_por_serie(ordenados, k=3, max_por_serie=None) == ordenados[:3]


def test_recomendar_aplica_el_tope_por_serie(tmp_path):
    # 4 issues de la misma serie + 1 de otra; el perfil (issue 1) tiene como
    # vecinos a los otros 3 de su serie con más similitud que a la otra serie
    ids = np.array(["m/usm/1", "m/usm/2", "m/usm/3", "m/usm/4", "d/batman/1"])
    sim = sp.csr_matrix(([0.9, 0.8, 0.7, 0.1], ([0, 0, 0, 0], [1, 2, 3, 4])), shape=(5, 5))
    cache_path = tmp_path / "sim.npz"
    np.savez_compressed(
        cache_path, data=sim.data, indices=sim.indices, indptr=sim.indptr, shape=np.array(sim.shape), comic_ids=ids
    )
    db_path = tmp_path / "comics.db"
    conn = core_db.conectar(db_path)
    core_db.crear_esquema(conn)
    with conn:
        for id_comic in ids:
            conn.execute("INSERT INTO comics (id_comic, titulo) VALUES (?, ?)", (id_comic, id_comic))
    conn.close()

    recomendador = Recomendador(cache_path, db_path)

    assert recomendador.recomendar("x", k=3, perfil_coldstart={"m/usm/1": 9.0}) == ["m/usm/2", "m/usm/3", "d/batman/1"]
