"""Tests para `preparar_pipeline_cacheado` -- solo el mecanismo de caché
(hit/miss, clave por seed/config), no el pipeline real (sería lento y ya
lo ejercitan `scripts/evaluate_ranker.py` y compañía). `preparar_pipeline`
se reemplaza por un stub barato vía monkeypatch.
"""

import pandas as pd

from recsys.models import ranker as R


def _datos_minimos() -> tuple:
    interacciones = pd.DataFrame({"id_lector": [1]})
    libros = pd.DataFrame({"id_libro": [1]})
    lectores = pd.DataFrame({"id_lector": [1]})
    return interacciones, libros, lectores


def test_cache_hit_no_vuelve_a_llamar_preparar_pipeline(tmp_path, monkeypatch):
    llamadas = []

    def _stub(interacciones, libros, lectores, seed, **kwargs):
        llamadas.append(seed)
        return {"seed": seed, "marca": "calculado"}

    monkeypatch.setattr(R, "preparar_pipeline", _stub)
    interacciones, libros, lectores = _datos_minimos()

    ctx1 = R.preparar_pipeline_cacheado(interacciones, libros, lectores, seed=42, cache_dir=tmp_path)
    ctx2 = R.preparar_pipeline_cacheado(interacciones, libros, lectores, seed=42, cache_dir=tmp_path)

    assert ctx1 == ctx2 == {"seed": 42, "marca": "calculado"}
    assert llamadas == [42]  # la segunda llamada fue un cache hit, no recalculo


def test_cache_distingue_por_seed_y_fuentes_activas(tmp_path, monkeypatch):
    llamadas = []

    def _stub(interacciones, libros, lectores, seed, fuentes_activas=None, **kwargs):
        llamadas.append((seed, fuentes_activas))
        return {"seed": seed, "fuentes_activas": fuentes_activas}

    monkeypatch.setattr(R, "preparar_pipeline", _stub)
    interacciones, libros, lectores = _datos_minimos()

    R.preparar_pipeline_cacheado(interacciones, libros, lectores, seed=42, cache_dir=tmp_path)
    R.preparar_pipeline_cacheado(interacciones, libros, lectores, seed=7, cache_dir=tmp_path)
    R.preparar_pipeline_cacheado(
        interacciones,
        libros,
        lectores,
        seed=42,
        cache_dir=tmp_path,
        fuentes_activas=frozenset({"als"}),
    )

    # las tres son configuraciones distintas -- ningun cache hit entre ellas
    assert len(llamadas) == 3

    # pero repetir la primera exacta si pega en cache (no llama de nuevo)
    R.preparar_pipeline_cacheado(interacciones, libros, lectores, seed=42, cache_dir=tmp_path)
    assert len(llamadas) == 3


def test_cache_hit_devuelve_el_mismo_contenido_que_el_cache_miss(tmp_path, monkeypatch):
    """El contexto que devuelve `preparar_pipeline_cacheado` tiene que ser
    idéntico venga de un cache-miss (recién calculado) o de un cache-hit
    (pasó por un pickle.dump/pickle.load de ida y vuelta) -- ya no hace
    falta ninguna transformación especial de dtypes acá: la compresión de
    `id_lector`/`id_libro` a `category` es parte normal de lo que devuelve
    `generar_candidatos_con_features_por_lotes` (ver
    `tests/test_ranker.py`), no un paso extra del cacheo."""
    candidatos_test = pd.DataFrame(
        {
            "id_lector": pd.Categorical(["u1", "u1", "u2"]),
            "id_libro": pd.Categorical(["a", "b", "a"]),
            "score_als": [0.9, 0.1, 0.5],
        }
    )
    test_final = pd.DataFrame({"id_lector": ["u1", "u2"], "id_libro": ["a", "a"]})
    contexto_original = {"candidatos_test": candidatos_test, "test_final": test_final, "otro": 123}

    monkeypatch.setattr(R, "preparar_pipeline", lambda *a, **kw: contexto_original)
    interacciones, libros, lectores = _datos_minimos()

    ctx_miss = R.preparar_pipeline_cacheado(interacciones, libros, lectores, seed=42, cache_dir=tmp_path)
    ctx_hit = R.preparar_pipeline_cacheado(interacciones, libros, lectores, seed=42, cache_dir=tmp_path)

    pd.testing.assert_frame_equal(ctx_hit["candidatos_test"], ctx_miss["candidatos_test"])
    assert ctx_hit["otro"] == ctx_miss["otro"] == 123
