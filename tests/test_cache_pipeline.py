"""Tests para `preparar_pipeline_cacheado` -- solo el mecanismo de caché
(hit/miss, clave por seed/config), no el pipeline real (sería lento y ya
lo ejercitan `scripts/evaluate_ranker.py` y compañía). `preparar_pipeline`
se reemplaza por un stub barato vía monkeypatch.
"""

import pickle

import numpy as np
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


def test_cache_preserva_candidatos_test_tras_comprimir_y_descomprimir(tmp_path, monkeypatch):
    """El contexto que devuelve `preparar_pipeline_cacheado` tiene que ser
    idéntico venga de un cache-miss (recién calculado) o de un cache-hit
    (pasó por `_comprimir_para_cache`/`_descomprimir_de_cache`, ver
    docstring) -- ver `test_cache_pipeline_grande_reduce_tamano_en_disco`
    para por qué existe esa compresión."""
    candidatos_test = pd.DataFrame(
        {
            "id_lector": ["u1", "u1", "u2"],
            "id_libro": ["a", "b", "a"],
            "score_als": [0.9, 0.1, 0.5],
        }
    )
    test_final = pd.DataFrame({"id_lector": ["u1", "u2"], "id_libro": ["a", "a"]})
    contexto_original = {"candidatos_test": candidatos_test, "test_final": test_final, "otro": 123}

    monkeypatch.setattr(R, "preparar_pipeline", lambda *a, **kw: contexto_original)
    interacciones, libros, lectores = _datos_minimos()

    ctx_miss = R.preparar_pipeline_cacheado(interacciones, libros, lectores, seed=42, cache_dir=tmp_path)
    # el contexto de un cache-miss no queda mutado a category
    assert ctx_miss["candidatos_test"]["id_libro"].dtype != "category"

    ctx_hit = R.preparar_pipeline_cacheado(interacciones, libros, lectores, seed=42, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(
        ctx_hit["candidatos_test"].reset_index(drop=True),
        candidatos_test.reset_index(drop=True),
        check_dtype=False,
    )
    assert ctx_hit["candidatos_test"]["id_libro"].dtype != "category"  # se restaura tras el cache-hit
    assert ctx_hit["otro"] == 123


def test_comprimir_para_cache_reduce_tamano_en_disco():
    """`candidatos_test` real puede tener millones de filas con `id_lector`/
    `id_libro` repetidos miles de veces sin deduplicar -- picklear eso
    directo (sin pasar por `_comprimir_para_cache`) llegó a pesar 3-4 GB y
    tardaba más en recargarse de lo que tardaba recalcular el contexto
    entero. Este test cuantifica la mejora con un tamaño de sobra para que
    la diferencia sea evidente, sin necesitar datos reales."""
    rng = np.random.default_rng(0)
    n = 200_000
    usuarios = [f"usuario_{i}" for i in range(2_000)]
    libros = [f"libro-con-slug-largo-de-verdad-numero-{i}" for i in range(5_000)]
    candidatos_test = pd.DataFrame(
        {
            "id_lector": rng.choice(usuarios, n),
            "id_libro": rng.choice(libros, n),
            "score_als": rng.random(n),
        }
    )
    contexto = {"candidatos_test": candidatos_test}

    tamano_sin_comprimir = len(pickle.dumps(contexto, protocol=pickle.HIGHEST_PROTOCOL))
    tamano_comprimido = len(pickle.dumps(R._comprimir_para_cache(contexto), protocol=pickle.HIGHEST_PROTOCOL))

    assert tamano_comprimido < tamano_sin_comprimir / 2
