"""Tests para el score bayesiano de popularidad."""

import pandas as pd
import pytest

from recsys.models.popularity import fit_popularity


def test_score_con_C_explicito():
    interacciones = pd.DataFrame(
        {
            "id_libro": ["a", "a", "b"],
            "rating": [10, 10, 4],
        }
    )
    # m = (10+10+4)/3 = 8; C=1 fijo
    # "a": n=2, avg=10 -> score = (2/3)*10 + (1/3)*8 = 9.333...
    # "b": n=1, avg=4  -> score = (1/2)*4  + (1/2)*8 = 6.0
    resultado = fit_popularity(interacciones, C=1).set_index("id_libro")

    assert resultado.loc["a", "score"] == pytest.approx(9.3333, abs=1e-3)
    assert resultado.loc["b", "score"] == pytest.approx(6.0, abs=1e-3)
    assert resultado.index[0] == "a"  # ordenado por score descendente


def test_C_none_usa_la_media_de_n_por_defecto():
    interacciones = pd.DataFrame(
        {
            "id_libro": ["a", "a", "a", "b"],
            "rating": [10, 10, 10, 4],
        }
    )
    # n por libro: a=3, b=1 -> C default = mean([3,1]) = 2
    resultado_default = fit_popularity(interacciones)
    resultado_C2 = fit_popularity(interacciones, C=2)

    pd.testing.assert_frame_equal(resultado_default, resultado_C2)


def test_mas_interacciones_pesa_mas_el_rating_propio():
    interacciones = pd.DataFrame(
        {
            "id_libro": ["popular"] * 50 + ["nuevo"],
            "rating": [10] * 50 + [1],
        }
    )
    resultado = fit_popularity(interacciones, C=5).set_index("id_libro")

    # "popular" tiene mucha evidencia -> su score queda cerca de su propio
    # promedio (10), no del promedio global.
    assert resultado.loc["popular", "score"] > 9.0
    # "nuevo" tiene una sola interaccion mala -> se shrinkea fuerte hacia
    # el promedio global, no cae a 1.
    assert resultado.loc["nuevo", "score"] > 3.0
