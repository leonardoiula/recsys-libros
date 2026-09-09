import pandas as pd

from comics_recsys.models.popularity import fit_popularity


def test_con_igual_rating_promedio_gana_el_que_tiene_mas_evidencia():
    # "filler" domina el dataset y tira la media global bien abajo (~5.4);
    # "A" y "B" tienen el MISMO rating promedio (9.0, por encima de la
    # media global), pero "A" tiene mucha más evidencia -- el shrinkage
    # bayesiano lo acerca menos a la media global que a "B", que tiene
    # pocas reviews y por lo tanto se diluye más hacia la media general.
    inter = pd.DataFrame(
        {
            "id_comic": ["filler"] * 1000 + ["A"] * 100 + ["B"] * 2,
            "rating": [5.0] * 1000 + [9.0] * 100 + [9.0] * 2,
        }
    )
    resultado = fit_popularity(inter, C=50)
    orden = resultado["id_comic"].tolist()
    assert orden.index("A") < orden.index("B") < orden.index("filler")


def test_c_por_default_es_el_promedio_de_n():
    inter = pd.DataFrame({"id_comic": ["a", "a", "b"], "rating": [8.0, 8.0, 6.0]})
    resultado_default = fit_popularity(inter)
    resultado_explicito = fit_popularity(inter, C=inter.groupby("id_comic").size().mean())
    pd.testing.assert_frame_equal(resultado_default, resultado_explicito)
