"""Tests para las métricas de evaluación (NDCG@k, recall@k)."""

import pandas as pd

from recsys.evaluation import evaluar_ndcg, evaluar_recall_personalizado, ndcg_at_k, recall_at_k


def test_ndcg_ranking_perfecto():
    relevantes = {"a", "b", "c"}
    recomendados = ["a", "b", "c", "d", "e"]
    assert ndcg_at_k(recomendados, relevantes, k=5) == 1.0


def test_ndcg_sin_relevantes():
    assert ndcg_at_k(["a", "b", "c"], set(), k=3) == 0.0
    # relevantes existen pero no aparecen entre los recomendados
    assert ndcg_at_k(["a", "b", "c"], {"z"}, k=3) == 0.0


def test_ndcg_orden_importa():
    relevantes = {"a", "z"}
    en_primera_posicion = ["a", "b", "c"]
    en_segunda_posicion = ["b", "a", "c"]

    score_primera = ndcg_at_k(en_primera_posicion, relevantes, k=3)
    score_segunda = ndcg_at_k(en_segunda_posicion, relevantes, k=3)

    assert score_primera > score_segunda


def test_evaluar_ndcg_promedia_por_usuario_y_filtra_leidos():
    val_df = pd.DataFrame(
        {
            "id_lector": ["u1", "u2"],
            "id_libro": ["a", "z"],
        }
    )
    ranking_global = ["x", "a", "b", "c", "z"]
    # u1 ya leyó "x", así que debería quedar filtrado del ranking
    libros_leidos = {"u1": {"x"}, "u2": set()}

    ndcg = evaluar_ndcg(val_df, ranking_global, libros_leidos, k=3)

    # u1: tras filtrar "x" queda ["a", "b", "c"], "a" en posición 0 -> ndcg 1.0
    # u2: top-3 es ["x", "a", "b"], "z" no entra -> ndcg 0.0
    assert ndcg == 0.5


def test_recall_at_k_un_solo_relevante():
    assert recall_at_k(["x", "a", "b"], {"a"}, k=3) == 1.0
    assert recall_at_k(["x", "y", "b"], {"a"}, k=3) == 0.0
    # esta fuera del top-k -> no cuenta
    assert recall_at_k(["a", "x", "y"], {"z"}, k=2) == 0.0


def test_recall_at_k_varios_relevantes_es_fraccion():
    recomendados = ["a", "x", "b", "y"]
    relevantes = {"a", "b", "z"}
    # de los 3 relevantes, 2 (a, b) aparecen en el top-4 -> 2/3
    assert recall_at_k(recomendados, relevantes, k=4) == 2 / 3


def test_recall_at_k_sin_relevantes_es_cero():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_evaluar_recall_personalizado_promedia_por_usuario():
    val_df = pd.DataFrame(
        {
            "id_lector": ["u1", "u2"],
            "id_libro": ["a", "z"],
        }
    )
    recomendaciones = {"u1": ["a", "b"], "u2": ["x", "y"]}

    recall = evaluar_recall_personalizado(val_df, recomendaciones, k=2)

    # u1: "a" esta en el top-2 -> 1.0; u2: "z" no esta -> 0.0
    assert recall == 0.5
