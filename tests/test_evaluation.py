"""Tests para las métricas de evaluación (NDCG@k, recall@k)."""

import pandas as pd
import pytest

from recsys.evaluation import (
    evaluar_multisplit,
    evaluar_ndcg,
    evaluar_ndcg_ponderado_por_actividad,
    evaluar_recall_personalizado,
    ndcg_at_k,
    pesos_por_actividad,
    recall_at_k,
)


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


def test_evaluar_multisplit_reporta_media_y_desvio():
    resultados_por_seed = {1: 0.10, 2: 0.20, 3: 0.30}

    resultado = evaluar_multisplit(lambda seed: resultados_por_seed[seed], seeds=[1, 2, 3])

    assert resultado["valores"] == [0.10, 0.20, 0.30]
    assert resultado["media"] == pytest.approx(0.20)
    assert resultado["desvio"] > 0.0


def test_evaluar_multisplit_un_solo_seed_desvio_cero():
    resultado = evaluar_multisplit(lambda seed: 0.5, seeds=[1])

    assert resultado["media"] == 0.5
    assert resultado["desvio"] == 0.0


def test_pesos_por_actividad_bucketiza_por_proporcion():
    # 1 usuario en [0,2), 2 en [2,5), 1 en [5,20)
    n_interacciones = pd.Series([1, 3, 4, 15])

    pesos = pesos_por_actividad(n_interacciones, bins=[0, 2, 5, 20])

    assert pesos[pd.Interval(0, 2, closed="left")] == pytest.approx(0.25)
    assert pesos[pd.Interval(2, 5, closed="left")] == pytest.approx(0.5)
    assert pesos[pd.Interval(5, 20, closed="left")] == pytest.approx(0.25)


def test_evaluar_ndcg_ponderado_por_actividad_pesa_distinto_que_el_promedio_parejo():
    bins = [0, 5, float("inf")]

    # poblacion de referencia (ej. ejemplo.csv): 1/3 liviana, 2/3 pesada
    pesos = pesos_por_actividad(pd.Series([1, 1, 10, 10, 10, 10]), bins=bins)

    # validacion local: u1/u2 livianos (NDCG 1.0 y 0.0), u3 pesado (NDCG 1.0)
    val_df = pd.DataFrame({"id_lector": ["u1", "u2", "u3"], "id_libro": ["a", "b", "c"]})
    recomendaciones = {"u1": ["a"], "u2": ["x"], "u3": ["c"]}
    n_interacciones_por_usuario = {"u1": 1, "u2": 1, "u3": 10}

    promedio_parejo = sum(
        ndcg_at_k(recomendaciones[u], {val_df.set_index("id_lector")["id_libro"][u]}, k=1)
        for u in ["u1", "u2", "u3"]
    ) / 3
    assert promedio_parejo == pytest.approx(2 / 3)

    ponderado = evaluar_ndcg_ponderado_por_actividad(
        val_df, recomendaciones, k=1, n_interacciones_por_usuario=n_interacciones_por_usuario,
        pesos_por_bucket=pesos, bins=bins,
    )

    # bucket liviano: NDCG medio (1.0+0.0)/2=0.5, peso 1/3; bucket pesado:
    # NDCG medio 1.0, peso 2/3 -> ponderado = 1/3*0.5 + 2/3*1.0 = 0.8333,
    # bien distinto del promedio parejo (2/3)
    assert ponderado == pytest.approx(1 / 3 * 0.5 + 2 / 3 * 1.0)
    assert ponderado != pytest.approx(promedio_parejo)


def test_evaluar_ndcg_ponderado_por_actividad_ignora_buckets_sin_datos_locales():
    bins = [0, 5, float("inf")]
    # la poblacion de referencia tiene usuarios pesados, pero la
    # validacion local no tiene NINGUN usuario pesado -- ese bucket se
    # ignora (renormaliza sobre lo que si hay) en vez de contar como 0.
    pesos = pesos_por_actividad(pd.Series([1, 10, 10]), bins=bins)

    val_df = pd.DataFrame({"id_lector": ["u1"], "id_libro": ["a"]})
    recomendaciones = {"u1": ["a"]}
    n_interacciones_por_usuario = {"u1": 1}

    ponderado = evaluar_ndcg_ponderado_por_actividad(
        val_df, recomendaciones, k=1, n_interacciones_por_usuario=n_interacciones_por_usuario,
        pesos_por_bucket=pesos, bins=bins,
    )

    assert ponderado == pytest.approx(1.0)
