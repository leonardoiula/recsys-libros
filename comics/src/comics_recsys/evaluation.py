"""Métricas de evaluación para el recomendador de comics.

Los building blocks que NO dependen de nombres de columna (`ndcg_at_k`,
`recall_at_k`, `evaluar_multisplit`, `pesos_por_actividad`, y las constantes
de bucketing de actividad) se reusan tal cual de `recsys.evaluation` -- son
agnósticos de dominio, no tiene sentido duplicarlos. Acá solo reimplementamos
los wrappers que sí hardcodean `id_lector`/`id_libro`, adaptados a
`id_usuario`/`id_comic`.
"""

from __future__ import annotations

import pandas as pd
from recsys.evaluation import (  # noqa: F401 (reexportados para uso desde comics_recsys.evaluation)
    BINS_ACTIVIDAD_DEFAULT,
    evaluar_multisplit,
    ndcg_at_k,
    pesos_por_actividad,
    recall_at_k,
)


def evaluar_ndcg(
    val_df: pd.DataFrame,
    ranking_global: list,
    comics_calificados: dict,
    k: int,
) -> float:
    """Igual que `recsys.evaluation.evaluar_ndcg`: promedia NDCG@k aplicando
    el mismo `ranking_global` a todos los usuarios, filtrando lo ya calificado."""
    relevantes_por_usuario = val_df.groupby("id_usuario")["id_comic"].agg(set)

    scores = []
    for id_usuario, relevantes in relevantes_por_usuario.items():
        calificados = comics_calificados.get(id_usuario, set())
        recomendados = [comic for comic in ranking_global if comic not in calificados][:k]
        scores.append(ndcg_at_k(recomendados, relevantes, k))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def evaluar_ndcg_personalizado(
    val_df: pd.DataFrame,
    recomendaciones: dict,
    k: int,
) -> float:
    """Igual que `recsys.evaluation.evaluar_ndcg_personalizado`, con un
    ranking ya armado y filtrado por usuario (modelos personalizados)."""
    relevantes_por_usuario = val_df.groupby("id_usuario")["id_comic"].agg(set)

    scores = []
    for id_usuario, relevantes in relevantes_por_usuario.items():
        recomendados = recomendaciones.get(id_usuario, [])[:k]
        scores.append(ndcg_at_k(recomendados, relevantes, k))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def evaluar_recall_personalizado(
    val_df: pd.DataFrame,
    recomendaciones: dict,
    k: int,
) -> float:
    """Igual que `recsys.evaluation.evaluar_recall_personalizado` -- métrica
    de diagnóstico (cobertura de candidatos), no de entrega."""
    relevantes_por_usuario = val_df.groupby("id_usuario")["id_comic"].agg(set)

    scores = []
    for id_usuario, relevantes in relevantes_por_usuario.items():
        recomendados = recomendaciones.get(id_usuario, [])[:k]
        scores.append(recall_at_k(recomendados, relevantes, k))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def evaluar_ndcg_ponderado_por_actividad(
    val_df: pd.DataFrame,
    recomendaciones: dict,
    k: int,
    n_interacciones_por_usuario: dict,
    pesos_por_bucket: pd.Series,
    bins: list = BINS_ACTIVIDAD_DEFAULT,
) -> float:
    """Igual que `recsys.evaluation.evaluar_ndcg_ponderado_por_actividad` --
    diagnóstico para chequear si la brecha entre validación local y un
    subconjunto de referencia es composición de población o ruido de muestra."""
    relevantes_por_usuario = val_df.groupby("id_usuario")["id_comic"].agg(set)

    filas = [
        {
            "n": n_interacciones_por_usuario.get(id_usuario, 0),
            "ndcg": ndcg_at_k(recomendaciones.get(id_usuario, [])[:k], relevantes, k),
        }
        for id_usuario, relevantes in relevantes_por_usuario.items()
    ]
    if not filas:
        return 0.0

    df = pd.DataFrame(filas)
    df["bucket"] = pd.cut(df["n"], bins=bins, right=False)
    ndcg_por_bucket = df.groupby("bucket", observed=True)["ndcg"].mean()

    total_peso = 0.0
    total_ponderado = 0.0
    for bucket, peso in pesos_por_bucket.items():
        if bucket in ndcg_por_bucket.index:
            total_ponderado += peso * ndcg_por_bucket[bucket]
            total_peso += peso

    return total_ponderado / total_peso if total_peso else 0.0
