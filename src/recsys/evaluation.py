"""Métricas de evaluación para el sistema de recomendación (NDCG@k)."""

from __future__ import annotations

import math

import pandas as pd


def ndcg_at_k(recomendados: list, relevantes: set, k: int) -> float:
    """NDCG@k de una lista de recomendados contra un conjunto de relevantes.

    Usa descuento 1/log2(posicion+2) (posición 0-indexed) y normaliza por
    el IDCG del ranking ideal (todos los relevantes disponibles, hasta k,
    al principio del ranking).
    """
    if k <= 0 or not relevantes:
        return 0.0

    dcg = sum(
        1.0 / math.log2(pos + 2)
        for pos, item in enumerate(recomendados[:k])
        if item in relevantes
    )

    n_ideal = min(len(relevantes), k)
    idcg = sum(1.0 / math.log2(pos + 2) for pos in range(n_ideal))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluar_ndcg(
    val_df: pd.DataFrame,
    ranking_global: list,
    libros_leidos: dict,
    k: int,
) -> float:
    """Promedia el NDCG@k sobre todos los usuarios de `val_df`.

    Para cada usuario arma su ranking recomendado tomando `ranking_global`
    y filtrando los libros que ya leyó (según `libros_leidos`), y lo compara
    contra los libros que efectivamente leyó en validación.
    """
    relevantes_por_usuario = val_df.groupby("id_lector")["id_libro"].agg(set)

    scores = []
    for id_lector, relevantes in relevantes_por_usuario.items():
        leidos = libros_leidos.get(id_lector, set())
        recomendados = [libro for libro in ranking_global if libro not in leidos][:k]
        scores.append(ndcg_at_k(recomendados, relevantes, k))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def evaluar_ndcg_personalizado(
    val_df: pd.DataFrame,
    recomendaciones: dict,
    k: int,
) -> float:
    """Promedia el NDCG@k sobre los usuarios de `val_df`, usando un ranking
    ya armado y filtrado por usuario (a diferencia de `evaluar_ndcg`, que
    aplica el mismo ranking global a todos). Pensada para modelos
    personalizados como popularidad segmentada por género.
    """
    relevantes_por_usuario = val_df.groupby("id_lector")["id_libro"].agg(set)

    scores = []
    for id_lector, relevantes in relevantes_por_usuario.items():
        recomendados = recomendaciones.get(id_lector, [])[:k]
        scores.append(ndcg_at_k(recomendados, relevantes, k))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)
