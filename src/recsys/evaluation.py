"""Métricas de evaluación para el sistema de recomendación (NDCG@k, recall@k)."""

from __future__ import annotations

import math
import statistics
from typing import Callable

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


def evaluar_multisplit(fn_entrenar_y_evaluar: Callable[[int], float], seeds: list) -> dict:
    """Corre `fn_entrenar_y_evaluar(seed)` (entrena + evalúa un pipeline
    completo, devuelve un NDCG@k u otra métrica) sobre varios `seeds` y
    reporta media y desvío en vez de un solo número.

    Nace de un caso real: un sweep de hiperparámetros de ALS sobre un
    único split (`seed=42`) encontró un config que mejoraba el NDCG local
    +11.5% pero empeoraba el score real de Kaggle -13.5% -- sobreajuste
    al ruido específico de ese split. Evaluar sobre varios seeds y mirar
    el desvío, no solo la media, ayuda a detectar mejoras frágiles que no
    generalizan antes de confiar en ellas (ver `experiments/bitacora.md`,
    sección "Regresión en Kaggle").

    Devuelve {"valores": [...], "media": ..., "desvio": ...} (`desvio` es
    el desvío estándar muestral; 0.0 si hay un solo seed).
    """
    valores = [fn_entrenar_y_evaluar(seed) for seed in seeds]
    media = sum(valores) / len(valores)
    desvio = statistics.stdev(valores) if len(valores) > 1 else 0.0
    return {"valores": valores, "media": media, "desvio": desvio}


def recall_at_k(recomendados: list, relevantes: set, k: int) -> float:
    """Recall@k: qué fracción de los `relevantes` aparece en los primeros
    k `recomendados`.

    Métrica de *diagnóstico*, no de entrega: separa un problema de
    cobertura (el libro correcto ni siquiera está entre los candidatos
    del modelo) de uno de ranking (está, pero mal ordenado dentro del
    top-k real de la entrega). Con un solo libro relevante por usuario
    (el caso de este proyecto, `n_val=1`), equivale a "el libro correcto
    apareció en el top-k, sí o no".
    """
    if k <= 0 or not relevantes:
        return 0.0
    encontrados = len(set(recomendados[:k]) & relevantes)
    return encontrados / len(relevantes)


def evaluar_recall_personalizado(
    val_df: pd.DataFrame,
    recomendaciones: dict,
    k: int,
) -> float:
    """Promedia el recall@k sobre los usuarios de `val_df`, usando un
    ranking ya armado y filtrado por usuario. Mismo patrón que
    `evaluar_ndcg_personalizado`, útil para medir Recall@200 (o cualquier
    k mayor al de entrega) como diagnóstico de cobertura del modelo.
    """
    relevantes_por_usuario = val_df.groupby("id_lector")["id_libro"].agg(set)

    scores = []
    for id_lector, relevantes in relevantes_por_usuario.items():
        recomendados = recomendaciones.get(id_lector, [])[:k]
        scores.append(recall_at_k(recomendados, relevantes, k))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)
