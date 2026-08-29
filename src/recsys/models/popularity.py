"""Modelo baseline de popularidad con score bayesiano."""

from __future__ import annotations

import pandas as pd


def fit_popularity(interacciones: pd.DataFrame, C: float | None = None) -> pd.DataFrame:
    """Calcula un ranking de libros por popularidad usando un score bayesiano.

    score = (n / (n + C)) * avg_rating + (C / (C + n)) * m

    donde n es la cantidad de interacciones del libro, avg_rating su rating
    promedio, y m el rating promedio global (siempre calculado a partir de
    los datos recibidos, para poder usar esta función tanto con train como
    con el dataset completo).

    `C` es configurable porque no hay un único valor "correcto": es el
    umbral de interacciones a partir del cual empezamos a confiar más en
    el rating propio del libro que en el promedio global. Si no se pasa
    explícitamente, se calcula con `n.mean()`. Se sospechó que esto era
    demasiado agresivo (la distribución de interacciones por libro es
    fuertemente right-skewed -- mediana 2, media ~9.6 -- así que deja al
    ~86% del catálogo con más peso en `m` que en su propio rating) y se
    probaron alternativas menos agresivas (mediana, media geométrica) con
    un sweep contra NDCG@20 real bajo el split corregido. Sorpresa: la
    media actual ganó por lejos (0.0066 vs 0.0007 geométrica vs 0.0002
    mediana) -- con un solo libro relevante por usuario en validación
    (leave-one-out estricto), lo que más ayuda es un ranking dominado por
    libros de atractivo ampliamente comprobado; un `C` chico deja subir
    demasiado a libros de nicho con 2-3 ratings altos por azar, diluyendo
    el top-k. Ver `experiments/bitacora.md` para el detalle del sweep.

    Devuelve un DataFrame con columnas [id_libro, n, avg_rating, score],
    ordenado por score descendente.
    """
    stats = (
        interacciones.groupby("id_libro")["rating"]
        .agg(n="count", avg_rating="mean")
        .reset_index()
    )

    if C is None:
        C = stats["n"].mean()
    m = interacciones["rating"].mean()

    stats["score"] = (stats["n"] / (stats["n"] + C)) * stats["avg_rating"] + (
        C / (C + stats["n"])
    ) * m

    return stats.sort_values("score", ascending=False).reset_index(drop=True)
