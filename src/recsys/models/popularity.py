"""Modelo baseline de popularidad con score bayesiano."""

from __future__ import annotations

import pandas as pd


def fit_popularity(interacciones: pd.DataFrame) -> pd.DataFrame:
    """Calcula un ranking de libros por popularidad usando un score bayesiano.

    score = (n / (n + C)) * avg_rating + (C / (C + n)) * m

    donde n es la cantidad de interacciones del libro, avg_rating su rating
    promedio, C el promedio de interacciones por libro y m el rating
    promedio global. C y m se calculan únicamente a partir de los datos
    recibidos en `interacciones`, para poder usar esta función tanto con
    train como con el dataset completo.

    Devuelve un DataFrame con columnas [id_libro, n, avg_rating, score],
    ordenado por score descendente.
    """
    stats = (
        interacciones.groupby("id_libro")["rating"]
        .agg(n="count", avg_rating="mean")
        .reset_index()
    )

    C = stats["n"].mean()
    m = interacciones["rating"].mean()

    stats["score"] = (stats["n"] / (stats["n"] + C)) * stats["avg_rating"] + (
        C / (C + stats["n"])
    ) * m

    return stats.sort_values("score", ascending=False).reset_index(drop=True)
