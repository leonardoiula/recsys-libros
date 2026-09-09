"""Modelo baseline de popularidad con score bayesiano, para comics.

Misma fórmula y mismo motivo que `recsys.models.popularity.fit_popularity`
(ver ese docstring para el detalle del porqué de un shrinkage bayesiano en
vez de un promedio simple): un comic con pocas reviews y un rating alto por
azar no debería rankear por encima de uno con atractivo ampliamente
comprobado. Se reimplementa acá por el nombre de columna (`id_comic` en vez
de `id_libro`) -- el valor de `C` que resultó mejor para libros (`n.mean()`)
es un punto de partida razonable, pero habría que re-validarlo con un sweep
propio contra el NDCG@k de comics antes de asumir que generaliza: la
distribución de reviews por comic no es necesariamente la misma que la de
ratings por libro.
"""

from __future__ import annotations

import pandas as pd


def fit_popularity(interacciones: pd.DataFrame, C: float | None = None) -> pd.DataFrame:
    """score = (n / (n + C)) * avg_rating + (C / (C + n)) * m

    Devuelve un DataFrame [id_comic, n, avg_rating, score], ordenado por
    score descendente. `C` por default es `n.mean()` si no se pasa.
    """
    stats = (
        interacciones.groupby("id_comic")["rating"]
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
