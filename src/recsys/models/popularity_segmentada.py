"""Popularidad segmentada por género literario, con fallback a franja de
nacimiento y a popularidad global.

Cadena de fallback por usuario: si conocemos su género literario preferido,
recomendamos primero entre lo más popular de ese género. Si no alcanza para
completar k candidatos (o no conocemos su género), completamos con lo más
popular de su franja de nacimiento. Si tampoco alcanza o no la conocemos,
completamos con la popularidad global (v0).
"""

from __future__ import annotations

import pandas as pd

from recsys.models.popularity import fit_popularity


def _normalizar_genero(generos: pd.Series) -> pd.Series:
    """Normaliza el género literario para agrupar variantes de capitalización
    (ej. "Histórica y aventuras" / "HIstórica Y Aventuras" son el mismo
    género). Strings vacíos quedan como NaN, igual que los géneros nulos.
    """
    normalizado = generos.str.strip().str.lower()
    return normalizado.replace("", pd.NA)


def genero_preferido_por_usuario(interacciones: pd.DataFrame, libros: pd.DataFrame) -> dict:
    """Género literario más leído por cada usuario.

    Devuelve {id_lector: genero} solo para usuarios con al menos una
    interacción con un libro de género conocido.
    """
    generos = libros[["id_libro", "genero"]].copy()
    generos["genero"] = _normalizar_genero(generos["genero"])

    merged = interacciones.merge(generos, on="id_libro", how="left").dropna(subset=["genero"])
    conteos = merged.groupby(["id_lector", "genero"]).size().rename("n").reset_index()

    idx_preferido = conteos.groupby("id_lector")["n"].idxmax()
    preferido = conteos.loc[idx_preferido].set_index("id_lector")["genero"]
    return preferido.to_dict()


def fit_popularity_por_genero(interacciones: pd.DataFrame, libros: pd.DataFrame) -> dict:
    """Ranking de popularidad (score bayesiano) dentro de cada género literario.

    Devuelve {genero: DataFrame} — mismo formato que `fit_popularity`, pero
    con C y m calculados únicamente con las interacciones de ese género.
    """
    generos = libros[["id_libro", "genero"]].copy()
    generos["genero"] = _normalizar_genero(generos["genero"])

    merged = interacciones.merge(generos, on="id_libro", how="left").dropna(subset=["genero"])

    return {
        genero: fit_popularity(grupo[["id_libro", "rating"]])
        for genero, grupo in merged.groupby("genero")
    }


def franja_nacimiento_por_usuario(lectores: pd.DataFrame) -> dict:
    """Década de nacimiento de cada lector (ej. "1980s"), como proxy de edad.

    Se usa década en vez de edad real porque no hay una fecha de referencia
    confiable para calcularla a partir de `nacimiento`. Devuelve
    {id_lector: franja} solo para lectores con `nacimiento` numérico válido.
    """
    nacimiento = pd.to_numeric(lectores["nacimiento"], errors="coerce")
    franja = ((nacimiento // 10) * 10).astype("Int64")
    franja_str = franja.astype("string") + "s"

    resultado = pd.Series(franja_str.values, index=lectores["id_lector"])
    return resultado.dropna().to_dict()


def fit_popularity_por_franja_nacimiento(interacciones: pd.DataFrame, lectores: pd.DataFrame) -> dict:
    """Ranking de popularidad (score bayesiano) dentro de cada franja de
    nacimiento. Devuelve {franja: DataFrame}, mismo formato que `fit_popularity`.
    """
    franjas = franja_nacimiento_por_usuario(lectores)

    con_franja = interacciones.copy()
    con_franja["franja"] = con_franja["id_lector"].map(franjas)
    con_franja = con_franja.dropna(subset=["franja"])

    return {
        franja: fit_popularity(grupo[["id_libro", "rating"]])
        for franja, grupo in con_franja.groupby("franja")
    }


def recomendar_por_usuario(
    usuarios: list,
    ranking_por_genero: dict,
    genero_por_usuario: dict,
    ranking_por_franja: dict,
    franja_por_usuario: dict,
    ranking_global: list,
    libros_leidos: dict,
    k: int,
) -> dict:
    """Arma el ranking recomendado de cada usuario con la cadena de fallback
    género -> franja de nacimiento -> popularidad global.

    Si una fuente no alcanza para completar los k candidatos (porque el
    usuario ya leyó buena parte de ese segmento, o porque no conocemos su
    género/franja), se completa con la siguiente fuente de la cadena, sin
    repetir libros ya recomendados ni libros ya leídos.
    """
    recomendaciones = {}

    for id_lector in usuarios:
        vistos = set(libros_leidos.get(id_lector, set()))
        candidatos: list = []

        fuentes = []
        genero = genero_por_usuario.get(id_lector)
        if genero is not None and genero in ranking_por_genero:
            fuentes.append(ranking_por_genero[genero]["id_libro"].tolist())

        franja = franja_por_usuario.get(id_lector)
        if franja is not None and franja in ranking_por_franja:
            fuentes.append(ranking_por_franja[franja]["id_libro"].tolist())

        fuentes.append(ranking_global)

        for fuente in fuentes:
            if len(candidatos) >= k:
                break
            for libro in fuente:
                if len(candidatos) >= k:
                    break
                if libro in vistos:
                    continue
                candidatos.append(libro)
                vistos.add(libro)

        recomendaciones[id_lector] = candidatos

    return recomendaciones
