"""Popularidad segmentada por género literario, con fallback a franja de
nacimiento y a popularidad global.

Cadena de fallback por usuario: si conocemos su género literario preferido,
recomendamos primero entre lo más popular de ese género. Si no alcanza para
completar k candidatos (o no conocemos su género), completamos con lo más
popular de su franja de nacimiento. Si tampoco alcanza o no la conocemos,
completamos con la popularidad global (v0).
"""

from __future__ import annotations

import unicodedata

import pandas as pd

from recsys.models.popularity import fit_popularity


def _quitar_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def _normalizar_genero(generos: pd.Series) -> pd.Series:
    """Normaliza el género literario para agrupar variantes de capitalización
    y acentuación (ej. "Histórica y aventuras" / "HIstórica Y Aventuras" /
    "historica y aventuras" son el mismo género). Strings vacíos quedan
    como NaN, igual que los géneros nulos.

    La acentuación inconsistente en el dato crudo no es solo un problema
    cosmético: separa en dos categorías lo que es la misma categoría real
    (ej. "clásicos de la literatura" con 754 libros vs "clasicos de la
    literatura" con 2 -- claramente un typo de tilde, no un género
    distinto). Ignorar acentos los une sin necesitar un mapa de alias a
    mano -- confirmado con los datos reales, no se encontró ningún caso
    en el que dos géneros temáticamente distintos coincidieran al
    ignorar acentos.
    """
    normalizado = generos.str.strip().str.lower().map(_quitar_acentos, na_action="ignore")
    return normalizado.replace("", pd.NA)


# Macro-géneros (taxonomía de dominio, acordada con el usuario a partir de
# la distribución real de las 52 categorías granulares limpias -- ver
# `experiments/bitacora.md`). Se listan explícitamente las categorías de
# las 9 familias con identidad temática propia; todo lo que no aparezca acá
# (~27 categorías minúsculas: humor, autoayuda, cocina, economía, música,
# deportes, medicina, derecho, idiomas, ...) cae en el catch-all por
# default -- no hace falta enumerarlas.
MACRO_GENERO_POR_GENERO: dict[str, str] = {
    "narrativa": "narrativa y clasicos",
    "ficcion literaria": "narrativa y clasicos",
    "literatura contemporanea": "narrativa y clasicos",
    "clasicos de la literatura": "narrativa y clasicos",
    "novela": "narrativa y clasicos",
    "novela negra, intriga, terror": "novela negra y suspenso",
    "novela negra": "novela negra y suspenso",
    "fantastica, ciencia ficcion": "fantastico y ciencia ficcion",
    "romantica, erotica": "romantica y erotica",
    "historica y aventuras": "historica y aventuras",
    "infantil y juvenil": "infantil y juvenil",
    "juvenil": "infantil y juvenil",
    "lecturas complementarias": "infantil y juvenil",
    "comics, novela grafica": "comic y novela grafica",
    "ensayo": "ensayo, biografia y no ficcion",
    "biografias, memorias": "ensayo, biografia y no ficcion",
    "no ficcion": "ensayo, biografia y no ficcion",
    "historia": "ensayo, biografia y no ficcion",
    "historia militar": "ensayo, biografia y no ficcion",
    "historia del cine": "ensayo, biografia y no ficcion",
    "filosofia contemporanea": "ensayo, biografia y no ficcion",
    "feminismo y mujer": "ensayo, biografia y no ficcion",
    "naturaleza y ciencia": "ensayo, biografia y no ficcion",
    "poesia, teatro": "poesia y teatro",
    "poesia": "poesia y teatro",
}
MACRO_GENERO_DEFAULT = "practico y miscelaneo"


def genero_macro(genero_normalizado: str | float | None) -> str | None:
    """Mapea un género ya normalizado (ver `_normalizar_genero`) a su
    macro-género (10 familias de dominio). `None`/NaN se preserva tal
    cual -- un género desconocido sigue siendo desconocido, no cae al
    catch-all (que es para géneros *conocidos* pero de nicho)."""
    if genero_normalizado is None or pd.isna(genero_normalizado):
        return None
    return MACRO_GENERO_POR_GENERO.get(genero_normalizado, MACRO_GENERO_DEFAULT)


def normalizar_genero_macro(generos: pd.Series) -> pd.Series:
    """Atajo: `_normalizar_genero` + `genero_macro` en un solo paso, para
    no repetir el join en cada lugar que necesita el macro-género de un
    libro a partir de su columna `genero` cruda."""
    return _normalizar_genero(generos).map(genero_macro, na_action="ignore")


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


def fit_popularidad_por_genero_macro(interacciones: pd.DataFrame, libros: pd.DataFrame) -> dict:
    """Calco de `fit_popularity_por_genero`, pero agrupando por
    macro-género (`normalizar_genero_macro`, 10 familias de dominio) en
    vez de por las 52 categorías granulares -- varias de esas categorías
    tienen muy pocas interacciones para un score bayesiano confiable; a
    nivel macro hay volumen real en las 10 familias.

    Devuelve {macro_genero: DataFrame} -- mismo formato que
    `fit_popularity`.
    """
    generos = libros[["id_libro", "genero"]].copy()
    generos["genero_macro"] = normalizar_genero_macro(generos["genero"])

    merged = interacciones.merge(
        generos[["id_libro", "genero_macro"]], on="id_libro", how="left"
    ).dropna(subset=["genero_macro"])

    return {
        genero_macro: fit_popularity(grupo[["id_libro", "rating"]])
        for genero_macro, grupo in merged.groupby("genero_macro")
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
