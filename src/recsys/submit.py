"""Arma el csv de entrega para Kaggle."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from recsys.data import (
    libros_leidos_por_usuario,
    load_interacciones,
    load_lectores,
    load_libros,
)
from recsys.models.als import fit_als
from recsys.models.als import recomendar_por_usuario as recomendar_por_usuario_als
from recsys.models.popularity import fit_popularity
from recsys.models.popularity_segmentada import (
    fit_popularity_por_franja_nacimiento,
    fit_popularity_por_genero,
    franja_nacimiento_por_usuario,
    genero_preferido_por_usuario,
    recomendar_por_usuario,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
EJEMPLO_PATH = ROOT_DIR / "data" / "raw" / "ejemplo.csv"
SUBMISSIONS_DIR = ROOT_DIR / "outputs" / "submissions"


def _recomendaciones_popularity(usuarios: list, k: int) -> dict:
    """v0: mismo ranking global de popularidad para todos los usuarios."""
    interacciones = load_interacciones()
    libros_leidos = libros_leidos_por_usuario(interacciones)
    ranking_global = fit_popularity(interacciones)["id_libro"].tolist()

    return {
        id_lector: [
            libro
            for libro in ranking_global
            if libro not in libros_leidos.get(id_lector, set())
        ][:k]
        for id_lector in usuarios
    }


def _recomendaciones_popularity_segmentada(usuarios: list, k: int) -> dict:
    """v1: ranking por usuario con fallback género -> franja de nacimiento -> global."""
    interacciones = load_interacciones()
    lectores = load_lectores()
    libros = load_libros()
    libros_leidos = libros_leidos_por_usuario(interacciones)

    return recomendar_por_usuario(
        usuarios=usuarios,
        ranking_por_genero=fit_popularity_por_genero(interacciones, libros),
        genero_por_usuario=genero_preferido_por_usuario(interacciones, libros),
        ranking_por_franja=fit_popularity_por_franja_nacimiento(interacciones, lectores),
        franja_por_usuario=franja_nacimiento_por_usuario(lectores),
        ranking_global=fit_popularity(interacciones)["id_libro"].tolist(),
        libros_leidos=libros_leidos,
        k=k,
    )


def _recomendaciones_als(usuarios: list, k: int) -> dict:
    """v2: filtrado colaborativo ALS sobre ratings, fallback a popularidad global.

    Se evaluó (y se descartó) rutear usuarios livianos hacia la
    popularidad por género de v1 en vez de ALS, bajo la hipótesis de que
    el embedding de ALS para poca actividad sería poco confiable. Bajo el
    split corregido (temporal, n_val=1) los datos no lo respaldan: ALS le
    gana a género en *todos* los buckets de actividad medidos, incluso
    con una sola interacción de historial (ver
    `recsys.models.als.recomendar_hibrido` y `experiments/bitacora.md`
    para el detalle). Se mantiene ALS puro acá.
    """
    interacciones = load_interacciones()
    libros_leidos = libros_leidos_por_usuario(interacciones)
    modelo, matriz, fila_por_usuario, libros_por_columna = fit_als(interacciones)

    return recomendar_por_usuario_als(
        usuarios=usuarios,
        modelo=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario=fila_por_usuario,
        libros_por_columna=libros_por_columna,
        ranking_global=fit_popularity(interacciones)["id_libro"].tolist(),
        libros_leidos=libros_leidos,
        k=k,
    )


# Cada modelo mapea a una función (usuarios, k) -> {id_lector: [id_libro, ...]}
# ya entrenada con todos los datos y con los libros ya leídos filtrados.
MODELOS = {
    "popularity": _recomendaciones_popularity,
    "popularity_segmentada": _recomendaciones_popularity_segmentada,
    "als": _recomendaciones_als,
}


def armar_submission(ejemplo_df: pd.DataFrame, recomendaciones: dict) -> pd.DataFrame:
    """Arma el DataFrame de entrega a partir de recomendaciones ya armadas por usuario.

    Para cada usuario de `ejemplo_df` (en el orden en que aparecen) toma
    tantas filas de `recomendaciones[id_lector]` como filas tenga ese
    usuario en `ejemplo_df` (su k), preservando el orden del ranking.
    """
    filas = []
    for id_lector, grupo in ejemplo_df.groupby("id_lector", sort=False):
        k = len(grupo)
        candidatos = recomendaciones.get(id_lector, [])[:k]
        filas.extend({"id_lector": id_lector, "id_libro": libro} for libro in candidatos)

    return pd.DataFrame(filas, columns=["id_lector", "id_libro"])


def _nombre_submission(model: str, tag: str | None) -> str:
    """Arma un nombre de archivo único con timestamp -- nunca pisa una
    submission anterior (cada corrida es una versión distinta, y las que
    ya se subieron a Kaggle no se pueden regenerar del código actual sin
    volver al commit correspondiente)."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sufijo_tag = ""
    if tag:
        tag_limpio = re.sub(r"[^a-zA-Z0-9._-]+", "-", tag).strip("-")
        if tag_limpio:
            sufijo_tag = f"_{tag_limpio}"

    base = f"{model}_{timestamp}{sufijo_tag}"
    nombre = f"{base}.csv"
    contador = 2
    while (SUBMISSIONS_DIR / nombre).exists():
        nombre = f"{base}-{contador}.csv"
        contador += 1
    return nombre


def generar_submission(model: str, tag: str | None = None) -> Path:
    """Entrena `model` con todos los datos y guarda el csv de entrega.

    El archivo se guarda como `{model}_{timestamp}[_{tag}].csv` -- nunca
    con el nombre pelado `{model}.csv`, para no pisar submissions
    anteriores (en particular, las que ya se subieron a Kaggle).
    """
    if model not in MODELOS:
        raise ValueError(f"Modelo desconocido: {model!r}. Opciones: {sorted(MODELOS)}")

    ejemplo_df = pd.read_csv(EJEMPLO_PATH)
    usuarios = ejemplo_df["id_lector"].unique().tolist()
    k = int(ejemplo_df.groupby("id_lector").size().max())

    recomendaciones = MODELOS[model](usuarios, k)
    submission = armar_submission(ejemplo_df, recomendaciones)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS_DIR / _nombre_submission(model, tag)
    submission.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un csv de entrega para Kaggle.")
    parser.add_argument("--model", choices=sorted(MODELOS), required=True)
    parser.add_argument("--tag", default=None, help="Sufijo descriptivo opcional para el nombre del archivo.")
    args = parser.parse_args()

    out_path = generar_submission(args.model, tag=args.tag)
    print(f"Submission guardada en {out_path}")


if __name__ == "__main__":
    main()
