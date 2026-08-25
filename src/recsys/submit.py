"""Arma el csv de entrega para Kaggle."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from recsys.data import libros_leidos_por_usuario, load_interacciones
from recsys.models.popularity import fit_popularity

ROOT_DIR = Path(__file__).resolve().parents[2]
EJEMPLO_PATH = ROOT_DIR / "data" / "raw" / "ejemplo.csv"
SUBMISSIONS_DIR = ROOT_DIR / "outputs" / "submissions"

MODELOS = {"popularity": fit_popularity}


def armar_submission(
    ejemplo_df: pd.DataFrame,
    ranking_global: list,
    libros_leidos: dict,
) -> pd.DataFrame:
    """Arma el DataFrame de entrega.

    Para cada usuario de `ejemplo_df` (en el orden en que aparecen) toma
    `ranking_global`, filtra los libros que ya leyó y completa tantos
    candidatos como filas tenga ese usuario en `ejemplo_df` (su k),
    preservando el orden del ranking.
    """
    filas = []
    for id_lector, grupo in ejemplo_df.groupby("id_lector", sort=False):
        k = len(grupo)
        leidos = libros_leidos.get(id_lector, set())
        candidatos = [libro for libro in ranking_global if libro not in leidos][:k]
        filas.extend({"id_lector": id_lector, "id_libro": libro} for libro in candidatos)

    return pd.DataFrame(filas, columns=["id_lector", "id_libro"])


def generar_submission(model: str) -> Path:
    """Entrena `model` con todos los datos y guarda el csv de entrega."""
    if model not in MODELOS:
        raise ValueError(f"Modelo desconocido: {model!r}. Opciones: {sorted(MODELOS)}")

    interacciones = load_interacciones()
    libros_leidos = libros_leidos_por_usuario(interacciones)
    ranking_global = MODELOS[model](interacciones)["id_libro"].tolist()

    ejemplo_df = pd.read_csv(EJEMPLO_PATH)
    submission = armar_submission(ejemplo_df, ranking_global, libros_leidos)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS_DIR / f"{model}.csv"
    submission.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un csv de entrega para Kaggle.")
    parser.add_argument("--model", choices=sorted(MODELOS), required=True)
    args = parser.parse_args()

    out_path = generar_submission(args.model)
    print(f"Submission guardada en {out_path}")


if __name__ == "__main__":
    main()
