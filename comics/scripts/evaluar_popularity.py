"""
Entrena+evalúa el baseline de popularidad sobre comics/data/raw/comics.db y
loguea el resultado en comics/experiments/log.csv (mismo formato que el log
del proyecto de libros). No hace submit a ningún lado -- este TP no tiene
leaderboard externo, por eso `ndcg_kaggle` queda vacío.

Uso: uv run python comics/scripts/evaluar_popularity.py [--k 20] [--seeds 42 7 123]
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comics_recsys.data import comics_calificados_por_usuario, load_interacciones, split_train_val  # noqa: E402
from comics_recsys.evaluation import evaluar_multisplit, evaluar_ndcg  # noqa: E402
from comics_recsys.models.popularity import fit_popularity  # noqa: E402

COMICS_DIR = Path(__file__).resolve().parents[1]
LOG_PATH = COMICS_DIR / "experiments" / "log.csv"
LOG_COLUMNAS = ["fecha", "modelo", "k", "frac_val", "seed", "ndcg_local", "ndcg_kaggle", "notas"]


def entrenar_y_evaluar(interacciones, k: int, seed: int) -> float:
    train, val = split_train_val(interacciones, n_val=1, seed=seed)
    ranking_global = fit_popularity(train)["id_comic"].tolist()
    calificados = comics_calificados_por_usuario(train)
    return evaluar_ndcg(val, ranking_global, calificados, k)


def loguear(modelo: str, k: int, seeds: list[int], resultado: dict, notas: str) -> None:
    nuevo = not LOG_PATH.exists()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if nuevo:
            writer.writerow(LOG_COLUMNAS)
        writer.writerow(
            [
                date.today().isoformat(),
                modelo,
                k,
                "",
                ",".join(str(s) for s in seeds),
                f"{resultado['media']:.6f}",
                "",
                notas,
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    args = parser.parse_args()

    interacciones = load_interacciones()
    resultado = evaluar_multisplit(lambda seed: entrenar_y_evaluar(interacciones, args.k, seed), args.seeds)

    print(f"NDCG@{args.k} por seed: {resultado['valores']}")
    print(f"media={resultado['media']:.6f} desvio={resultado['desvio']:.6f}")

    n_comics = interacciones["id_comic"].nunique()
    n_usuarios = interacciones["id_usuario"].nunique()
    notas = (
        f"Primer baseline sobre dataset scrapeado ({n_comics} comics, {n_usuarios} usuarios, "
        f"{len(interacciones)} interacciones). Popularidad bayesiana (C=n.mean(), tal cual libros, "
        f"sin sweep propio todavia)."
    )
    loguear("popularity", args.k, args.seeds, resultado, notas)
    print(f"Logueado en {LOG_PATH}")


if __name__ == "__main__":
    main()
