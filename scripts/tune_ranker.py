"""Búsqueda de hiperparámetros de LightGBM para el ranker, con validación
cruzada real desde el arranque.

Uso: uv run python scripts/tune_ranker.py

Cada trial de optuna se evalúa con `SEEDS_TUNING` (2 seeds, no 1 -- nunca
menos, después del episodio de ALS+optuna donde un sweep sobre un único
split mejoró el NDCG local pero empeoró el score real de Kaggle, ver
`experiments/bitacora.md`). Al terminar la búsqueda, confirma el mejor
config encontrado con los 3 seeds completos (`SEEDS_FINAL`, mismos de
`scripts/evaluate_ranker.py`) para comparar contra los hiperparámetros
conservadores actuales -- reusa `ranker.evaluar_pipeline`, no duplica el
split de tres niveles ni el armado de candidatos/dataset.

Referencia (misma corrida de `evaluate_ranker.py`, ya con las features
de autor/año/género/recencia, hiperparámetros conservadores,
`n_por_fuente=150`, 3 seeds): ranker = 0.106815 +- 0.001498.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_libros
from recsys.evaluation import evaluar_multisplit
from recsys.models.ranker import evaluar_pipeline

K = 20
N_POR_FUENTE = 150
SEEDS_TUNING = [42, 7]
SEEDS_FINAL = [42, 7, 123]
N_TRIALS = 10

BASELINE_CONSERVADOR = 0.106815  # ver docstring, misma corrida de evaluate_ranker.py

interacciones = load_interacciones()
libros = load_libros()

optuna.logging.set_verbosity(optuna.logging.WARNING)


def ndcg_ranker(seed: int, lgbm_params: dict | None) -> float:
    return evaluar_pipeline(interacciones, libros, seed, n_por_fuente=N_POR_FUENTE, lgbm_params=lgbm_params, k=K)[
        "ndcg_ranker"
    ]


def objective(trial: optuna.Trial) -> float:
    params = dict(
        num_leaves=trial.suggest_int("num_leaves", 7, 63, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 400),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )
    valores = [ndcg_ranker(seed, params) for seed in SEEDS_TUNING]
    media = sum(valores) / len(valores)
    print(f"  [trial {trial.number:>3}] media={media:.6f} valores={valores} params={params}")
    return media


def main() -> None:
    print(f"Baseline (hiperparametros conservadores, referencia): {BASELINE_CONSERVADOR:.6f}")

    print(f"\n=== Optuna: LightGBM ({N_TRIALS} trials, {len(SEEDS_TUNING)} seeds c/u) ===")
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"Busqueda terminada en {time.time()-t0:.1f}s. Mejor media (2 seeds): {study.best_value:.6f}")
    print(f"Mejor config: {study.best_params}")

    print(f"\n=== Confirmando el mejor config con los {len(SEEDS_FINAL)} seeds completos ===")
    confirmado = evaluar_multisplit(lambda s: ndcg_ranker(s, study.best_params), SEEDS_FINAL)
    print(f"Tuneado: {confirmado['media']:.6f} +- {confirmado['desvio']:.6f}  {confirmado['valores']}")

    print("\n=== Resumen ===")
    print(f"Conservador (referencia): {BASELINE_CONSERVADOR:.6f}")
    print(f"Tuneado (optuna, {len(SEEDS_FINAL)} seeds): {confirmado['media']:.6f} +- {confirmado['desvio']:.6f}")


if __name__ == "__main__":
    main()
