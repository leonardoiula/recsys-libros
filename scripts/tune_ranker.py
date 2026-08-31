"""Búsqueda de hiperparámetros de LightGBM para el ranker, con validación
cruzada real desde el arranque.

Uso: uv run python scripts/tune_ranker.py

Cada trial de optuna se evalúa con `SEEDS_TUNING` (2 seeds, no 1 -- nunca
menos, después del episodio de ALS+optuna donde un sweep sobre un único
split mejoró el NDCG local pero empeoró el score real de Kaggle, ver
`experiments/bitacora.md`). Al terminar la búsqueda, confirma el mejor
config encontrado con los 3 seeds completos (`SEEDS_FINAL`, mismos de
`scripts/evaluate_ranker.py`).

Contexto cacheado por seed, no `ranker.evaluar_pipeline` por trial: con
el set de 23 features, armar candidatos/dataset (`ranker.preparar_pipeline`)
tarda ~280-300s y entrenar LightGBM con una config (`ranker.evaluar_con_params`)
tarda ~22s -- ver el docstring de `preparar_pipeline` para el detalle de
costos. Como el armado de candidatos no depende de `lgbm_params`, se
arma **una sola vez por seed** (`_contexto`, cacheado en `_CONTEXTOS`) y
se reusa en todos los trials -- sin este cacheo, 10 trials x 2 seeds +
confirmación final con 3 seeds tardarían ~2.5 horas; con él, ~20-25 min.

Referencia (`scripts/evaluate_ranker.py`, ranker de 23 features
confirmado en Kaggle -- ver `experiments/log.csv`, fila con
`ndcg_kaggle=0.04831`, 3 seeds): ranker = 0.109735 +- 0.003719.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_lectores, load_libros
from recsys.evaluation import evaluar_multisplit
from recsys.models.ranker import evaluar_con_params, preparar_pipeline

K = 20
N_POR_FUENTE = 150
SEEDS_TUNING = [42, 7]
SEEDS_FINAL = [42, 7, 123]
N_TRIALS = 10

BASELINE_CONSERVADOR = 0.109735  # ver docstring, misma corrida de evaluate_ranker.py

interacciones = load_interacciones()
libros = load_libros()
lectores = load_lectores()

optuna.logging.set_verbosity(optuna.logging.WARNING)

_CONTEXTOS: dict[int, dict] = {}


def _contexto(seed: int) -> dict:
    """Arma el contexto de `preparar_pipeline` la primera vez que se pide
    ese seed, y lo cachea -- es la parte cara (~280-300s) del pipeline,
    la misma para todos los trials de un mismo seed."""
    if seed not in _CONTEXTOS:
        _CONTEXTOS[seed] = preparar_pipeline(interacciones, libros, lectores, seed, n_por_fuente=N_POR_FUENTE, k=K)
    return _CONTEXTOS[seed]


def ndcg_ranker(seed: int, lgbm_params: dict | None) -> float:
    return evaluar_con_params(_contexto(seed), lgbm_params)["ndcg_ranker"]


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
