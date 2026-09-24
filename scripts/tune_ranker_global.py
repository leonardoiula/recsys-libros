"""Búsqueda de hiperparámetros de LightGBM para el ranker bajo el protocolo de corte
temporal GLOBAL (`scripts/evaluate_global_cutoff.py`), con validación cruzada real
desde el arranque -- mismo patrón que `scripts/tune_ranker.py`, pero sobre
`preparar_contexto_global`/`evaluar_con_params_global` en vez de
`ranker.preparar_pipeline`/`ranker.evaluar_con_params`.

Uso: uv run python scripts/tune_ranker_global.py

Por qué hace falta un tuning propio acá (no alcanza con reusar la conclusión de
`tune_ranker.py`, que 3 veces encontró que tunear no ayuda): el corte global entrena
con ~4.471 grupos (vs ~7.900-40.000 del split por usuario según `n_cortes`) -- los
hiperparámetros conservadores de producción (`num_leaves=31, learning_rate=0.05,
n_estimators=200`) se eligieron y confirmaron en el régimen de N grande; con un
dataset de entrenamiento bastante más chico el balance sesgo/varianza óptimo podría
ser distinto (más regularización, menos hojas). Nunca se probó.

Cada trial se evalúa con `SEEDS_TUNING` (2 seeds, no 1 -- mismo criterio que
`tune_ranker.py`). A diferencia del split por usuario, acá `seed` no cambia qué datos
caen en cada partición (`CORTES_VENTANA` son fechas fijas) -- varía el
`random_state` de ALS en cada ventana de la ventana rodante, así que sigue siendo una
fuente de variación real (entrenamiento), no la misma que mide el CV del split por
usuario (composición de datos). Al terminar la búsqueda, confirma el mejor config
encontrado Y el baseline conservador con los 3 seeds completos (`SEEDS_FINAL`), para
poder decidir con el mismo criterio que el resto del proyecto (positivo en los 3,
memoria del desvío entre seeds) en vez de solo la media de 2.

Costo: `preparar_contexto_global` (~7-8 min/seed) se arma UNA vez por seed y se
cachea a disco (`data/cache/global_ctx_*.pkl`) -- se reusa en todos los trials Y en
la confirmación final. Con 3 seeds, armar los contextos son ~21-24 min (una sola vez);
cada trial de `evaluar_con_params_global` es ~30-60s. `N_TRIALS=10` x 2 seeds + la
confirmación final (2 configs x 3 seeds, con seeds 42/7 ya cacheados de la búsqueda)
tardan en total ~45-55 min en frío.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_global_cutoff import evaluar_con_params_global, preparar_contexto_global

SEEDS_TUNING = [42, 7]
SEEDS_FINAL = [42, 7, 123]
N_TRIALS = 10

# Referencia: hiperparámetros conservadores de producción (fit_ranker defaults en
# ranker.py) medidos bajo este mismo protocolo, seed=42, 5 cortes -- ver
# experiments/log.csv, fila validacion_corte_global (2026-09-24).
BASELINE_SEED42 = 0.049061

optuna.logging.set_verbosity(optuna.logging.WARNING)

_CONTEXTOS: dict[int, dict] = {}


def _contexto(seed: int) -> dict:
    """Arma (o recupera del cache a disco) el contexto de `preparar_contexto_global`
    para `seed` -- la parte cara, la misma para todos los trials de un mismo seed."""
    if seed not in _CONTEXTOS:
        t0 = time.time()
        _CONTEXTOS[seed] = preparar_contexto_global(seed=seed, verbose=False)
        print(f"  contexto seed={seed} listo en {time.time()-t0:.0f}s", flush=True)
    return _CONTEXTOS[seed]


def ndcg_ranker(seed: int, lgbm_params: dict | None) -> float:
    return evaluar_con_params_global(_contexto(seed), lgbm_params=lgbm_params)["ndcg_ranker"]


def evaluar_multisplit(lgbm_params: dict | None, seeds: list[int]) -> dict:
    valores = [ndcg_ranker(s, lgbm_params) for s in seeds]
    media = sum(valores) / len(valores)
    desvio = statistics.stdev(valores) if len(valores) > 1 else 0.0
    return {"valores": valores, "media": media, "desvio": desvio}


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
    print(f"  [trial {trial.number:>3}] media={media:.6f} valores={valores} params={params}", flush=True)
    return media


def main() -> None:
    print(f"Baseline conservador de referencia (seed=42, ya logueado): {BASELINE_SEED42:.6f}")

    print(f"\n=== Optuna: LightGBM bajo corte global ({N_TRIALS} trials, {len(SEEDS_TUNING)} seeds c/u) ===")
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"Busqueda terminada en {time.time()-t0:.1f}s. Mejor media (2 seeds): {study.best_value:.6f}")
    print(f"Mejor config: {study.best_params}")

    print(f"\n=== Confirmando con los {len(SEEDS_FINAL)} seeds completos: tuneado vs conservador ===")
    tuneado = evaluar_multisplit(study.best_params, SEEDS_FINAL)
    conservador = evaluar_multisplit(None, SEEDS_FINAL)  # None = defaults de fit_ranker (conservadores)

    print(f"\nConservador (defaults de produccion): {conservador['media']:.6f} +- {conservador['desvio']:.6f}  {conservador['valores']}")
    print(f"Tuneado (optuna, {len(SEEDS_FINAL)} seeds):          {tuneado['media']:.6f} +- {tuneado['desvio']:.6f}  {tuneado['valores']}")
    diferencia = tuneado["media"] - conservador["media"]
    positivo_en_todos = all(t > c for t, c in zip(tuneado["valores"], conservador["valores"]))
    print(f"\nDiferencia (tuneado - conservador): {diferencia:+.6f}")
    print(f"Positivo en los {len(SEEDS_FINAL)} seeds individualmente: {positivo_en_todos}")
    print(f"Mejor config encontrado: {study.best_params}")


if __name__ == "__main__":
    main()
