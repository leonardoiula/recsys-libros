"""Búsqueda sistemática de hiperparámetros para ALS y BPR con optuna.

Uso: uv run python scripts/tune_als.py

Corre dos estudios de optuna (maximizando NDCG@20 personalizado sobre el
split temporal corregido, `n_val=1`/`seed=42`, ver `experiments/bitacora.md`):

- ALS: busca `factors`, `regularization` y `alpha` (la fórmula de
  confianza `1 + alpha*rating`, ver `src/recsys/models/als.py`).
  `iterations=20` queda fijo -- ya se confirmó rendimiento decreciente
  en un sweep manual anterior.
- BPR: busca `factors`, `regularization`, `learning_rate` e `iterations`
  -- es optimización por SGD, converge distinto a ALS, no se puede fijar
  `iterations` de antemano con la misma confianza.

Al final imprime una tabla comparativa: ALS con los hiperparámetros
"actuales" de producción (factors=128, regularization=0.1, alpha=1.0,
sin tunear) vs el mejor ALS encontrado vs el mejor BPR encontrado, más
Recall@200 de cada uno como diagnóstico de cobertura (¿el modelo ubica
al libro correcto en un candidato amplio, aunque no en el top-20?).

Este script NO decide solo qué va a `submit.py` -- eso se define después
de ver los resultados (ver experiments/bitacora.md).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, split_train_val, libros_leidos_por_usuario
from recsys.evaluation import evaluar_ndcg_personalizado, evaluar_recall_personalizado
from recsys.models.als import fit_als, recomendar_por_usuario
from recsys.models.bpr import fit_bpr
from recsys.models.popularity import fit_popularity

K = 20
K_DIAGNOSTICO = 200
N_TRIALS = 30  # subir si hay tiempo/presupuesto de computo disponible

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main() -> None:
    interacciones = load_interacciones()
    train, val = split_train_val(interacciones, n_val=1, seed=42)
    libros_leidos_train = libros_leidos_por_usuario(train)
    usuarios_val = val["id_lector"].unique().tolist()
    ranking_global = fit_popularity(train)["id_libro"].tolist()

    print(f"train={len(train)} val={len(val)} usuarios_val={len(usuarios_val)}")

    def ndcg_als(factors: int, regularization: float, alpha: float) -> float:
        modelo, matriz, fila_por_usuario, libros_por_columna = fit_als(
            train, factors=factors, regularization=regularization, iterations=20, alpha=alpha, seed=42
        )
        recs = recomendar_por_usuario(
            usuarios=usuarios_val, modelo=modelo, matriz_usuario_libro=matriz,
            fila_por_usuario=fila_por_usuario, libros_por_columna=libros_por_columna,
            ranking_global=ranking_global, libros_leidos=libros_leidos_train, k=K,
        )
        return evaluar_ndcg_personalizado(val, recs, K)

    def objective_als(trial: optuna.Trial) -> float:
        factors = trial.suggest_int("factors", 32, 256, log=True)
        regularization = trial.suggest_float("regularization", 0.001, 1.0, log=True)
        alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
        ndcg = ndcg_als(factors, regularization, alpha)
        print(f"  [ALS trial {trial.number:>3}] factors={factors:>3} reg={regularization:.4f} alpha={alpha:.3f} -> NDCG@20={ndcg:.6f}")
        return ndcg

    def objective_bpr(trial: optuna.Trial) -> float:
        factors = trial.suggest_int("factors", 32, 256, log=True)
        regularization = trial.suggest_float("regularization", 1e-5, 0.1, log=True)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
        iterations = trial.suggest_int("iterations", 50, 300)
        modelo, matriz, fila_por_usuario, libros_por_columna = fit_bpr(
            train, factors=factors, regularization=regularization,
            learning_rate=learning_rate, iterations=iterations, seed=42,
        )
        recs = recomendar_por_usuario(
            usuarios=usuarios_val, modelo=modelo, matriz_usuario_libro=matriz,
            fila_por_usuario=fila_por_usuario, libros_por_columna=libros_por_columna,
            ranking_global=ranking_global, libros_leidos=libros_leidos_train, k=K,
        )
        ndcg = evaluar_ndcg_personalizado(val, recs, K)
        print(f"  [BPR trial {trial.number:>3}] factors={factors:>3} reg={regularization:.5f} lr={learning_rate:.4f} iter={iterations:>3} -> NDCG@20={ndcg:.6f}")
        return ndcg

    print(f"\n=== Baseline: ALS actual (factors=128, regularization=0.1, alpha=1.0) ===")
    t0 = time.time()
    ndcg_baseline = ndcg_als(128, 0.1, 1.0)
    print(f"NDCG@20 = {ndcg_baseline:.6f} ({time.time()-t0:.1f}s)")

    print(f"\n=== Optuna: ALS ({N_TRIALS} trials) ===")
    study_als = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_als.optimize(objective_als, n_trials=N_TRIALS)

    print(f"\n=== Optuna: BPR ({N_TRIALS} trials) ===")
    study_bpr = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_bpr.optimize(objective_bpr, n_trials=N_TRIALS)

    # Recall@200 de diagnostico para los tres candidatos finales
    def recall200_als(factors: int, regularization: float, alpha: float) -> float:
        modelo, matriz, fila_por_usuario, libros_por_columna = fit_als(
            train, factors=factors, regularization=regularization, iterations=20, alpha=alpha, seed=42
        )
        recs = recomendar_por_usuario(
            usuarios=usuarios_val, modelo=modelo, matriz_usuario_libro=matriz,
            fila_por_usuario=fila_por_usuario, libros_por_columna=libros_por_columna,
            ranking_global=ranking_global, libros_leidos=libros_leidos_train, k=K_DIAGNOSTICO,
        )
        return evaluar_recall_personalizado(val, recs, K_DIAGNOSTICO)

    def recall200_bpr(factors: int, regularization: float, learning_rate: float, iterations: int) -> float:
        modelo, matriz, fila_por_usuario, libros_por_columna = fit_bpr(
            train, factors=factors, regularization=regularization,
            learning_rate=learning_rate, iterations=iterations, seed=42,
        )
        recs = recomendar_por_usuario(
            usuarios=usuarios_val, modelo=modelo, matriz_usuario_libro=matriz,
            fila_por_usuario=fila_por_usuario, libros_por_columna=libros_por_columna,
            ranking_global=ranking_global, libros_leidos=libros_leidos_train, k=K_DIAGNOSTICO,
        )
        return evaluar_recall_personalizado(val, recs, K_DIAGNOSTICO)

    recall_baseline = recall200_als(128, 0.1, 1.0)
    recall_als = recall200_als(**study_als.best_params)
    recall_bpr = recall200_bpr(**study_bpr.best_params)

    print("\n=== Resumen ===")
    print(f"{'candidato':<30} {'NDCG@20':>10} {'Recall@200':>12}  params")
    print(f"{'ALS actual (sin tunear)':<30} {ndcg_baseline:>10.6f} {recall_baseline:>12.4f}  factors=128 regularization=0.1 alpha=1.0")
    print(f"{'ALS tuneado (optuna)':<30} {study_als.best_value:>10.6f} {recall_als:>12.4f}  {study_als.best_params}")
    print(f"{'BPR tuneado (optuna)':<30} {study_bpr.best_value:>10.6f} {recall_bpr:>12.4f}  {study_bpr.best_params}")


if __name__ == "__main__":
    main()
