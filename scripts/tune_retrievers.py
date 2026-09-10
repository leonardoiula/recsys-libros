"""Optimización de retrievers colaborativos alternativos como fuentes de
candidatos (estrategia 2 de `experiments/estrategias.md`: retrieval
aprendido para levantar el techo de recall ~0.535).

Uso: uv run python scripts/tune_retrievers.py

Sin `torch`/`gensim` en el entorno -> un two-tower NN queda afuera. Se
tunean con optuna, maximizando **recall@200** en el val (n_val=1,
seed=42), las alternativas colaborativas de `implicit`:

- BPR  (ranking pairwise, ya envuelto en `models/bpr.py`)
- LMF  (logistic matrix factorization)
- BM25 item-KNN   (`implicit.nearest_neighbours.BM25Recommender`)
- Cosine item-KNN (`implicit.nearest_neighbours.CosineRecommender`)

Baseline de referencia: ALS con la config de producción
(`factors=128, reg=0.1`, rating crudo, `bm25=(10, 0.75)`).

**Número decisivo**: para el mejor de cada familia, el *recall
complementario* -- recall de `ALS-top-200 ∪ retriever-top-200` vs
`ALS-top-200` sola. Si el retriever no agrega cobertura sobre ALS, no
sirve como fuente nueva.

Escribe el progreso a `data/cache/tune_retrievers.json`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import BM25Recommender, CosineRecommender

from recsys.data import libros_leidos_por_usuario, load_interacciones, split_train_val
from recsys.models.als import construir_matriz_usuario_libro, fit_als

K_DIAG = 200
N_TRIALS = 25
SEED = 42
OUT = Path(__file__).resolve().parents[1] / "data" / "cache" / "tune_retrievers.json"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _recomendar(modelo, matriz_csr, filas_val, libros_por_columna, k):
    """top-k ids de libro por usuario (array de listas), filtrando ya-leídos."""
    ids, _ = modelo.recommend(filas_val, matriz_csr[filas_val], N=k, filter_already_liked_items=True)
    cols = np.asarray(libros_por_columna, dtype=object)
    return [list(cols[fila]) for fila in ids]


def _recall(recs_por_fila, objetivos_por_fila) -> float:
    hits = sum(1 for recs, obj in zip(recs_por_fila, objetivos_por_fila) if obj in recs)
    return hits / len(recs_por_fila)


def main() -> None:
    interacciones = load_interacciones()
    train, val = split_train_val(interacciones, n_val=1, seed=SEED)

    # matriz binaria compartida (para BPR/LMF/KNN) + índices
    matriz_bin, fila_por_usuario, libros_por_columna = construir_matriz_usuario_libro(train, alpha=None)
    matriz_bin = (matriz_bin > 0).astype(np.float32).tocsr()

    val_obj = dict(zip(val["id_lector"], val["id_libro"]))
    usuarios_val = [u for u in val["id_lector"] if u in fila_por_usuario]
    filas_val = np.array([fila_por_usuario[u] for u in usuarios_val])
    objetivos = [val_obj[u] for u in usuarios_val]
    print(f"train={len(train)} val evaluables={len(usuarios_val)} libros={len(libros_por_columna)}", flush=True)

    resultados: dict = {}
    if OUT.exists():
        resultados = json.loads(OUT.read_text())

    # --- baseline ALS (config de producción) ---
    t0 = time.time()
    als_modelo, als_matriz, _, als_cols = fit_als(train, bm25=(10.0, 0.75))
    als_recs = _recomendar(als_modelo, als_matriz.tocsr(), filas_val, als_cols, K_DIAG)
    als_recall = _recall(als_recs, objetivos)
    als_sets = [set(r) for r in als_recs]
    resultados["ALS_prod"] = {"recall@200": als_recall, "params": "factors=128,reg=0.1,raw,bm25=(10,0.75)"}
    OUT.write_text(json.dumps(resultados, indent=2))
    print(f"ALS_prod recall@{K_DIAG} = {als_recall:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    def evaluar_retriever(modelo) -> tuple[float, float]:
        modelo.fit(matriz_bin)
        recs = _recomendar(modelo, matriz_bin, filas_val, libros_por_columna, K_DIAG)
        solo = _recall(recs, objetivos)
        comp = _recall([list(s | set(r)) for s, r in zip(als_sets, recs)], objetivos)
        return solo, comp

    def correr_familia(nombre: str, objective, N=N_TRIALS):
        if nombre in resultados:
            print(f"[{nombre}] ya en {OUT.name}, salteo", flush=True)
            return
        print(f"\n=== {nombre} ({N} trials) ===", flush=True)
        t0 = time.time()
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=N)
        best = study.best_params
        # recall complementario del mejor
        solo, comp = evaluar_retriever(_construir(nombre, best))
        resultados[nombre] = {
            "recall@200_solo": solo, "recall@200_union_con_ALS": comp,
            "delta_vs_ALS_solo": comp - als_recall, "params": best,
        }
        OUT.write_text(json.dumps(resultados, indent=2))
        print(f"[{nombre}] mejor solo={solo:.4f}  union_con_ALS={comp:.4f}  "
              f"(ALS sola {als_recall:.4f}, delta {comp-als_recall:+.4f})  {best}  ({time.time()-t0:.0f}s)", flush=True)

    def _construir(nombre, p):
        if nombre == "BPR":
            return BayesianPersonalizedRanking(factors=p["factors"], learning_rate=p["learning_rate"],
                                               regularization=p["regularization"], iterations=p["iterations"],
                                               random_state=SEED)
        if nombre == "LMF":
            return LogisticMatrixFactorization(factors=p["factors"], learning_rate=p["learning_rate"],
                                               regularization=p["regularization"], iterations=p["iterations"],
                                               random_state=SEED)
        if nombre == "BM25_KNN":
            return BM25Recommender(K=p["K"], K1=p["K1"], B=p["B"])
        if nombre == "Cosine_KNN":
            return CosineRecommender(K=p["K"])
        raise ValueError(nombre)

    def obj_bpr(t):
        p = dict(factors=t.suggest_int("factors", 32, 256, log=True),
                 learning_rate=t.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
                 regularization=t.suggest_float("regularization", 1e-5, 1e-1, log=True),
                 iterations=t.suggest_int("iterations", 50, 300))
        solo, _ = evaluar_retriever(_construir("BPR", p))
        return solo

    def obj_lmf(t):
        p = dict(factors=t.suggest_int("factors", 32, 256, log=True),
                 learning_rate=t.suggest_float("learning_rate", 1e-3, 1.0, log=True),
                 regularization=t.suggest_float("regularization", 1e-4, 10.0, log=True),
                 iterations=t.suggest_int("iterations", 20, 120))
        solo, _ = evaluar_retriever(_construir("LMF", p))
        return solo

    def obj_bm25(t):
        p = dict(K=t.suggest_int("K", 20, 400), K1=t.suggest_float("K1", 1.0, 1000.0, log=True),
                 B=t.suggest_float("B", 0.0, 1.0))
        solo, _ = evaluar_retriever(_construir("BM25_KNN", p))
        return solo

    def obj_cos(t):
        solo, _ = evaluar_retriever(_construir("Cosine_KNN", {"K": t.suggest_int("K", 20, 400)}))
        return solo

    correr_familia("BM25_KNN", obj_bm25)
    correr_familia("Cosine_KNN", obj_cos, N=12)
    correr_familia("BPR", obj_bpr)
    correr_familia("LMF", obj_lmf)

    print("\n=== RESUMEN ===")
    print(f"ALS_prod recall@200 sola: {als_recall:.4f}")
    for k, v in resultados.items():
        if k == "ALS_prod":
            continue
        print(f"  {k:<12} solo={v['recall@200_solo']:.4f}  union_con_ALS={v['recall@200_union_con_ALS']:.4f}  "
              f"delta={v['delta_vs_ALS_solo']:+.4f}  {v['params']}")


if __name__ == "__main__":
    main()
