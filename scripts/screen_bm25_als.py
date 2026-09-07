"""Pre-screen barato: ¿conviene pesar la matriz de ALS con BM25 antes del
`modelo.fit()` para bajar el peso de los power users?

Uso: uv run python scripts/screen_bm25_als.py

Contexto: el 11,5% de los usuarios (100+ interacciones) concentra el 64%
de la señal; en un ALS con `confianza = rating` crudo esos usuarios
dominan la factorización. `fit_als(..., bm25=(K1, B))` aplica
`implicit.nearest_neighbours.bm25_weight` solo a la copia que va a
`modelo.fit()` (ver docstring de `src/recsys/models/als.py`).

Este script mide ALS **solo** (sin el ranker de dos etapas), sobre el
split temporal corregido (`n_val=1`, `seed=42`), para una grilla chica de
`(K1, B)` más la fila baseline (`bm25=None`). Reporta NDCG@20 (métrica de
entrega) y Recall@200 (diagnóstico de cobertura, mismo que usa
`scripts/tune_als.py`). NO decide solo qué se adopta -- si alguna config
mejora el NDCG sin hundir el Recall, el siguiente paso es cablearla en
`ranker.py` y correr `scripts/evaluate_ranker.py` (CV 3 seeds).

Ojo: `bm25_weight` SIEMPRE aplica el término IDF (incluso con B=0), así
que la referencia real de "sin cambio" es la fila `bm25=None`, no `B=0`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import libros_leidos_por_usuario, load_interacciones, split_train_val
from recsys.evaluation import evaluar_ndcg_personalizado, evaluar_recall_personalizado
from recsys.models.als import fit_als, recomendar_por_usuario
from recsys.models.popularity import fit_popularity

K = 20
K_DIAGNOSTICO = 200
SEED = 42

K1_GRID = [1.0, 10.0, 100.0]
B_GRID = [0.25, 0.5, 0.75, 1.0]


def main() -> None:
    interacciones = load_interacciones()
    train, val = split_train_val(interacciones, n_val=1, seed=SEED)
    libros_leidos_train = libros_leidos_por_usuario(train)
    usuarios_val = val["id_lector"].unique().tolist()
    ranking_global = fit_popularity(train)["id_libro"].tolist()

    print(f"train={len(train)} val={len(val)} usuarios_val={len(usuarios_val)}\n")

    def evaluar(bm25: tuple[float, float] | None) -> tuple[float, float, float]:
        t0 = time.time()
        modelo, matriz, fila_por_usuario, libros_por_columna = fit_als(
            train, factors=128, regularization=0.1, iterations=20, alpha=None, seed=SEED, bm25=bm25
        )
        recs_k = recomendar_por_usuario(
            usuarios=usuarios_val, modelo=modelo, matriz_usuario_libro=matriz,
            fila_por_usuario=fila_por_usuario, libros_por_columna=libros_por_columna,
            ranking_global=ranking_global, libros_leidos=libros_leidos_train, k=K,
        )
        recs_diag = recomendar_por_usuario(
            usuarios=usuarios_val, modelo=modelo, matriz_usuario_libro=matriz,
            fila_por_usuario=fila_por_usuario, libros_por_columna=libros_por_columna,
            ranking_global=ranking_global, libros_leidos=libros_leidos_train, k=K_DIAGNOSTICO,
        )
        ndcg = evaluar_ndcg_personalizado(val, recs_k, K)
        recall = evaluar_recall_personalizado(val, recs_diag, K_DIAGNOSTICO)
        return ndcg, recall, time.time() - t0

    print("=== Baseline: sin BM25 (bm25=None) ===")
    ndcg_base, recall_base, dt = evaluar(None)
    print(f"NDCG@{K}={ndcg_base:.6f}  Recall@{K_DIAGNOSTICO}={recall_base:.4f}  ({dt:.0f}s)\n")

    filas = []
    for k1 in K1_GRID:
        for b in B_GRID:
            ndcg, recall, dt = evaluar((k1, b))
            filas.append((k1, b, ndcg, recall, dt))
            print(
                f"K1={k1:>6.1f} B={b:.2f}  NDCG@{K}={ndcg:.6f} ({ndcg - ndcg_base:+.6f})  "
                f"Recall@{K_DIAGNOSTICO}={recall:.4f} ({recall - recall_base:+.4f})  ({dt:.0f}s)"
            )

    print(f"\n=== Resumen (ordenado por NDCG@{K}) ===")
    print(f"{'config':<22} {'NDCG@20':>10} {'d_NDCG':>10} {'Recall@200':>12} {'d_Recall':>10}")
    print(f"{'baseline (bm25=None)':<22} {ndcg_base:>10.6f} {0.0:>+10.6f} {recall_base:>12.4f} {0.0:>+10.4f}")
    for k1, b, ndcg, recall, _ in sorted(filas, key=lambda r: -r[2]):
        print(
            f"{'K1=' + str(k1) + ' B=' + str(b):<22} {ndcg:>10.6f} {ndcg - ndcg_base:>+10.6f} "
            f"{recall:>12.4f} {recall - recall_base:>+10.4f}"
        )


if __name__ == "__main__":
    main()
