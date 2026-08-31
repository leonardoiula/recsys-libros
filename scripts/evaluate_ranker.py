"""Evaluación cross-validada del ranker de dos etapas (ALS + género +
popularidad -> LightGBM) contra ALS solo.

Uso: uv run python scripts/evaluate_ranker.py

Para cada seed en `SEEDS`, corre `ranker.evaluar_pipeline` (split de
**tres niveles** -- evita que el ranker vea, como features, scores
calculados con la misma etiqueta que tiene que predecir -- ver docstring
de `ranker.py`), con los hiperparámetros conservadores por default de
`fit_ranker` (`lgbm_params=None`).

Se evalúa sobre varios seeds (no uno solo) a propósito: un sweep de ALS
sobre un único split mejoró el NDCG local +11.5% pero empeoró el score
real de Kaggle -13.5% (ver experiments/bitacora.md, "Regresión en
Kaggle") -- este script existe para no repetir ese error con el ranker.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_lectores, load_libros
from recsys.evaluation import evaluar_multisplit
from recsys.models.ranker import FEATURES, evaluar_pipeline

K = 20
N_POR_FUENTE = 150
SEEDS = [42, 7, 123]

interacciones = load_interacciones()
libros = load_libros()
lectores = load_lectores()


def main() -> None:
    resultados_por_seed = {}
    for seed in SEEDS:
        t0 = time.time()
        resultados_por_seed[seed] = evaluar_pipeline(
            interacciones, libros, lectores, seed, n_por_fuente=N_POR_FUENTE, k=K
        )
        r = resultados_por_seed[seed]
        print(f"seed={seed}: ALS={r['ndcg_als']:.6f}  ranker={r['ndcg_ranker']:.6f}  ({time.time()-t0:.1f}s)")
        modelo_ranker = r.get("modelo_ranker")
        if modelo_ranker is not None:
            importancias = sorted(zip(FEATURES, modelo_ranker.feature_importances_), key=lambda t: -t[1])
            print("  feature_importances_:", ", ".join(f"{f}={v}" for f, v in importancias))

    resumen_als = evaluar_multisplit(lambda s: resultados_por_seed[s]["ndcg_als"], SEEDS)
    resumen_ranker = evaluar_multisplit(lambda s: resultados_por_seed[s]["ndcg_ranker"], SEEDS)

    print("\n=== Resumen (media +- desvio sobre 3 seeds) ===")
    print(f"ALS solo:            {resumen_als['media']:.6f} +- {resumen_als['desvio']:.6f}  {resumen_als['valores']}")
    print(f"Ranker (dos etapas): {resumen_ranker['media']:.6f} +- {resumen_ranker['desvio']:.6f}  {resumen_ranker['valores']}")


if __name__ == "__main__":
    main()
