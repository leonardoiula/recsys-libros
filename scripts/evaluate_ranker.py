"""Evaluación cross-validada del ranker de dos etapas (ALS + género +
popularidad -> LightGBM) contra ALS solo.

Uso: uv run python scripts/evaluate_ranker.py

Para cada seed en `SEEDS`, arma el contexto con `ranker.preparar_pipeline_cacheado`
(split de **tres niveles** -- evita que el ranker vea, como features, scores
calculados con la misma etiqueta que tiene que predecir -- ver docstring
de `ranker.py`; cachea a disco para no repetir el armado si se vuelve a
correr sobre el mismo seed/config) y lo evalúa con `ranker.evaluar_con_params`,
con los hiperparámetros conservadores por default de `fit_ranker`
(`lgbm_params=None`).

Se evalúa sobre varios seeds (no uno solo) a propósito: un sweep de ALS
sobre un único split mejoró el NDCG local +11.5% pero empeoró el score
real de Kaggle -13.5% (ver experiments/legacy/bitacora.md, "Regresión en
Kaggle") -- este script existe para no repetir ese error con el ranker.

Además del NDCG@20 sin ponderar, reporta el NDCG@20 ponderado por la
actividad real de los usuarios de `ejemplo.csv` (`evaluation.pesos_por_actividad`/
`evaluar_ndcg_ponderado_por_actividad`) -- es un DIAGNÓSTICO reportado, no
cambia ningún criterio de decisión: ya se investigó a fondo
(`experiments/legacy/bitacora.md`, "Investigando el sesgo sistemático") que
reponderar por actividad no cambia el signo de ninguna comparación hecha
en este proyecto. Se agrega para no tener que rehacer ese análisis a mano
en cada ronda futura.
"""

from __future__ import annotations

import gc
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_lectores, load_libros
from recsys.evaluation import (
    evaluar_multisplit,
    evaluar_ndcg_ponderado_por_actividad,
    ndcg_at_k,
    pesos_por_actividad,
)
from recsys.models.ranker import (
    FEATURES,
    evaluar_con_params,
    preparar_pipeline_cacheado,
)

K = 20
N_POR_FUENTE = 150
N_POR_FUENTE_AUTOR = 500  # tope propio de la fuente de autor (None = usar N_POR_FUENTE); ver scripts/screen_presupuesto_autor.py
N_CORTES = 1  # ventana rodante del reranker (ver N_CORTES_RANKER en ranker.py); subir a 3/5 para validar la estrategia 1
SEEDS = [42, 7, 123]
EJEMPLO_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "ejemplo.csv"

# Buckets de actividad (interacciones totales del usuario) para el desglose
# por seed -- mismo criterio que scripts/screen_presupuesto_autor.py, para
# poder ver si un bucket concreto (p.ej. 5-9, casuales) regresa de forma
# consistente entre seeds y no solo en uno.
BINS_ACT = [0, 2, 5, 10, 20, 50, 100, math.inf]
LABELS_ACT = ["1", "2-4", "5-9", "10-19", "20-49", "50-99", "100+"]


def ndcg_por_bucket(test_final: pd.DataFrame, recs: dict, n_int_total: pd.Series) -> dict:
    relevantes = test_final.groupby("id_lector")["id_libro"].agg(set).to_dict()
    por_bucket = defaultdict(list)
    for id_lector, rel in relevantes.items():
        act = n_int_total.get(id_lector, 0)
        lab = LABELS_ACT[min(np.searchsorted(BINS_ACT, act, side="right") - 1, len(LABELS_ACT) - 1)]
        por_bucket[lab].append(ndcg_at_k(recs.get(id_lector, []), rel, K))
    return {lab: (len(v), float(np.mean(v))) for lab, v in por_bucket.items()}

interacciones = load_interacciones()
libros = load_libros()
lectores = load_lectores()

n_interacciones_por_usuario_total = interacciones.groupby("id_lector").size()
usuarios_ejemplo = pd.read_csv(EJEMPLO_PATH)["id_lector"].unique().tolist()
pesos_actividad_ejemplo = pesos_por_actividad(n_interacciones_por_usuario_total.reindex(usuarios_ejemplo).fillna(0))


def main() -> None:
    resultados_por_seed = {}
    for seed in SEEDS:
        t0 = time.time()
        ctx = preparar_pipeline_cacheado(
            interacciones, libros, lectores, seed,
            n_por_fuente=N_POR_FUENTE, n_por_fuente_autor=N_POR_FUENTE_AUTOR, k=K, n_cortes=N_CORTES,
        )
        resultados_por_seed[seed] = evaluar_con_params(ctx, None)
        r = resultados_por_seed[seed]
        print(f"seed={seed}: ALS={r['ndcg_als']:.6f}  ranker={r['ndcg_ranker']:.6f}  ({time.time()-t0:.1f}s)")

        ndcg_als_ponderado = evaluar_ndcg_ponderado_por_actividad(
            ctx["test_final"], ctx["recs_als"], K, n_interacciones_por_usuario_total.to_dict(), pesos_actividad_ejemplo
        )
        ndcg_ranker_ponderado = evaluar_ndcg_ponderado_por_actividad(
            ctx["test_final"], r["recs_ranker"], K, n_interacciones_por_usuario_total.to_dict(), pesos_actividad_ejemplo
        )
        print(
            f"  ponderado por actividad de ejemplo.csv -- ALS={ndcg_als_ponderado:.6f}  "
            f"ranker={ndcg_ranker_ponderado:.6f}"
        )

        bucket_seed = ndcg_por_bucket(ctx["test_final"], r["recs_ranker"], n_interacciones_por_usuario_total)
        r["ndcg_por_bucket"] = bucket_seed
        print("  NDCG@20 ranker por bucket de actividad:",
              {lab: round(bucket_seed[lab][1], 4) for lab in LABELS_ACT if lab in bucket_seed})

        modelo_ranker = r.get("modelo_ranker")
        if modelo_ranker is not None:
            importancias = sorted(zip(FEATURES, modelo_ranker.feature_importances_), key=lambda t: -t[1])
            print("  feature_importances_:", ", ".join(f"{f}={v}" for f, v in importancias))

        # Libera el contexto de este seed (candidatos_test/X/y/group, lo más
        # pesado del pipeline) ANTES de construir el del siguiente -- si no,
        # `ctx = preparar_pipeline_cacheado(...)` de la próxima vuelta arma el
        # contexto nuevo completo mientras el de esta vuelta sigue vivo (el
        # lado derecho se evalúa antes de reasignar `ctx`), duplicando el pico
        # de memoria en la transición entre semillas en vez de liberarlo.
        del ctx
        gc.collect()

    resumen_als = evaluar_multisplit(lambda s: resultados_por_seed[s]["ndcg_als"], SEEDS)
    resumen_ranker = evaluar_multisplit(lambda s: resultados_por_seed[s]["ndcg_ranker"], SEEDS)

    print("\n=== Resumen (media +- desvio sobre 3 seeds) ===")
    print(f"ALS solo:            {resumen_als['media']:.6f} +- {resumen_als['desvio']:.6f}  {resumen_als['valores']}")
    print(f"Ranker (dos etapas): {resumen_ranker['media']:.6f} +- {resumen_ranker['desvio']:.6f}  {resumen_ranker['valores']}")

    print("\n=== NDCG@20 ranker por bucket de actividad (una columna por seed + media) ===")
    print(f"{'bucket':>8} " + " ".join(f"{s:>10}" for s in SEEDS) + f" {'media':>10} {'n(42)':>7}")
    for lab in LABELS_ACT:
        vals = [resultados_por_seed[s]["ndcg_por_bucket"].get(lab) for s in SEEDS]
        if all(v is not None for v in vals):
            medias = [v[1] for v in vals]
            print(f"{lab:>8} " + " ".join(f"{m:>10.4f}" for m in medias) + f" {np.mean(medias):>10.4f} {vals[0][0]:>7}")


if __name__ == "__main__":
    main()
