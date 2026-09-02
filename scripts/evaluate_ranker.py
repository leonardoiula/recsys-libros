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
real de Kaggle -13.5% (ver experiments/bitacora.md, "Regresión en
Kaggle") -- este script existe para no repetir ese error con el ranker.

Además del NDCG@20 sin ponderar, reporta el NDCG@20 ponderado por la
actividad real de los usuarios de `ejemplo.csv` (`evaluation.pesos_por_actividad`/
`evaluar_ndcg_ponderado_por_actividad`) -- es un DIAGNÓSTICO reportado, no
cambia ningún criterio de decisión: ya se investigó a fondo
(`experiments/bitacora.md`, "Investigando el sesgo sistemático") que
reponderar por actividad no cambia el signo de ninguna comparación hecha
en este proyecto. Se agrega para no tener que rehacer ese análisis a mano
en cada ronda futura.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_lectores, load_libros
from recsys.evaluation import (
    evaluar_multisplit,
    evaluar_ndcg_ponderado_por_actividad,
    pesos_por_actividad,
)
from recsys.models.ranker import (
    FEATURES,
    evaluar_con_params,
    preparar_pipeline_cacheado,
)

K = 20
N_POR_FUENTE = 150
SEEDS = [42, 7, 123]
EJEMPLO_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "ejemplo.csv"

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
        ctx = preparar_pipeline_cacheado(interacciones, libros, lectores, seed, n_por_fuente=N_POR_FUENTE, k=K)
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
