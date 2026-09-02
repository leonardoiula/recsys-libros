"""Compara, sobre los 3 seeds de siempre, refitear la etapa 1 (ALS +
popularidad + género + `calcular_features_auxiliares`) sobre
`train_candidatos_full` antes de generar los candidatos de test, contra
reusar el fit de `train_candidatos` (el comportamiento actual).

Uso: uv run python scripts/comparar_refit_etapa1.py

Por qué existe: `submit.py::_recomendaciones_ranker` hoy fitea la etapa 1
sobre `train_candidatos` (sin la interacción más reciente de cada
usuario) también para generar los candidatos FINALES de la submission
real -- no solo para entrenar el ranker (ahí sí hace falta, para no
filtrar la etiqueta). La práctica estándar es refitear sobre todos los
datos disponibles después de entrenar el modelo supervisado; acá se mide
localmente si eso realmente ayuda antes de tocar producción --
`ranker.preparar_pipeline(..., refit_para_test=True)` reproduce en el
split de evaluación local el mismo patrón que tendría `submit.py` (que no
tiene un `test_final` que reservar).

No es un test pareado por usuario (el ranker en sí se re-entrena en cada
config, con el ruido de punto flotante propio de LightGBM) -- se usa el
mismo criterio que otros cambios estructurales del pipeline en este
proyecto: comparar media±desvío sobre 3 seeds y mirar si el signo es
consistente en los 3.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_lectores, load_libros
from recsys.evaluation import evaluar_multisplit
from recsys.models.ranker import evaluar_con_params, preparar_pipeline_cacheado

K = 20
N_POR_FUENTE = 150
SEEDS = [42, 7, 123]

interacciones = load_interacciones()
libros = load_libros()
lectores = load_lectores()


def _correr(refit_para_test: bool) -> dict:
    resultados_por_seed = {}
    for seed in SEEDS:
        t0 = time.time()
        ctx = preparar_pipeline_cacheado(
            interacciones, libros, lectores, seed, n_por_fuente=N_POR_FUENTE, k=K, refit_para_test=refit_para_test
        )
        resultados_por_seed[seed] = evaluar_con_params(ctx, None)
        r = resultados_por_seed[seed]
        print(
            f"  refit={refit_para_test} seed={seed}: ALS={r['ndcg_als']:.6f}  "
            f"ranker={r['ndcg_ranker']:.6f}  ({time.time()-t0:.1f}s)",
            flush=True,
        )
    return resultados_por_seed


def main() -> None:
    print("=== sin refit (comportamiento actual: etapa 1 fiteada solo con train_candidatos) ===")
    sin_refit = _correr(refit_para_test=False)
    print("\n=== con refit (etapa 1 refiteada sobre train_candidatos_full para los candidatos de test) ===")
    con_refit = _correr(refit_para_test=True)

    resumen_sin = evaluar_multisplit(lambda s: sin_refit[s]["ndcg_ranker"], SEEDS)
    resumen_con = evaluar_multisplit(lambda s: con_refit[s]["ndcg_ranker"], SEEDS)

    print("\n=== Resumen (media +- desvio sobre 3 seeds, NDCG@20 del ranker) ===")
    print(f"sin refit: {resumen_sin['media']:.6f} +- {resumen_sin['desvio']:.6f}  {resumen_sin['valores']}")
    print(f"con refit: {resumen_con['media']:.6f} +- {resumen_con['desvio']:.6f}  {resumen_con['valores']}")

    diffs = [con_refit[s]["ndcg_ranker"] - sin_refit[s]["ndcg_ranker"] for s in SEEDS]
    print(f"\ndiferencia (con refit - sin refit) por seed: {diffs}")
    print(f"positivo en los 3 seeds: {all(d > 0 for d in diffs)}")


if __name__ == "__main__":
    main()
