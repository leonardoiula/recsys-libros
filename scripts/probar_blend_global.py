"""Blend de dos FAMILIAS de modelos distintas -- ALS (factorización latente pura) +
LGBMRanker (árboles sobre 40 features, que ya incluyen `score_als`/`rank_als` como
insumo) -- vía Reciprocal Rank Fusion (RRF), evaluado bajo el protocolo de corte
temporal global (`scripts/evaluate_global_cutoff.py`).

Por qué esto y no otra cosa: es la única estrategia de mayor calibre que
`experiments/estrategias.md` deja explícitamente "sin explorar todavía" -- el
proyecto ya cerró seed-bagging (5 copias del mismo modelo, plano en Kaggle) y
señaló que "el valor real del ensamble estaría en blend de familias distintas, no
en copias del mismo modelo". RRF es la versión más barata y defendible de esa idea:
no entrena nada nuevo (reusa el ALS y el LGBMRanker que `construir_y_evaluar` ya
entrena), solo combina los dos rankings — `score(item) = Σ 1/(K_RRF + rank)` sobre
cada lista donde aparece el item, técnica estándar de fusión de rankings en IR
(Cormack et al. 2009), robusta a que los scores de ALS y LightGBM no estén en la
misma escala (por eso se fusiona por RANK, no por score crudo).

Uso: uv run python scripts/probar_blend_global.py (~7-8 min, una sola corrida de
`construir_y_evaluar` -- no hace falta una segunda config completa, el blend se arma
en memoria sobre los pools que ya devuelve).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_global_cutoff import K, construir_y_evaluar
from recsys.evaluation import ndcg_at_k

N_BOOTSTRAP = 2000
K_RRF = 60  # constante estándar de Cormack et al. 2009 -- suaviza la contribución de items en posiciones altas


def _fusionar_rrf(pool_ranker: list, pool_als: list, k_rrf: int = K_RRF) -> list:
    """RRF: score(item) = suma de 1/(k_rrf + rank) sobre cada pool donde aparece
    (rank 0-indexed). Devuelve los items ordenados por score descendente."""
    scores: dict = {}
    for pool in (pool_ranker, pool_als):
        for rank, item in enumerate(pool):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(scores, key=lambda item: -scores[item])


def main() -> None:
    r = construir_y_evaluar()

    recs_ranker_pool = r["recs_ranker_pool"]
    recs_als_pool = r["recs_als_pool"]
    relevantes_por_usuario = r["relevantes_por_usuario"]
    libros_leidos = r["libros_leidos_hasta_t2"]
    ranking_global = r["ranking_global"]
    usuarios = list(recs_ranker_pool.keys())

    ndcg_blend_por_usuario: dict = {}
    for id_lector in usuarios:
        fusionado = _fusionar_rrf(recs_ranker_pool.get(id_lector, []), recs_als_pool.get(id_lector, []))
        recomendados = fusionado[:K]
        if len(recomendados) < K:
            vistos = set(libros_leidos.get(id_lector, set())) | set(recomendados)
            extra = [libro for libro in ranking_global if libro not in vistos]
            recomendados = recomendados + extra[: K - len(recomendados)]
        ndcg_blend_por_usuario[id_lector] = ndcg_at_k(recomendados, relevantes_por_usuario.get(id_lector, set()), K)

    valores_ranker = np.array([r["ndcg_por_usuario"][u] for u in usuarios])
    valores_blend = np.array([ndcg_blend_por_usuario[u] for u in usuarios])
    diferencia = valores_blend - valores_ranker
    n = len(diferencia)

    print(f"\nNDCG@{K} ranker solo:      {valores_ranker.mean():.6f}")
    print(f"NDCG@{K} ALS solo (oficial): {r['ndcg_als']:.6f}")
    print(f"NDCG@{K} blend RRF (ranker+ALS): {valores_blend.mean():.6f}")

    se_pareado = diferencia.std(ddof=1) / math.sqrt(n)
    sigma_pareado = diferencia.mean() / se_pareado if se_pareado else float("nan")
    print(f"\nPAREADO (blend - ranker): dif media {diferencia.mean():+.6f}   sd {diferencia.std(ddof=1):.6f}   "
          f"SE {se_pareado:.6f}   -> {sigma_pareado:.2f} sigma")
    print(f"usuarios donde cambia el NDCG: {(diferencia != 0).mean():.4f}"
          f"  (mejora {(diferencia > 0).sum()}, empeora {(diferencia < 0).sum()})")

    rng = np.random.default_rng(0)
    bootstrap = np.array([rng.choice(diferencia, size=n, replace=True).mean() for _ in range(N_BOOTSTRAP)])
    print(f"bootstrap 95% CI: [{np.percentile(bootstrap, 2.5):+.6f}, {np.percentile(bootstrap, 97.5):+.6f}]"
          f"   P(dif > 0) {(bootstrap > 0).mean():.4f}")


if __name__ == "__main__":
    main()
