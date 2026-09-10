"""¿Ensamblar el `LGBMRanker` (seed-bag) mejora sobre el modelo único?

Uso: uv run python scripts/comparar_ensamble_pareado.py

Estrategia 3 de `experiments/estrategias.md`. Sobre UN contexto cacheado
(seed=42, `n_por_fuente_autor=500` = producción) compara, con test PAREADO
por usuario, sin tocar `ranker.py`:

- base           : `fit_ranker` con los defaults actuales (determinístico)
- estocastico-1  : 1 modelo con `subsample`/`colsample` < 1 (para aislar
                   si esa aleatoriedad sola cambia algo)
- ensamble-N raw : N modelos estocásticos con seeds distintos, promedio de
                   los scores crudos de `.predict()`
- ensamble-N rank: idem pero promediando el rank DENTRO del usuario (más
                   robusto a que los scores de lambdarank no tengan escala
                   comparable entre modelos)

Reporta el paired test de cada uno vs base + el Δ por decil de popularidad
del objetivo.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_lectores, load_libros
from recsys.evaluation import ndcg_at_k
from recsys.models import ranker as R

K = 20
N_POR_FUENTE = 150
N_POR_FUENTE_AUTOR = 500
SEED = 42
N_BOOTSTRAP = 2000
N_BAG = 5
SEEDS_BAG = [42, 7, 123, 2024, 99]
PARAMS_ESTOC = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8)


def _reportar(dif: np.ndarray, nombre: str) -> None:
    n = len(dif)
    se = dif.std(ddof=1) / math.sqrt(n)
    sigma = dif.mean() / se if se else float("nan")
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(dif, size=n, replace=True).mean() for _ in range(N_BOOTSTRAP)])
    print(f"  {nombre:<22} dif {dif.mean():+.6f}  {sigma:>5.2f} sigma  "
          f"CI95 [{np.percentile(boot, 2.5):+.6f}, {np.percentile(boot, 97.5):+.6f}]  "
          f"P(mejora) {(boot > 0).mean():.4f}  (mejora {(dif > 0).sum()}, empeora {(dif < 0).sum()})")


def main() -> None:
    interacciones = load_interacciones()
    libros = load_libros()
    lectores = load_lectores()
    pop_global = interacciones.groupby("id_libro").size()

    t0 = time.time()
    ctx = R.preparar_pipeline_cacheado(
        interacciones, libros, lectores, SEED,
        n_por_fuente=N_POR_FUENTE, n_por_fuente_autor=N_POR_FUENTE_AUTOR, k=K,
    )
    print(f"contexto listo en {time.time() - t0:.0f}s", flush=True)

    X, y, group = ctx["X"], ctx["y"], ctx["group"]
    feats = list(X.columns)
    cand = {u: g for u, g in ctx["candidatos_test"].groupby("id_lector", sort=False)}
    rel = ctx["test_final"].groupby("id_lector")["id_libro"].agg(set).to_dict()
    libros_leidos, ranking_global, usuarios = ctx["libros_leidos_hasta_ranker"], ctx["ranking_global"], ctx["usuarios_test"]

    def ndcg_dict(score_fn) -> dict:
        out = {}
        for u in usuarios:
            g = cand.get(u)
            if g is None or len(g) == 0:
                recs = []
            else:
                s = score_fn(g[feats])
                recs = list(g["id_libro"].to_numpy()[np.argsort(-s)][:K])
            if len(recs) < K:
                vistos = set(libros_leidos.get(u, set())) | set(recs)
                recs += [b for b in ranking_global if b not in vistos][: K - len(recs)]
            out[u] = ndcg_at_k(recs, rel.get(u, set()), K)
        return out

    print("entrenando modelos...", flush=True)
    m_base = R.fit_ranker(X, y, group)
    m_estoc = R.fit_ranker(X, y, group, random_state=SEED, **PARAMS_ESTOC)
    bag = [R.fit_ranker(X, y, group, random_state=s, **PARAMS_ESTOC) for s in SEEDS_BAG]
    print(f"listo en {time.time() - t0:.0f}s", flush=True)

    configs = {
        "base": lambda Xg: m_base.predict(Xg),
        "estocastico-1": lambda Xg: m_estoc.predict(Xg),
        f"ensamble-{N_BAG} raw": lambda Xg: np.mean([m.predict(Xg) for m in bag], axis=0),
        f"ensamble-{N_BAG} rank": lambda Xg: np.mean([rankdata(m.predict(Xg)) for m in bag], axis=0),
    }
    ndcg = {k: ndcg_dict(fn) for k, fn in configs.items()}

    print("\nNDCG@20 medio:")
    for k in configs:
        print(f"  {k:<24} {np.mean([ndcg[k][u] for u in usuarios]):.6f}")

    base_arr = np.array([ndcg["base"][u] for u in usuarios])
    print("\ntest pareado vs base:")
    for k in [c for c in configs if c != "base"]:
        _reportar(np.array([ndcg[k][u] for u in usuarios]) - base_arr, k)

    objetivo = dict(zip(ctx["test_final"]["id_lector"], ctx["test_final"]["id_libro"]))
    pop = np.array([float(pop_global.get(objetivo.get(u), 0)) for u in usuarios])
    mejor = f"ensamble-{N_BAG} rank"
    dfd = pd.DataFrame({"d": np.array([ndcg[mejor][u] for u in usuarios]) - base_arr, "pop": pop})
    dfd["decil"] = pd.qcut(dfd["pop"], 10, labels=False, duplicates="drop")
    print(f"\nd NDCG ({mejor} - base) por decil de popularidad del objetivo:")
    print(dfd.groupby("decil").agg(n=("d", "size"), pop_mediana=("pop", "median"),
                                   d_media=("d", "mean")).to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
