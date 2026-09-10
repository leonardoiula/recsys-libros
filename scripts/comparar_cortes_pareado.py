"""Compara el entrenamiento del `LGBMRanker` con **ventana rodante**
(`n_cortes > 1`) contra 1 solo corte, con un test PAREADO por usuario
sobre el mismo split/seed -- mismo aparato estadístico que
`scripts/comparar_generadores_pareado.py`, pero lo que varía acá es
cuántos cortes temporales por usuario alimentan el dataset de
entrenamiento del ranker (`n_cortes` de `preparar_pipeline`), no las
fuentes ni las features.

Uso: uv run python scripts/comparar_cortes_pareado.py

Estrategia 1 de `experiments/estrategias.md`. El corte `j` predice la
interacción `x_{m-j}` de cada usuario con una etapa 1 fiteada solo sobre
`x_1..x_{m-1-j}` -- N× la supervisión (7.932 queries hoy), sin leakage y
con 1 solo positivo por grupo. `test_final` es el mismo en todas las
configs (la interacción más reciente de cada usuario), así que los
`usuarios_test` se parean 1 a 1 y `recall_de_candidatos` NO debería
cambiar entre configs (el test-side es idéntico -- si cambia, hay un
bug).

Distinto del experimento `n_val_ranker` (revertido, −10 σ): aquel metía
varios positivos en UN grupo y compartía una etapa 1 para todos los
cortes. Ver `N_CORTES_RANKER` en `ranker.py`.

Usa `preparar_pipeline_cacheado`: la primera corrida arma un contexto por
valor de `n_cortes` (el de más cortes es el más caro, ~N× el armado del
dataset de entrenamiento), corridas siguientes sobre el mismo seed/config
son casi instantáneas.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_lectores, load_libros
from recsys.models import ranker as R

K = 20
N_POR_FUENTE = 150
N_POR_FUENTE_AUTOR = 500  # config de produccion (ver submit.py); comparte cache con evaluate_ranker.py
SEED = 42
"""El test pareado ya tiene ~5x mas poder que promediar 3 seeds
independientes (ver `comparar_generadores_pareado.py`) -- un seed suele
alcanzar."""
N_BOOTSTRAP = 2000

BASE = 1
CORTES = [3]  # cada uno se compara pareado contra BASE. Con `ctx_base` (n_cortes=1) vivo,
# n_cortes=5 (contexto ~11 GB) es limite de RAM en esta maquina (32 GB) -- para 5 usar el
# CV de `scripts/evaluate_ranker.py` (libera cada contexto antes del siguiente).
# El record actual (0.06753) es con n_cortes=5.


def _reportar_pareado(valores_a: np.ndarray, valores_b: np.ndarray, nombre_a: str, nombre_b: str) -> None:
    diferencia = valores_a - valores_b
    n = len(diferencia)

    print(f"\nn usuarios de test: {n}")
    print(f"NDCG@{K} {nombre_a}: {valores_a.mean():.6f}")
    print(f"NDCG@{K} {nombre_b}: {valores_b.mean():.6f}")

    se_no_pareado = math.sqrt(valores_a.var(ddof=1) / n + valores_b.var(ddof=1) / n)
    sigma_no_pareado = diferencia.mean() / se_no_pareado if se_no_pareado else float("nan")
    print(f"  no pareado: dif medias {diferencia.mean():+.6f}   SE {se_no_pareado:.6f}   -> {sigma_no_pareado:.2f} sigma")

    se_pareado = diferencia.std(ddof=1) / math.sqrt(n)
    sigma_pareado = diferencia.mean() / se_pareado if se_pareado else float("nan")
    print(f"  PAREADO:    dif media {diferencia.mean():+.6f}   sd {diferencia.std(ddof=1):.6f}   "
          f"SE {se_pareado:.6f}   -> {sigma_pareado:.2f} sigma")
    print(f"  usuarios donde cambia el NDCG: {(diferencia != 0).mean():.4f}"
          f"  (mejora {(diferencia > 0).sum()}, empeora {(diferencia < 0).sum()})")

    rng = np.random.default_rng(0)
    bootstrap = np.array([rng.choice(diferencia, size=n, replace=True).mean() for _ in range(N_BOOTSTRAP)])
    print(f"  bootstrap 95% CI: [{np.percentile(bootstrap, 2.5):+.6f}, {np.percentile(bootstrap, 97.5):+.6f}]"
          f"   P(dif > 0) {(bootstrap > 0).mean():.4f}")
    if se_pareado:
        print(f"  ganancia de poder: SE no pareado / SE pareado = {se_no_pareado / se_pareado:.1f}x")


def _contexto(interacciones, libros, lectores, n_cortes):
    return R.preparar_pipeline_cacheado(
        interacciones, libros, lectores, SEED,
        n_por_fuente=N_POR_FUENTE, n_por_fuente_autor=N_POR_FUENTE_AUTOR, k=K, n_cortes=n_cortes,
    )


def main() -> None:
    interacciones = load_interacciones()
    libros = load_libros()
    lectores = load_lectores()

    t0 = time.time()
    ctx_base = _contexto(interacciones, libros, lectores, BASE)
    recall_base = R.recall_de_candidatos(ctx_base)
    ndcg_base = R.ndcg_por_usuario(ctx_base, R.FEATURES)
    usuarios = ctx_base["usuarios_test"]
    print(f"n_cortes={BASE}: {len(ctx_base['group'])} grupos de entrenamiento, "
          f"recall={recall_base:.4f}, NDCG@{K}={np.mean([ndcg_base[u] for u in usuarios]):.6f}  "
          f"({time.time() - t0:.0f}s)", flush=True)
    valores_base = np.array([ndcg_base[u] for u in usuarios])

    for nc in CORTES:
        t0 = time.time()
        ctx = _contexto(interacciones, libros, lectores, nc)
        recall = R.recall_de_candidatos(ctx)
        assert ctx["usuarios_test"] == usuarios, "test-side deberia ser identico entre n_cortes"
        ndcg = R.ndcg_por_usuario(ctx, R.FEATURES)
        print(f"\nn_cortes={nc}: {len(ctx['group'])} grupos de entrenamiento "
              f"({len(ctx['group']) / len(ctx_base['group']):.2f}x), recall={recall:.4f} "
              f"(base {recall_base:.4f})  ({time.time() - t0:.0f}s)", flush=True)
        _reportar_pareado(
            np.array([ndcg[u] for u in usuarios]), valores_base,
            f"n_cortes={nc}", f"n_cortes={BASE}",
        )


if __name__ == "__main__":
    main()
