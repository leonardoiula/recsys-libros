"""Compara dos GENERADORES DE CANDIDATOS completos (no subconjuntos de
`FEATURES`) con un test PAREADO por usuario, sobre el mismo split/seed --
mismo aparato estadístico que `scripts/comparar_features_pareado.py`, pero
lo que varía acá es qué *fuentes* proponen candidatos (`fuentes_activas` de
`generar_candidatos_con_features`/`preparar_pipeline`), no qué columnas usa
el ranker.

Uso: uv run python scripts/comparar_generadores_pareado.py

Por qué existe: `comparar_features_pareado.py` solo puede aislar si el
ranker aprovecha las 3 features de *tracking* de una fuente nueva (ej.
`score_coleido_candidato`/`rank_coleido_candidato`/`en_coleido_candidato`)
-- pero A y B ahí comparten el mismo pool de candidatos (la fuente sigue
activa en ambos casos), así que nunca aisló si la fuente en sí (los
candidatos NUEVOS que trae, no sus 3 features) mueve el NDCG con el poder
estadístico del test pareado. Quedó anotado explícitamente como pendiente
en `experiments/decisiones.md` tras la ronda de la 6ª fuente (co-lectura
ítem-ítem/kNN), confirmada en Kaggle con el criterio de "positivo en los 3
seeds" (que en el pasado confirmó casos que después resultaron ruido).

Default: valida retroactivamente esa 6ª fuente -- `FUENTES_A` son las 6
fuentes actuales, `FUENTES_B` son esas mismas 6 menos "coleido". Editar
`FUENTES_A`/`FUENTES_B` (subconjuntos de `ranker.FUENTES_CANDIDATOS`) para
comparar otro par.

Reporta, para cada config, `recall_de_candidatos` (el techo barato que ya
usa `recall_candidatos.py`) ADEMÁS del NDCG pareado -- `decisiones.md`
insiste en mirar los dos números juntos: subir recall no siempre se
traduce en NDCG (ver el caso de `n_por_fuente=500`).

Usa `preparar_pipeline_cacheado`: la primera corrida arma las dos
configuraciones completas (~2x el costo de una sola, cada fuente activa/
inactiva es un contexto distinto), corridas siguientes sobre el mismo
seed/config son casi instantáneas.
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
SEEDS = [42]
"""El test pareado ya tiene ~5x más poder que promediar 3 seeds
independientes (ver `comparar_features_pareado.py`) -- un seed suele
alcanzar. Se puede extender a `[42, 7, 123]` para ver consistencia entre
splits, al costo de ~2x tiempo por seed extra (dos contextos por seed)."""
N_BOOTSTRAP = 2000

FUENTES_A = None  # las 6 fuentes actuales (default de generar_candidatos_con_features)
FUENTES_B = R.FUENTES_CANDIDATOS - {"coleido"}


def _reportar_pareado(valores_a: np.ndarray, valores_b: np.ndarray, nombre_a: str, nombre_b: str) -> None:
    diferencia = valores_a - valores_b
    n = len(diferencia)

    print(f"\nn usuarios de test: {n}")
    print(f"NDCG@{K} {nombre_a}: {valores_a.mean():.6f}")
    print(f"NDCG@{K} {nombre_b}: {valores_b.mean():.6f}")

    print("\n--- comparación NO pareada (promedios independientes) ---")
    se_no_pareado = math.sqrt(valores_a.var(ddof=1) / n + valores_b.var(ddof=1) / n)
    sigma_no_pareado = diferencia.mean() / se_no_pareado if se_no_pareado else float("nan")
    print(f"diferencia de medias: {diferencia.mean():+.6f}   SE no pareado: {se_no_pareado:.6f}   -> {sigma_no_pareado:.2f} sigma")

    print("\n--- comparación PAREADA por usuario (recomendada) ---")
    se_pareado = diferencia.std(ddof=1) / math.sqrt(n)
    sigma_pareado = diferencia.mean() / se_pareado if se_pareado else float("nan")
    print(f"diferencia media pareada: {diferencia.mean():+.6f}   sd de la diferencia: {diferencia.std(ddof=1):.6f}   SE pareado: {se_pareado:.6f}")
    print(f"-> {sigma_pareado:.2f} sigma")
    print(
        f"usuarios donde cambia el NDCG: {(diferencia != 0).mean():.4f}"
        f"  (mejora {(diferencia > 0).sum()}, empeora {(diferencia < 0).sum()})"
    )

    rng = np.random.default_rng(0)
    bootstrap = np.array([rng.choice(diferencia, size=n, replace=True).mean() for _ in range(N_BOOTSTRAP)])
    print(
        f"bootstrap 95% CI de la diferencia: "
        f"[{np.percentile(bootstrap, 2.5):+.6f}, {np.percentile(bootstrap, 97.5):+.6f}]"
    )
    print(f"P(diferencia > 0) bootstrap: {(bootstrap > 0).mean():.4f}")
    if se_pareado:
        print(f"\nganancia de poder: SE no pareado / SE pareado = {se_no_pareado/se_pareado:.1f}x")


def main() -> None:
    interacciones = load_interacciones()
    libros = load_libros()
    lectores = load_lectores()

    nombre_a = "todas" if FUENTES_A is None else "+".join(sorted(FUENTES_A))
    nombre_b = "todas" if FUENTES_B is None else "+".join(sorted(FUENTES_B))

    for seed in SEEDS:
        print(f"\n=== seed={seed}: fuentes_A=[{nombre_a}] vs fuentes_B=[{nombre_b}] ===")

        t0 = time.time()
        ctx_a = R.preparar_pipeline_cacheado(
            interacciones, libros, lectores, seed, n_por_fuente=N_POR_FUENTE, k=K, fuentes_activas=FUENTES_A
        )
        ctx_b = R.preparar_pipeline_cacheado(
            interacciones, libros, lectores, seed, n_por_fuente=N_POR_FUENTE, k=K, fuentes_activas=FUENTES_B
        )
        print(f"contextos listos en {time.time()-t0:.0f}s", flush=True)

        recall_a = R.recall_de_candidatos(ctx_a)
        recall_b = R.recall_de_candidatos(ctx_b)
        print(f"recall de candidatos -- A: {recall_a:.4f}   B: {recall_b:.4f}")

        # El split (y por lo tanto usuarios_test) no depende de fuentes_activas,
        # solo del seed -- A y B comparten exactamente los mismos usuarios, se
        # parean 1 a 1 sin reindexar.
        assert ctx_a["usuarios_test"] == ctx_b["usuarios_test"]

        ndcg_a = R.ndcg_por_usuario(ctx_a, R.FEATURES)
        ndcg_b = R.ndcg_por_usuario(ctx_b, R.FEATURES)
        print(f"listo en {time.time()-t0:.0f}s", flush=True)

        usuarios = ctx_a["usuarios_test"]
        valores_a = np.array([ndcg_a[u] for u in usuarios])
        valores_b = np.array([ndcg_b[u] for u in usuarios])
        _reportar_pareado(valores_a, valores_b, f"fuentes_A ({nombre_a})", f"fuentes_B ({nombre_b})")


if __name__ == "__main__":
    main()
