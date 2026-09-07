"""Pre-screen: ¿conviene darle a la fuente de autor un presupuesto propio
más grande que el `n_por_fuente=150` de las otras 5 fuentes?

Uso: uv run python scripts/screen_presupuesto_autor.py

Contexto: `scripts/diagnostico_presupuesto_autor.py` midió que el 39% de
los usuarios agota el presupuesto de 150 de la fuente de autor en sus
autores favoritos, y hasta un ~5% (cota superior) pierde el libro-objetivo
del set de candidatos solo por ese tope. `generar_candidatos_con_features`
acepta ahora `n_por_fuente_autor` (default `None` = usar `n_por_fuente`)
para el tope TOTAL de esa fuente sin tocar las otras.

Este script barre unos pocos valores de `n_por_fuente_autor` sobre
seed=42, reportando -- misma disciplina que las fuentes anteriores:

- `recall_de_candidatos`: el techo barato (¿el objetivo está entre los
  candidatos?), mismo que `scripts/recall_candidatos.py`.
- NDCG@20 del ranker (un solo seed) y su test PAREADO por usuario contra
  el baseline (`n_por_fuente_autor=None`) -- `decisiones.md` insiste en
  mirar recall y NDCG juntos: subir recall no siempre sube el NDCG (ver
  `n_por_fuente=500`).

Si algún valor mejora recall Y el NDCG pareado sale por encima del ruido,
el siguiente paso es el CV de 3 seeds (`scripts/evaluate_ranker.py` con
`N_POR_FUENTE_AUTOR` seteado). Si no, se descarta sin gastar el CV.
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
SEED = 42
N_BOOTSTRAP = 2000

VALORES = [None, 300, 500, 800]  # None = baseline (usa n_por_fuente=150)


def _reportar_pareado(dif: np.ndarray, nombre: str) -> None:
    n = len(dif)
    se = dif.std(ddof=1) / math.sqrt(n)
    sigma = dif.mean() / se if se else float("nan")
    print(f"  {nombre}: dif media pareada {dif.mean():+.6f}  SE {se:.6f}  -> {sigma:.2f} sigma")
    print(
        f"    usuarios donde cambia: {(dif != 0).mean():.4f} "
        f"(mejora {(dif > 0).sum()}, empeora {(dif < 0).sum()})"
    )
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(dif, size=n, replace=True).mean() for _ in range(N_BOOTSTRAP)])
    print(
        f"    bootstrap 95% CI [{np.percentile(boot, 2.5):+.6f}, {np.percentile(boot, 97.5):+.6f}]"
        f"   P(mejora) {(boot > 0).mean():.4f}"
    )


def main() -> None:
    interacciones = load_interacciones()
    libros = load_libros()
    lectores = load_lectores()

    ndcg_por_usuario_baseline = None
    usuarios_baseline = None

    for valor in VALORES:
        etiqueta = "baseline (None=150)" if valor is None else f"n_por_fuente_autor={valor}"
        t0 = time.time()
        ctx = R.preparar_pipeline_cacheado(
            interacciones, libros, lectores, SEED,
            n_por_fuente=N_POR_FUENTE, n_por_fuente_autor=valor, k=K,
        )
        tamanos = ctx["candidatos_test"].groupby("id_lector").size()
        recall = R.recall_de_candidatos(ctx)
        res = R.evaluar_con_params(ctx, None)
        ndcg_usuario = R.ndcg_por_usuario(ctx, R.FEATURES)
        usuarios = ctx["usuarios_test"]

        print(f"\n=== {etiqueta} ({time.time()-t0:.0f}s) ===")
        print(f"candidatos/usuario  media={tamanos.mean():.1f}  mediana={tamanos.median():.0f}  p90={tamanos.quantile(.9):.0f}  max={tamanos.max()}")
        print(f"recall del set de candidatos: {recall:.4f}")
        print(f"NDCG@{K} ranker (seed={SEED}): {res['ndcg_ranker']:.6f}   ALS solo: {res['ndcg_als']:.6f}")
        print(f"eficiencia de ranking (NDCG/recall): {res['ndcg_ranker']/recall:.4f}")

        if valor is None:
            ndcg_por_usuario_baseline = ndcg_usuario
            usuarios_baseline = usuarios
        else:
            assert usuarios == usuarios_baseline, "el split no debe depender de n_por_fuente_autor"
            dif = np.array([ndcg_usuario[u] - ndcg_por_usuario_baseline[u] for u in usuarios])
            _reportar_pareado(dif, f"pareado vs baseline")


if __name__ == "__main__":
    main()
