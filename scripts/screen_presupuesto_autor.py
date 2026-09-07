"""Pre-screen: ¿conviene darle a la fuente de autor un presupuesto propio
(`n_por_fuente_autor`) más grande que el `n_por_fuente=150` de las otras 5?

Uso (UN valor por proceso, para no acumular RAM -- ver abajo):

    uv run python scripts/screen_presupuesto_autor.py --valor none
    uv run python scripts/screen_presupuesto_autor.py --valor 300
    uv run python scripts/screen_presupuesto_autor.py --valor 500
    uv run python scripts/screen_presupuesto_autor.py --resumen

Contexto: `scripts/diagnostico_presupuesto_autor.py` midió que el 39% de
los usuarios agota el presupuesto de 150 de la fuente de autor en sus
autores favoritos, y hasta un ~5% (cota superior) pierde el libro-objetivo
del set de candidatos solo por ese tope. `generar_candidatos_con_features`
acepta `n_por_fuente_autor` (default `None` = usar `n_por_fuente`) para el
tope TOTAL de esa fuente sin tocar las otras.

**Por qué un proceso por valor**: cada contexto de `preparar_pipeline`
pesa ~3-4 GB en RAM; recorrer varios valores en un mismo proceso mantiene
el contexto anterior vivo mientras se arma el siguiente (Python/pandas no
siempre devuelven la memoria liberada al SO) y llegó a agotar la RAM de
la máquina. Cada `--valor` corre en su propio proceso, escribe su
resultado a `data/cache/screen_nfa/<valor>.json` (gitignored) y sale;
`--resumen` junta esos archivos.

Reporta, para cada valor -- misma disciplina que las fuentes anteriores:
- `recall_de_candidatos`: el techo barato (¿el objetivo está entre los
  candidatos?), mismo que `scripts/recall_candidatos.py`.
- NDCG@20 del ranker (un solo seed) + test PAREADO por usuario contra el
  baseline (`--valor none`).
- **Δ NDCG por bucket de actividad del usuario** (interacciones totales):
  chequeo de generalización -- ¿la mejora viene de los heavy users (que
  son los que tocan el tope y los que dominan Kaggle) SIN dañar a los
  casuales? Si un valor ayuda a los heavy y perjudica a los casuales es
  una bandera roja aunque el promedio suba.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_lectores, load_libros
from recsys.models import ranker as R

K = 20
N_POR_FUENTE = 150
SEED = 42
N_BOOTSTRAP = 2000
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "screen_nfa"

# Buckets de actividad = interacciones TOTALES del usuario (no solo train),
# para poder cruzar con la población de ejemplo.csv / Kaggle. Mismos cortes
# que evaluation.BINS_ACTIVIDAD_DEFAULT.
BINS = [0, 2, 5, 10, 20, 50, 100, math.inf]
LABELS = ["1", "2-4", "5-9", "10-19", "20-49", "50-99", "100+"]


def _clave(valor: int | None) -> str:
    return "none" if valor is None else str(valor)


def correr_valor(valor: int | None) -> None:
    interacciones = load_interacciones()
    libros = load_libros()
    lectores = load_lectores()
    n_int_total = interacciones.groupby("id_lector").size()

    t0 = time.time()
    ctx = R.preparar_pipeline_cacheado(
        interacciones, libros, lectores, SEED,
        n_por_fuente=N_POR_FUENTE, n_por_fuente_autor=valor, k=K,
    )
    tamanos = ctx["candidatos_test"].groupby("id_lector").size()
    recall = R.recall_de_candidatos(ctx)
    ndcg_u = R.ndcg_por_usuario(ctx, R.FEATURES)
    usuarios = list(ctx["usuarios_test"])
    ndcg_als = float(ctx["ndcg_als"])
    ndcg_mean = float(np.mean([ndcg_u[u] for u in usuarios]))

    act = n_int_total.reindex(usuarios).fillna(0).to_numpy()
    buckets = pd.cut(act, bins=BINS, labels=LABELS, right=False)
    por_bucket = {}
    for lab in LABELS:
        idx = [u for u, b in zip(usuarios, buckets) if b == lab]
        if idx:
            por_bucket[lab] = {"n": len(idx), "ndcg": float(np.mean([ndcg_u[u] for u in idx]))}

    salida = {
        "valor": _clave(valor),
        "recall": recall,
        "ndcg_mean": ndcg_mean,
        "ndcg_als": ndcg_als,
        "eficiencia": ndcg_mean / recall if recall else float("nan"),
        "cand_por_usuario": {
            "media": float(tamanos.mean()), "mediana": float(tamanos.median()),
            "p90": float(tamanos.quantile(0.9)), "max": int(tamanos.max()),
        },
        "por_bucket": por_bucket,
        "ndcg_por_usuario": {str(u): float(ndcg_u[u]) for u in usuarios},
        "segundos": time.time() - t0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{_clave(valor)}.json").write_text(json.dumps(salida))

    print(f"=== n_por_fuente_autor={_clave(valor)} ({salida['segundos']:.0f}s) ===")
    print(f"candidatos/usuario  media={tamanos.mean():.1f} mediana={tamanos.median():.0f} p90={tamanos.quantile(.9):.0f} max={tamanos.max()}")
    print(f"recall={recall:.4f}  NDCG@{K} ranker={ndcg_mean:.6f}  ALS={ndcg_als:.6f}  eficiencia={salida['eficiencia']:.4f}")
    print("por bucket de actividad:", {lab: round(v["ndcg"], 4) for lab, v in por_bucket.items()})

    del ctx, ndcg_u
    gc.collect()


def _pareado(dif: np.ndarray) -> str:
    n = len(dif)
    se = dif.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    sigma = dif.mean() / se if se else float("nan")
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(dif, size=n, replace=True).mean() for _ in range(N_BOOTSTRAP)])
    return (
        f"dif media pareada {dif.mean():+.6f}  {sigma:.2f} sigma  "
        f"CI95 [{np.percentile(boot, 2.5):+.6f}, {np.percentile(boot, 97.5):+.6f}]  "
        f"P(mejora) {(boot > 0).mean():.4f}  "
        f"(mejora {(dif > 0).sum()}, empeora {(dif < 0).sum()})"
    )


def resumen() -> None:
    archivos = {p.stem: json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("*.json"))}
    if "none" not in archivos:
        sys.exit(f"falta {OUT_DIR / 'none.json'} -- correr primero --valor none")
    base = archivos.pop("none")

    print(f"{'valor':>8} {'recall':>9} {'NDCG@20':>10} {'eficiencia':>11} {'cand/u p90':>11}")
    print(f"{'none':>8} {base['recall']:>9.4f} {base['ndcg_mean']:>10.6f} {base['eficiencia']:>11.4f} {base['cand_por_usuario']['p90']:>11.0f}")
    for clave, d in archivos.items():
        print(f"{clave:>8} {d['recall']:>9.4f} {d['ndcg_mean']:>10.6f} {d['eficiencia']:>11.4f} {d['cand_por_usuario']['p90']:>11.0f}")

    usuarios = list(base["ndcg_por_usuario"])
    ordenados = sorted(archivos.items(), key=lambda kv: int(kv[0]))
    previos = [("none", base)] + ordenados
    for i, (clave, d) in enumerate(ordenados):
        assert set(d["ndcg_por_usuario"]) == set(usuarios), f"{clave}: distinto set de usuarios que el baseline"
        dif_base = np.array([d["ndcg_por_usuario"][u] - base["ndcg_por_usuario"][u] for u in usuarios])
        clave_prev, d_prev = previos[i]
        dif_prev = np.array([d["ndcg_por_usuario"][u] - d_prev["ndcg_por_usuario"][u] for u in usuarios])
        print(f"\n--- n_por_fuente_autor={clave} ---")
        print(f"  vs none: {_pareado(dif_base)}")
        if clave_prev != "none":
            print(f"  vs {clave_prev}:  {_pareado(dif_prev)}")
        print(f"  d NDCG por bucket de actividad (interacciones totales del usuario), vs none:")
        for lab in LABELS:
            b0, b1 = base["por_bucket"].get(lab), d["por_bucket"].get(lab)
            if b0 and b1:
                print(f"    {lab:>7} (n={b1['n']:>4}): {b0['ndcg']:.4f} -> {b1['ndcg']:.4f}  ({b1['ndcg'] - b0['ndcg']:+.4f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--valor", help="'none' (baseline) o un entero para n_por_fuente_autor")
    g.add_argument("--resumen", action="store_true", help="junta los .json ya generados y compara")
    args = ap.parse_args()

    if args.resumen:
        resumen()
    else:
        valor = None if args.valor.lower() in ("none", "baseline") else int(args.valor)
        correr_valor(valor)


if __name__ == "__main__":
    main()
