"""Chequeo barato: ¿tienen headroom de recall las fuentes de resumen y
co-lectura si se les sube el presupuesto de 150 a 500?

Uso (UN proceso por config, para no acumular RAM):

    uv run python scripts/probe_presupuesto_fuentes.py --fuente coleido --n 150
    uv run python scripts/probe_presupuesto_fuentes.py --fuente coleido --n 500
    uv run python scripts/probe_presupuesto_fuentes.py --fuente resumen --n 150
    uv run python scripts/probe_presupuesto_fuentes.py --fuente resumen --n 500
    uv run python scripts/probe_presupuesto_fuentes.py --resumen

A diferencia de la fuente de autor (que tiene una patología de asignación
real: el presupuesto total se lo comen los autores favoritos y los
secundarios quedan en cero -- ver `scripts/diagnostico_presupuesto_autor.py`),
resumen y co-lectura son un simple top-N por usuario. Subirles el
presupuesto es el mismo escenario que `n_por_fuente=500` global, que ya
se descartó (+30% recall, NDCG plano). Este script mide si el recall de
CADA fuente por separado (`fuentes_activas={fuente}`) tiene headroom
150->500 antes de invertir en threadear un `n_por_fuente_<fuente>` y
correr el sweep completo.

Escribe `data/cache/probe_fuentes/<fuente>_<n>.json`; `--resumen` los junta.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_lectores, load_libros
from recsys.models import ranker as R

K = 20
SEED = 42
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "cache" / "probe_fuentes"
FUENTES = ["resumen", "coleido"]


def correr(fuente: str, n: int) -> None:
    interacciones = load_interacciones()
    libros = load_libros()
    lectores = load_lectores()

    t0 = time.time()
    ctx = R.preparar_pipeline_cacheado(
        interacciones, libros, lectores, SEED,
        n_por_fuente=n, k=K, fuentes_activas=frozenset({fuente}),
    )
    tamanos = ctx["candidatos_test"].groupby("id_lector").size()
    recall = R.recall_de_candidatos(ctx)

    salida = {
        "fuente": fuente, "n": n, "recall": recall,
        "cand_media": float(tamanos.mean()), "cand_p90": float(tamanos.quantile(0.9)),
        "cand_max": int(tamanos.max()), "segundos": time.time() - t0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{fuente}_{n}.json").write_text(json.dumps(salida))
    print(f"=== fuente={fuente} n={n} ({salida['segundos']:.0f}s) ===")
    print(f"recall (solo esta fuente): {recall:.4f}   candidatos/usuario media={tamanos.mean():.1f} p90={tamanos.quantile(.9):.0f} max={tamanos.max()}")

    del ctx
    gc.collect()


def resumen() -> None:
    archivos = {p.stem: json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("*.json"))}
    print(f"{'fuente':>10} {'recall@150':>12} {'recall@500':>12} {'d recall':>12} {'cand p90 @500':>14}")
    for fuente in FUENTES:
        a, b = archivos.get(f"{fuente}_150"), archivos.get(f"{fuente}_500")
        if a and b:
            print(f"{fuente:>10} {a['recall']:>12.4f} {b['recall']:>12.4f} {b['recall'] - a['recall']:>+12.4f} {b['cand_p90']:>14.0f}")
        else:
            print(f"{fuente:>10}  (faltan json: {fuente}_150 y/o {fuente}_500)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resumen", action="store_true", help="junta los .json ya generados")
    ap.add_argument("--fuente", choices=FUENTES)
    ap.add_argument("--n", type=int)
    args = ap.parse_args()

    if args.resumen:
        resumen()
    elif args.fuente and args.n is not None:
        correr(args.fuente, args.n)
    else:
        ap.error("pasar --resumen, o --fuente y --n juntos")


if __name__ == "__main__":
    main()
