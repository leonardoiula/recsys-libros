"""Diagnóstico: ¿cuántos ejemplos de entrenamiento del `LGBMRanker`
vienen de usuarios con muy poco historial, y valdría la pena filtrarlos?

Uso: uv run python scripts/diagnostico_historial_ranker.py

El split de 3 niveles (`preparar_pipeline`):
  train_candidatos_full, test_final = split_train_val(interacciones, n_val=1, seed)
  train_candidatos, train_ranker    = split_train_val(train_candidatos_full, n_val=1, seed+1000)

`train_ranker` = 1 fila por usuario = el positivo con el que se entrena el
`LGBMRanker`. Un usuario entra en `train_ranker` solo si tiene >=3
interacciones totales; y con exactamente 3, su `train_candidatos` (con el
que se calculan TODAS sus features: ALS, TF-IDF, co-lectura, recencia...)
tiene solo 1 interacción. Ese ejemplo aporta casi nada de señal y puede
diluir el gradiente de los usuarios ricos.

Este script (sin correr el pipeline) muestra la distribución de historial
en `train_candidatos` de los usuarios de `train_ranker`, y cruza el
historial con la popularidad del libro-objetivo (la franja media es la
que peor rankea el modelo).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, split_train_val

SEED = 42


def main() -> None:
    interacciones = load_interacciones()
    pop_global = interacciones.groupby("id_libro").size()

    train_candidatos_full, test_final = split_train_val(interacciones, n_val=1, seed=SEED)
    train_candidatos, train_ranker = split_train_val(train_candidatos_full, n_val=1, seed=SEED + 1000)

    hist = train_candidatos.groupby("id_lector").size()  # historial usable por usuario
    usuarios_ranker = train_ranker["id_lector"].unique()
    hist_r = hist.reindex(usuarios_ranker).fillna(0).astype(int)

    n = len(usuarios_ranker)
    print(f"usuarios en train_ranker (ejemplos de entrenamiento del LGBMRanker): {n}")
    print(f"usuarios en test_final (evaluación): {test_final['id_lector'].nunique()}")
    print(f"\nhistorial en train_candidatos de los usuarios de train_ranker:")
    print(f"  media={hist_r.mean():.1f}  mediana={hist_r.median():.0f}  p10={hist_r.quantile(.1):.0f}  p90={hist_r.quantile(.9):.0f}  max={hist_r.max()}")

    print("\n% de ejemplos de entrenamiento con historial <= umbral:")
    for u in (1, 2, 3, 5, 10, 20):
        frac = (hist_r <= u).mean()
        print(f"  <= {u:>2} interacciones: {frac:.1%}  ({int((hist_r <= u).sum())} usuarios)")

    # ¿los usuarios con poco historial tienen objetivos de popularidad distinta?
    objetivo = dict(zip(train_ranker["id_lector"], train_ranker["id_libro"]))
    df = pd.DataFrame({
        "hist": hist_r.values,
        "pop_objetivo": [float(pop_global.get(objetivo.get(u), 0)) for u in usuarios_ranker],
    })
    df["bucket_hist"] = pd.cut(df["hist"], [0, 2, 5, 10, 20, 50, np.inf],
                               labels=["1-2", "3-5", "6-10", "11-20", "21-50", "50+"], right=True)
    print("\npopularidad mediana del libro-objetivo por bucket de historial del usuario:")
    print(df.groupby("bucket_hist", observed=True).agg(
        n=("pop_objetivo", "size"), pop_objetivo_mediana=("pop_objetivo", "median")
    ).to_string(float_format=lambda x: f"{x:.0f}"))


if __name__ == "__main__":
    main()
