"""Diagnóstico: ¿el reranker falla en subir al top-k los libros MENOS
populares, incluso cuando ya están entre los candidatos?

Uso: uv run python scripts/diagnostico_posicion_popularidad.py

Por qué existe: `scripts/recall_candidatos.py` ya mide que, de los
usuarios donde el objetivo SÍ está entre los candidatos, el reranker
solo lo pone en el top-k el ~44% de las veces (posición mediana 29,
justo pasado el corte de k=20) -- esa plata se pierde en el *ranking en
sí*, no en la generación de candidatos. Este script agrega una variable
más a ese mismo análisis (`n_interacciones_libro`, la popularidad global
del objetivo, ya calculada como feature) para chequear la hipótesis que
motivó en su momento la 5ª fuente (resumen): los targets que las fuentes
basadas en popularidad/conectividad fallan en capturar son ~11x menos
populares que los que sí capturan. Acá se pregunta lo mismo pero un paso
más adelante en el pipeline: dado que el objetivo YA está entre los
candidatos, ¿el reranker todavía lo penaliza por ser poco popular?

Si la respuesta es sí, la mayoría de las features actuales (`score_als`,
`score_popularidad`, `rank_autor_candidato` vía popularidad global,
etc., casi todas correlacionadas con popularidad en mayor o menor
medida) probablemente no le dan al modelo suficiente palanca para
priorizar un objetivo correcto-pero-poco-popular por sobre cientos de
alternativas más populares -- la acción sería una feature/ajuste de
entrenamiento que compense ese sesgo en la etapa de *ranking*, no en la
de generación de candidatos (que es donde se viene atacando el sesgo
hasta ahora, con autor/resumen/co-lectura).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_lectores, load_libros
from recsys.models.ranker import FEATURES, evaluar_con_params, preparar_pipeline_cacheado

K = 20
N_POR_FUENTE = 150
SEED = 42


def main() -> None:
    interacciones = load_interacciones()
    libros = load_libros()
    lectores = load_lectores()

    t0 = time.time()
    ctx = preparar_pipeline_cacheado(interacciones, libros, lectores, SEED, n_por_fuente=N_POR_FUENTE, k=K)
    print(f"contexto listo en {time.time() - t0:.0f}s", flush=True)

    resultado = evaluar_con_params(ctx, None)
    modelo = resultado["modelo_ranker"]
    print(f"NDCG@{K} ranker: {resultado['ndcg_ranker']:.6f}")

    candidatos = ctx["candidatos_test"]
    test_final = ctx["test_final"]
    objetivo_por_usuario = dict(zip(test_final["id_lector"], test_final["id_libro"]))

    filas = []
    for id_lector, grupo in candidatos.groupby("id_lector", sort=False):
        objetivo = objetivo_por_usuario.get(id_lector)
        if objetivo is None:
            continue
        fila_objetivo = grupo[grupo["id_libro"] == objetivo]
        if fila_objetivo.empty:
            continue  # ya lo mide recall_de_candidatos -- acá solo interesan los casos "alcanzables"

        scores = modelo.predict(grupo[FEATURES])
        orden = grupo["id_libro"].to_numpy()[np.argsort(-scores)]
        posicion = int(np.where(orden == objetivo)[0][0])
        popularidad = float(fila_objetivo["n_interacciones_libro"].iloc[0])
        filas.append({"id_lector": id_lector, "posicion": posicion, "n_interacciones_libro": popularidad})

    df = pd.DataFrame(filas)
    print(f"\nusuarios con objetivo alcanzable (entre los candidatos): {len(df)}")

    en_top_k = df["posicion"] < K
    print(f"\npopularidad del objetivo (n_interacciones_libro) -- mediana:")
    print(f"  en el top-{K} ({en_top_k.sum()} usuarios):     {df.loc[en_top_k, 'n_interacciones_libro'].median():.1f}")
    print(f"  fuera del top-{K} ({(~en_top_k).sum()} usuarios): {df.loc[~en_top_k, 'n_interacciones_libro'].median():.1f}")

    correlacion = df["posicion"].corr(df["n_interacciones_libro"], method="spearman")
    print(f"\ncorrelación de Spearman (posición vs. popularidad): {correlacion:.4f}")
    print("(negativa y grande en magnitud = mientras MÁS popular el objetivo, MEJOR posición -- sesgo hacia popularidad)")

    print(f"\nP(en el top-{K}) por decil de popularidad del objetivo (decil 0 = menos popular):")
    df["decil"] = pd.qcut(df["n_interacciones_libro"], 10, labels=False, duplicates="drop")
    resumen = df.groupby("decil").agg(
        n=("posicion", "size"),
        popularidad_mediana=("n_interacciones_libro", "median"),
        p_en_top_k=("posicion", lambda s: (s < K).mean()),
    )
    print(resumen.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
