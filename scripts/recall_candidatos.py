"""Diagnóstico: recall del set de candidatos (unión de las 3 fuentes de la
etapa 1 del ranker) y cuánto de ese techo termina capturando el reranking.

Uso: uv run python scripts/recall_candidatos.py

Por qué existe: el NDCG@20 del ranker es aproximadamente
`recall_candidatos × fracción_del_techo_capturada_por_el_ranking`. Las
últimas rondas de features (género macro, tamaño de editorial, señales
cruzadas lector↔libro) movieron el segundo factor menos del 2% cada
vez, mientras que el primero está muy por debajo de 1 (ver
`experiments/modelo_actual.md`, sección "Recomendación: ¿cambiar de
paradigma?") -- este script mide justamente ese primer factor, para
decidir sobre la etapa 1 (candidatos) sin correr el pipeline completo
del ranker en cada intento. Compara distintos `n_por_fuente` editando
la constante de abajo y volviendo a correr.

`recall_candidatos` es el TECHO absoluto de NDCG@20 que puede lograr el
reranking con este set de candidatos -- ningún ajuste de LightGBM ni de
features puede recomendar un libro que no está entre los candidatos.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_lectores, load_libros
from recsys.models.ranker import FEATURES, evaluar_con_params, preparar_pipeline

K = 20
N_POR_FUENTE = 150
SEED = 42


def main() -> None:
    interacciones = load_interacciones()
    libros = load_libros()
    lectores = load_lectores()

    t0 = time.time()
    ctx = preparar_pipeline(interacciones, libros, lectores, SEED, n_por_fuente=N_POR_FUENTE, k=K)
    print(f"contexto listo en {time.time()-t0:.0f}s", flush=True)

    candidatos = ctx["candidatos_test"]
    test_final = ctx["test_final"]

    candidatos_por_usuario = candidatos.groupby("id_lector")["id_libro"].agg(set).to_dict()
    tamanos = candidatos.groupby("id_lector").size()

    hits = sum(
        1
        for _, fila in test_final.iterrows()
        if fila["id_libro"] in candidatos_por_usuario.get(fila["id_lector"], set())
    )
    total = len(test_final)
    recall = hits / total

    print(f"\nusuarios en test_final: {total}")
    print(f"candidatos por usuario: media={tamanos.mean():.1f} min={tamanos.min()} max={tamanos.max()}")
    print(f"RECALL del set de candidatos (techo de NDCG@{K} del ranker): {recall:.4f}")

    resultado = evaluar_con_params(ctx, None)
    print(f"NDCG@{K} ranker: {resultado['ndcg_ranker']:.6f}  ALS solo: {resultado['ndcg_als']:.6f}")
    print(f"fracción del techo capturada por el reranking: {resultado['ndcg_ranker']/recall:.4f}")

    # Posición del objetivo dentro del ranking del ranker, para los usuarios
    # donde el objetivo SÍ está entre los candidatos (si no está, ninguna
    # posición del ranker lo va a recuperar -- eso ya lo mide `recall`).
    modelo = resultado["modelo_ranker"]
    objetivo_por_usuario = dict(zip(test_final["id_lector"], test_final["id_libro"]))
    posiciones = []
    for id_lector, grupo in candidatos.groupby("id_lector", sort=False):
        objetivo = objetivo_por_usuario.get(id_lector)
        if objetivo is None or objetivo not in set(grupo["id_libro"]):
            continue
        scores = modelo.predict(grupo[FEATURES])
        orden = grupo["id_libro"].to_numpy()[np.argsort(-scores)]
        posiciones.append(int(np.where(orden == objetivo)[0][0]))

    posiciones_arr = np.array(posiciones)
    print(f"\nobjetivo entre los candidatos: {len(posiciones_arr)} usuarios")
    for limite in (1, 5, 10, 20, 50, 100, 200):
        print(f"  en top-{limite}: {(posiciones_arr < limite).mean():.4f}")
    print(f"  posición mediana: {np.median(posiciones_arr):.0f}")


if __name__ == "__main__":
    main()
