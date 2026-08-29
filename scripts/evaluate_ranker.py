"""Evaluación cross-validada del ranker de dos etapas (ALS + género +
popularidad -> LightGBM) contra ALS solo.

Uso: uv run python scripts/evaluate_ranker.py

Para cada seed en `SEEDS`, arma un split de **tres niveles** (evita que
el ranker vea, como features, scores calculados con la misma etiqueta
que tiene que predecir):

- `train_candidatos`: fit de ALS/popularidad/género (las señales).
- `train_ranker`: etiquetas conocidas (el "próximo libro" de cada
  usuario) para entrenar el `LGBMRanker`, con candidatos/features
  generados por los modelos fit solo en `train_candidatos`.
- `test_final`: hold-out final, aislado de todo lo anterior, para medir
  NDCG@20 del pipeline completo.

Se evalúa sobre varios seeds (no uno solo) a propósito: un sweep de ALS
sobre un único split mejoró el NDCG local +11.5% pero empeoró el score
real de Kaggle -13.5% (ver experiments/bitacora.md, "Regresión en
Kaggle") -- este script existe para no repetir ese error con el ranker.

ALS solo se evalúa con el mismo `train_candidatos` (no `train_candidatos_full`)
para que la comparación sea controlada: aísla el efecto de agregar la
capa de ranking, no el de tener más o menos datos de entrenamiento.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_libros, split_train_val, libros_leidos_por_usuario
from recsys.evaluation import evaluar_multisplit, evaluar_ndcg_personalizado
from recsys.models.als import fit_als
from recsys.models.als import recomendar_por_usuario as recomendar_als
from recsys.models.popularity import fit_popularity
from recsys.models.popularity_segmentada import fit_popularity_por_genero, genero_preferido_por_usuario
from recsys.models.ranker import armar_dataset_entrenamiento, fit_ranker, generar_candidatos_con_features
from recsys.models.ranker import recomendar_por_usuario as recomendar_ranker

K = 20
N_POR_FUENTE = 150
SEEDS = [42, 7, 123]

interacciones = load_interacciones()
libros = load_libros()


def evaluar_seed(seed: int) -> dict:
    train_candidatos_full, test_final = split_train_val(interacciones, n_val=1, seed=seed)
    train_candidatos, train_ranker = split_train_val(train_candidatos_full, n_val=1, seed=seed + 1000)

    # --- Stage 1: señales, fit solo con train_candidatos ---
    libros_leidos_stage1 = libros_leidos_por_usuario(train_candidatos)
    n_interacciones_por_usuario = train_candidatos.groupby("id_lector").size().to_dict()
    stats_popularidad = fit_popularity(train_candidatos)
    ranking_global = stats_popularidad["id_libro"].tolist()
    stats_por_genero = fit_popularity_por_genero(train_candidatos, libros)
    genero_por_usuario = genero_preferido_por_usuario(train_candidatos, libros)
    modelo_als, matriz, fila_por_usuario, libros_por_columna = fit_als(train_candidatos)

    args_candidatos = dict(
        modelo_als=modelo_als,
        matriz_usuario_libro=matriz,
        fila_por_usuario=fila_por_usuario,
        libros_por_columna=libros_por_columna,
        stats_popularidad=stats_popularidad,
        stats_por_genero=stats_por_genero,
        genero_por_usuario=genero_por_usuario,
        n_interacciones_por_usuario=n_interacciones_por_usuario,
        n_por_fuente=N_POR_FUENTE,
    )

    # --- Stage 2: entrenar el ranker con las etiquetas de train_ranker ---
    usuarios_ranker = train_ranker["id_lector"].unique().tolist()
    candidatos_train_ranker = generar_candidatos_con_features(
        usuarios=usuarios_ranker, libros_leidos=libros_leidos_stage1, **args_candidatos
    )
    X, y, group = armar_dataset_entrenamiento(
        candidatos_train_ranker, train_ranker[["id_lector", "id_libro"]], n_por_fuente=N_POR_FUENTE
    )
    modelo_ranker = fit_ranker(X, y, group)

    # --- Evaluación final en test_final (ya leido = train_candidatos + train_ranker) ---
    libros_leidos_hasta_ranker = libros_leidos_por_usuario(train_candidatos_full)
    usuarios_test = test_final["id_lector"].unique().tolist()

    candidatos_test = generar_candidatos_con_features(
        usuarios=usuarios_test, libros_leidos=libros_leidos_hasta_ranker, **args_candidatos
    )
    recs_ranker = recomendar_ranker(
        usuarios=usuarios_test, modelo_ranker=modelo_ranker, candidatos_df=candidatos_test,
        ranking_global=ranking_global, libros_leidos=libros_leidos_hasta_ranker, k=K,
    )
    ndcg_ranker = evaluar_ndcg_personalizado(test_final, recs_ranker, K)

    recs_als = recomendar_als(
        usuarios=usuarios_test, modelo=modelo_als, matriz_usuario_libro=matriz,
        fila_por_usuario=fila_por_usuario, libros_por_columna=libros_por_columna,
        ranking_global=ranking_global, libros_leidos=libros_leidos_hasta_ranker, k=K,
    )
    ndcg_als = evaluar_ndcg_personalizado(test_final, recs_als, K)

    return {"ndcg_als": ndcg_als, "ndcg_ranker": ndcg_ranker}


def main() -> None:
    resultados_por_seed = {}
    for seed in SEEDS:
        t0 = time.time()
        resultados_por_seed[seed] = evaluar_seed(seed)
        r = resultados_por_seed[seed]
        print(f"seed={seed}: ALS={r['ndcg_als']:.6f}  ranker={r['ndcg_ranker']:.6f}  ({time.time()-t0:.1f}s)")

    resumen_als = evaluar_multisplit(lambda s: resultados_por_seed[s]["ndcg_als"], SEEDS)
    resumen_ranker = evaluar_multisplit(lambda s: resultados_por_seed[s]["ndcg_ranker"], SEEDS)

    print("\n=== Resumen (media ± desvío sobre 3 seeds) ===")
    print(f"ALS solo:            {resumen_als['media']:.6f} ± {resumen_als['desvio']:.6f}  {resumen_als['valores']}")
    print(f"Ranker (dos etapas): {resumen_ranker['media']:.6f} ± {resumen_ranker['desvio']:.6f}  {resumen_ranker['valores']}")


if __name__ == "__main__":
    main()
