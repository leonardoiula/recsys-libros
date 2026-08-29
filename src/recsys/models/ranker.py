"""Ranker de dos etapas: candidatos de ALS + popularidad por género +
popularidad global, reordenados por un `LGBMRanker` (LightGBM,
objetivo `lambdarank`) entrenado para combinar esas tres señales mejor
de lo que cada una hace sola.

Un ranker que usa el score de otros modelos como *features* necesita que
esos scores salgan de datos que el ranker no vio como etiqueta -- si no,
memoriza en vez de aprender a combinar señales. Por eso el entrenamiento
de este módulo espera un split de **tres niveles** (ver
`scripts/evaluate_ranker.py`): un tramo para fitear ALS/popularidad/
género (las fuentes de candidatos), otro tramo con etiquetas conocidas
para entrenar el `LGBMRanker`, y un tercero de hold-out final para medir
NDCG@k del pipeline completo.

Este módulo se evalúa con validación cruzada sobre varios splits/seeds
(`evaluation.evaluar_multisplit`), no un solo split -- después del
episodio en el que un sweep de ALS sobre un único split mejoró el NDCG
local pero empeoró el score real de Kaggle, ver `experiments/bitacora.md`.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURES = [
    "score_als",
    "rank_als",
    "en_als",
    "score_popularidad",
    "rank_popularidad",
    "en_popularidad",
    "score_genero",
    "rank_genero",
    "en_genero",
    "n_interacciones_libro",
    "n_interacciones_usuario",
]


def generar_candidatos_con_features(
    usuarios: list,
    modelo_als,
    matriz_usuario_libro,
    fila_por_usuario: dict,
    libros_por_columna: list,
    stats_popularidad: pd.DataFrame,
    stats_por_genero: dict,
    genero_por_usuario: dict,
    libros_leidos: dict,
    n_interacciones_por_usuario: dict,
    n_por_fuente: int = 150,
) -> pd.DataFrame:
    """Arma, para cada usuario, la unión de candidatos de las tres fuentes
    (ALS, popularidad por género, popularidad global) con sus features.

    Reusa las estructuras que ya arman `fit_popularity` (`stats_popularidad`),
    `fit_popularity_por_genero` (`stats_por_genero`) y `genero_preferido_por_usuario`
    -- no duplica esa lógica. Cada candidato queda con `score_*`/`rank_*`
    (posición real dentro de esa fuente, no posición entre los candidatos
    finalmente elegidos) por fuente que lo propuso, y `en_*` (1/0) indicando
    qué fuentes lo propusieron. Un candidato que no vino de una fuente
    queda con score 0.0 y rank `n_por_fuente` (sentinel: "justo afuera de
    la ventana de esa fuente").

    Devuelve un DataFrame largo con columnas `id_lector`, `id_libro` +
    `FEATURES`.
    """
    n_por_libro = stats_popularidad.set_index("id_libro")["n"].to_dict()
    score_popularidad_por_libro = stats_popularidad.set_index("id_libro")["score"].to_dict()
    ranking_global_ids = stats_popularidad["id_libro"].tolist()
    rank_popularidad_por_libro = {libro: i for i, libro in enumerate(ranking_global_ids)}

    usuarios_con_als = [u for u in usuarios if u in fila_por_usuario]
    ids_items_als: dict = {}
    scores_als: dict = {}
    if usuarios_con_als:
        filas = np.array([fila_por_usuario[u] for u in usuarios_con_als])
        ids_items, scores = modelo_als.recommend(
            filas, matriz_usuario_libro[filas], N=n_por_fuente, filter_already_liked_items=True
        )
        for id_lector, fila_ids, fila_scores in zip(usuarios_con_als, ids_items, scores):
            ids_items_als[id_lector] = [libros_por_columna[idx] for idx in fila_ids]
            scores_als[id_lector] = list(fila_scores)

    filas_resultado = []
    for id_lector in usuarios:
        vistos = set(libros_leidos.get(id_lector, set()))
        candidatos: dict = {}

        for rank, (id_libro, score) in enumerate(
            zip(ids_items_als.get(id_lector, []), scores_als.get(id_lector, []))
        ):
            if id_libro in vistos:
                continue
            c = candidatos.setdefault(id_libro, {})
            c["score_als"] = float(score)
            c["rank_als"] = rank
            c["en_als"] = 1

        agregados = 0
        for id_libro in ranking_global_ids:
            if agregados >= n_por_fuente:
                break
            if id_libro in vistos:
                continue
            c = candidatos.setdefault(id_libro, {})
            c["score_popularidad"] = score_popularidad_por_libro[id_libro]
            c["rank_popularidad"] = rank_popularidad_por_libro[id_libro]
            c["en_popularidad"] = 1
            agregados += 1

        genero = genero_por_usuario.get(id_lector)
        if genero is not None and genero in stats_por_genero:
            tabla_genero = stats_por_genero[genero]
            agregados = 0
            for rank, (id_libro, score) in enumerate(
                zip(tabla_genero["id_libro"], tabla_genero["score"])
            ):
                if agregados >= n_por_fuente:
                    break
                if id_libro in vistos:
                    continue
                c = candidatos.setdefault(id_libro, {})
                c["score_genero"] = score
                c["rank_genero"] = rank
                c["en_genero"] = 1
                agregados += 1

        n_usuario = n_interacciones_por_usuario.get(id_lector, 0)
        for id_libro, f in candidatos.items():
            filas_resultado.append(
                {
                    "id_lector": id_lector,
                    "id_libro": id_libro,
                    "score_als": f.get("score_als", 0.0),
                    "rank_als": f.get("rank_als", n_por_fuente),
                    "en_als": f.get("en_als", 0),
                    "score_popularidad": f.get("score_popularidad", 0.0),
                    "rank_popularidad": f.get("rank_popularidad", n_por_fuente),
                    "en_popularidad": f.get("en_popularidad", 0),
                    "score_genero": f.get("score_genero", 0.0),
                    "rank_genero": f.get("rank_genero", n_por_fuente),
                    "en_genero": f.get("en_genero", 0),
                    "n_interacciones_libro": n_por_libro.get(id_libro, 0),
                    "n_interacciones_usuario": n_usuario,
                }
            )

    columnas = ["id_lector", "id_libro"] + FEATURES
    return pd.DataFrame(filas_resultado, columns=columnas)


def armar_dataset_entrenamiento(
    candidatos_df: pd.DataFrame,
    etiquetas_df: pd.DataFrame,
    n_por_fuente: int = 150,
) -> tuple[pd.DataFrame, pd.Series, list]:
    """Arma (X, y, group) para `lightgbm.LGBMRanker.fit` a partir de los
    candidatos generados y las etiquetas reales (el "próximo libro" de
    cada usuario en `train_ranker`, columnas `id_lector`/`id_libro`).

    Si el libro-etiqueta de un usuario no aparece entre sus candidatos
    generados (cobertura incompleta de las 3 fuentes), se lo inyecta
    igual con features "ausente" (mismo sentinel que usa
    `generar_candidatos_con_features`) -- si no, ese usuario no aporta
    ningún positivo y el ranker nunca aprendería de él. `group` es la
    cantidad de filas por usuario, en el mismo orden que `X`/`y`.
    """
    candidatos_por_usuario = {
        id_lector: grupo for id_lector, grupo in candidatos_df.groupby("id_lector", sort=False)
    }

    grupos_filas = []
    tamanos_grupo = []
    for _, fila_etiqueta in etiquetas_df.iterrows():
        id_lector = fila_etiqueta["id_lector"]
        libro_objetivo = fila_etiqueta["id_libro"]

        grupo = candidatos_por_usuario.get(
            id_lector, pd.DataFrame(columns=["id_lector", "id_libro"] + FEATURES)
        ).copy()

        if libro_objetivo not in set(grupo["id_libro"]):
            fila_faltante = dict.fromkeys(FEATURES, 0)
            fila_faltante["rank_als"] = n_por_fuente
            fila_faltante["rank_popularidad"] = n_por_fuente
            fila_faltante["rank_genero"] = n_por_fuente
            fila_faltante["id_lector"] = id_lector
            fila_faltante["id_libro"] = libro_objetivo
            grupo = pd.concat([grupo, pd.DataFrame([fila_faltante])], ignore_index=True)

        grupo = grupo.assign(y=(grupo["id_libro"] == libro_objetivo).astype(int))
        grupos_filas.append(grupo)
        tamanos_grupo.append(len(grupo))

    dataset = pd.concat(grupos_filas, ignore_index=True)
    return dataset[FEATURES], dataset["y"], tamanos_grupo


def fit_ranker(
    X: pd.DataFrame,
    y: pd.Series,
    group: list,
    X_eval: pd.DataFrame | None = None,
    y_eval: pd.Series | None = None,
    group_eval: list | None = None,
    **params,
) -> lgb.LGBMRanker:
    """Entrena un `LGBMRanker` (objetivo `lambdarank`) sobre el dataset
    armado por `armar_dataset_entrenamiento`.

    Hiperparámetros conservadores por default -- sin un sweep agresivo
    tipo optuna en esta primera versión, para no repetir el sobreajuste
    al proxy local que ya se vio con ALS (ver `experiments/bitacora.md`).
    Si se pasa un `X_eval`/`y_eval`/`group_eval`, se usa para early
    stopping (frena el boosting cuando deja de mejorar), no para buscar
    hiperparámetros.
    """
    defaults = dict(
        objective="lambdarank",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        random_state=42,
    )
    defaults.update(params)
    modelo = lgb.LGBMRanker(**defaults)

    fit_kwargs: dict = {}
    callbacks = []
    if X_eval is not None:
        fit_kwargs["eval_set"] = [(X_eval, y_eval)]
        fit_kwargs["eval_group"] = [group_eval]
        callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))

    modelo.fit(X, y, group=group, callbacks=callbacks or None, **fit_kwargs)
    return modelo


def recomendar_por_usuario(
    usuarios: list,
    modelo_ranker: lgb.LGBMRanker,
    candidatos_df: pd.DataFrame,
    ranking_global: list,
    libros_leidos: dict,
    k: int,
) -> dict:
    """Puntúa los candidatos de cada usuario con el ranker entrenado,
    ordena y devuelve el top-k. Fallback a `ranking_global` si a un
    usuario le faltan candidatos para completar k (mismo patrón que el
    resto de los modelos del proyecto)."""
    recomendaciones: dict = {}

    if len(candidatos_df):
        candidatos_df = candidatos_df.copy()
        candidatos_df["_score_ranker"] = modelo_ranker.predict(candidatos_df[FEATURES])
        for id_lector, grupo in candidatos_df.groupby("id_lector", sort=False):
            ordenado = grupo.sort_values("_score_ranker", ascending=False)
            recomendaciones[id_lector] = ordenado["id_libro"].tolist()

    for id_lector in usuarios:
        candidatos = recomendaciones.get(id_lector, [])
        if len(candidatos) < k:
            vistos = set(libros_leidos.get(id_lector, set())) | set(candidatos)
            extra = [libro for libro in ranking_global if libro not in vistos]
            candidatos = candidatos + extra[: k - len(candidatos)]
        recomendaciones[id_lector] = candidatos[:k]

    return recomendaciones
