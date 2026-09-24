"""Evalúa el ranker bajo un split de corte temporal **GLOBAL** (un solo corte de
calendario para todo el dataset), en vez del leave-one-out **por usuario** que usa
`scripts/evaluate_ranker.py` (`split_train_val`).

Motivación: la brecha NDCG@20 local-vs-Kaggle del proyecto es grande incluso después
de reponderar por actividad (~0.107 local ponderado vs ~0.063 Kaggle real para la
misma config, ver `experiments/estado_del_arte.md`, sección "Validación con corte
temporal global"). `split_train_val` deja que ALS/popularidad/co-lectura se fiteen
sobre TODA la población sin ningún corte de fecha -- para un usuario mediano de la
validación local (cuyo propio corte cae ~2018), casi un tercio de TODO el dataset es
cronológicamente POSTERIOR a su punto de validación. Kaggle evalúa específicamente a
usuarios activos hasta el final del rango de fechas del dataset (mediana de última
interacción 2024-10-25 en `ejemplo.csv` vs 2018-01-01 en la población general), donde
esa fuga casi no existe. Este script mide el NDCG@20 bajo un protocolo que replica esa
asimetría: un corte de calendario único, con la etapa 1 fiteada solo con datos
anteriores al corte -- igual que un sistema prospectivo real.

Dos cortes (`recsys.data.split_temporal_global`, dos llamadas):
- `FECHA_CORTE_T1`: separa `train_candidatos_global` (fitea ALS/popularidad/género/
  co-lectura/TF-IDF para generar los candidatos de ENTRENAMIENTO del ranker) de
  `train_ranker_global` (la primera interacción de cada usuario en la ventana
  `(T1, T2]` -- la etiqueta que el `LGBMRanker` aprende a predecir).
- `FECHA_CORTE_T2`: separa todo lo anterior de `test_final_global` (la primera
  interacción de cada usuario posterior a T2 -- el hold-out real). Antes de generar
  sus candidatos, la etapa 1 se REFITEA sobre todo `interacciones` con
  `fecha <= T2` (`train_candidatos_global` + `train_ranker_global`), mismo criterio
  que usa `submit.py` en producción (refit sobre todos los datos disponibles antes de
  generar la entrega final) -- así esta medición es comparable a lo que Kaggle
  realmente evalúa, no solo a la etapa de entrenamiento.

`FECHA_CORTE_T2` se eligió explorando la fecha de "última interacción visible" de los
usuarios de `ejemplo.csv` contra la población general (ver estado_del_arte.md):
2024-07-01 da una población de test (811 usuarios, 45.0% en el bucket de actividad
100+) sorprendentemente parecida a la de Kaggle (832 usuarios, 46.9% en 100+) sin
haber ajustado a mano para que coincida -- otra señal de que el mecanismo identificado
es el correcto.

**Ventana rodante de cortes globales** (`CORTES_VENTANA`, mismo espíritu que
`N_CORTES_RANKER`/`_ensamblar_dataset_ventana_rodante` en `ranker.py`, pero con
fechas de calendario en vez de un corte relativo por usuario): un solo corte
`(T1=2022-01-01, T2)` daba apenas ~1.190-2.070 etiquetas para el `LGBMRanker` (contra
~7.900 de `split_train_val` a `n_cortes=1`) Y dejaba un gap de 2.5 años entre la
etapa 1 con la que el ranker aprende (fiteada solo hasta T1) y la etapa 1 "fresca"
con la que se lo evalúa (refiteada hasta T2) -- un desajuste de distribución entre
train y test que el split POR USUARIO no tiene (ahí train_candidatos y el refit final
difieren en un puñado de filas, nunca años). Con varios cortes intermedios entre 2022
y T2, cada ventana `(cortes[i], cortes[i+1]]` fitea su PROPIA etapa 1 solo con datos
`<= cortes[i]` y aporta como etiqueta la primera interacción de cada usuario en esa
ventana -- ~4.470 etiquetas en total (más del doble) y cada ejemplo de entrenamiento
queda a 3-12 meses de su propia etapa 1 en vez de a 2.5 años.

`construir_y_evaluar(fuentes_activas=None, features=None)` es la función reusable --
`main()` la llama una vez para el diagnóstico base (popularidad/ALS/ranker);
`scripts/comparar_global_pareado.py` la llama dos veces (A/B) para un test PAREADO
por usuario, igual que `comparar_generadores_pareado.py` pero bajo este protocolo sin
fuga temporal en vez del split por usuario.

Uso: uv run python scripts/evaluate_global_cutoff.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import (
    libros_leidos_por_usuario,
    load_interacciones,
    load_lectores,
    load_libros,
    split_temporal_global,
)
from recsys.evaluation import evaluar_ndcg_personalizado, ndcg_at_k
from recsys.models.als import fit_als
from recsys.models.als import recomendar_por_usuario as recomendar_als
from recsys.models.popularity import fit_popularity
from recsys.models.popularity_segmentada import fit_popularity_por_genero, genero_preferido_por_usuario
from recsys.models.ranker import (
    BM25_ALS,
    FEATURES,
    N_BAG_RANKER,
    armar_dataset_entrenamiento_por_lotes,
    calcular_features_auxiliares,
    fit_ranker,
    generar_candidatos_con_features_por_lotes,
)
from recsys.submit import N_POR_AUTOR_RANKER, N_POR_FUENTE_AUTOR_RANKER, N_POR_FUENTE_RANKER

K = 20
K_POOL = 100
"""Tamaño del pool sin padding (`recs_ranker_pool`/`recs_als_pool`, ver
`construir_y_evaluar`) que se expone para blend/fusión de rankings entre familias de
modelos distintas (`scripts/probar_blend_global.py`) -- más grande que `K` para que
la fusión tenga margen real, no solo reordenar la intersección de dos top-20."""
FECHA_CORTE_T2 = "01-07-2024"
CORTES_VENTANA = ["01-01-2022", "01-01-2023", "01-07-2023", "01-01-2024", "01-04-2024", FECHA_CORTE_T2]
"""Fronteras de la ventana rodante de cortes globales. La ventana i fitea su etapa 1
con datos `<= CORTES_VENTANA[i]` y etiqueta con la primera interacción de cada
usuario en `(CORTES_VENTANA[i], CORTES_VENTANA[i+1]]`. El primer corte (2022-01-01)
deja ~90% de las interacciones para la etapa 1 más antigua; los siguientes se
espacian 3-12 meses para no dejar ninguna etapa 1 demasiado desactualizada respecto
de las etiquetas que le toca predecir.

Probado 2026-09-24 con 10 cortes desde 2021-01-01 (~7.960 etiquetas, ~2x, sin CV/pareado,
un solo seed): NDCG **bajó** levemente (0.045499 vs 0.049061 con 5 cortes) en vez de
subir -- al revés de lo esperado por analogía con `N_CORTES_RANKER` (que sí mejora
monótono con más cortes en el split por usuario). Con n=811 y una sola corrida no
alcanza para separar señal real de ruido, pero alcanza para NO asumir que "más
densidad" ayuda acá igual que en el split por usuario -- los cortes del split por
usuario son siempre relativos al propio historial de cada usuario (nunca "viejos" en
términos absolutos); los cortes globales de 2021 sí lo son, y podrían aportar señal
menos transferible al período que predice `test_final_global`. Revertido a 5 cortes
(más barato, ya validado). Ver `experiments/log.csv` y `experiments/estado_del_arte.md`."""

# Cache en memoria de proceso (no a disco, a diferencia de preparar_pipeline_cacheado):
# interacciones/libros/lectores no cambian entre llamadas de construir_y_evaluar dentro
# de una misma corrida (comparar_global_pareado.py llama dos veces seguidas) -- evita
# releer sqlite dos veces.
_DATOS_CACHE: dict = {}


def _cargar_datos():
    if not _DATOS_CACHE:
        _DATOS_CACHE["interacciones"] = load_interacciones()
        _DATOS_CACHE["libros"] = load_libros()
        _DATOS_CACHE["lectores"] = load_lectores()
    return _DATOS_CACHE["interacciones"], _DATOS_CACHE["libros"], _DATOS_CACHE["lectores"]


def _fit_stage1(train, libros, lectores, fuentes_activas=None):
    """Fitea ALS (BM25) + popularidad + género + features auxiliares sobre `train` --
    mismo bloque que arma `preparar_pipeline`/`submit.py` para un corte dado, con los
    mismos hiperparámetros de producción (n_por_fuente/n_por_autor/n_por_fuente_autor).
    `fuentes_activas` (default `None` = todas) se pasa tal cual a
    `generar_candidatos_con_features` vía `args_candidatos` -- ver `ranker.FUENTES_CANDIDATOS`."""
    stats_popularidad = fit_popularity(train)
    stats_por_genero = fit_popularity_por_genero(train, libros)
    genero_por_usuario = genero_preferido_por_usuario(train, libros)
    modelo_als, matriz, fila_por_usuario, libros_por_columna = fit_als(train, bm25=BM25_ALS)
    features_auxiliares = calcular_features_auxiliares(
        train, libros, lectores, matriz, fila_por_usuario, libros_por_columna
    )
    args_candidatos = dict(
        modelo_als=modelo_als,
        matriz_usuario_libro=matriz,
        fila_por_usuario=fila_por_usuario,
        libros_por_columna=libros_por_columna,
        stats_popularidad=stats_popularidad,
        stats_por_genero=stats_por_genero,
        genero_por_usuario=genero_por_usuario,
        n_interacciones_por_usuario=train.groupby("id_lector").size().to_dict(),
        features_auxiliares=features_auxiliares,
        n_por_fuente=N_POR_FUENTE_RANKER,
        n_por_autor=N_POR_AUTOR_RANKER,
        n_por_fuente_autor=N_POR_FUENTE_AUTOR_RANKER,
        fuentes_activas=fuentes_activas,
    )
    ranking_global = stats_popularidad["id_libro"].tolist()
    return args_candidatos, ranking_global


def construir_y_evaluar(
    fuentes_activas=None,
    features: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Arma la ventana rodante de cortes globales, entrena el ranker y evalúa sobre
    `test_final_global` -- ver docstring del módulo.

    `fuentes_activas` (default `None` = las 6 fuentes actuales de `FUENTES_CANDIDATOS`)
    y `features` (default `None` = `FEATURES` completo) permiten ablaciones, igual que
    `comparar_generadores_pareado.py`/`comparar_features_pareado.py` pero bajo el
    protocolo de corte global en vez del split por usuario.

    Devuelve `{"ndcg_pop", "ndcg_als", "ndcg_ranker", "ndcg_por_usuario" (dict id_lector
    -> NDCG@k, para tests pareados), "n_train", "n_grupos", "n_test"}`.
    """
    features = FEATURES if features is None else features
    t_inicio = time.time()
    interacciones, libros, lectores = _cargar_datos()

    hasta_t2, test_final = split_temporal_global(interacciones, FECHA_CORTE_T2)
    if verbose:
        print(f"test_final (primeras interacciones post {FECHA_CORTE_T2}): {len(test_final)} etiquetas", flush=True)

    X_partes: list = []
    y_partes: list = []
    group: list = []
    for i in range(len(CORTES_VENTANA) - 1):
        b_izq, b_der = CORTES_VENTANA[i], CORTES_VENTANA[i + 1]
        hasta_der, _ = split_temporal_global(interacciones, b_der)
        train_ventana, etiquetas_ventana = split_temporal_global(hasta_der, b_izq)

        t0 = time.time()
        libros_leidos_ventana = libros_leidos_por_usuario(train_ventana)
        args_candidatos, _ = _fit_stage1(train_ventana, libros, lectores, fuentes_activas=fuentes_activas)
        usuarios_ventana = etiquetas_ventana["id_lector"].unique().tolist()
        X_i, y_i, group_i = armar_dataset_entrenamiento_por_lotes(
            usuarios_ventana,
            etiquetas_ventana[["id_lector", "id_libro"]],
            libros_leidos_ventana,
            args_candidatos,
            n_por_fuente=N_POR_FUENTE_RANKER,
            n_por_autor=N_POR_AUTOR_RANKER,
        )
        X_partes.append(X_i)
        y_partes.append(y_i)
        group.extend(group_i)
        if verbose:
            print(f"[{time.time()-t_inicio:6.1f}s] ventana ({b_izq}, {b_der}]: etapa1+candidatos en "
                  f"{time.time()-t0:.1f}s -- {len(etiquetas_ventana)} etiquetas, {len(X_i)} filas", flush=True)

    X = pd.concat(X_partes, ignore_index=True)
    y = pd.concat(y_partes, ignore_index=True)
    if verbose:
        print(f"[{time.time()-t_inicio:6.1f}s] dataset de entrenamiento total: {len(X)} filas, {len(group)} grupos", flush=True)

    t0 = time.time()
    modelo_ranker = fit_ranker(X[features], y, group, n_bag=N_BAG_RANKER)
    if verbose:
        print(f"[{time.time()-t_inicio:6.1f}s] LGBMRanker entrenado en {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    libros_leidos_hasta_t2 = libros_leidos_por_usuario(hasta_t2)
    args_candidatos_test, ranking_global_test = _fit_stage1(hasta_t2, libros, lectores, fuentes_activas=fuentes_activas)
    if verbose:
        print(f"[{time.time()-t_inicio:6.1f}s] refit de etapa 1 (hasta T2) en {time.time()-t0:.1f}s", flush=True)

    usuarios_test = test_final["id_lector"].unique().tolist()
    t0 = time.time()
    candidatos_test = generar_candidatos_con_features_por_lotes(
        usuarios_test, libros_leidos_hasta_t2, args_candidatos_test
    )
    if verbose:
        print(f"[{time.time()-t_inicio:6.1f}s] candidatos de test armados en {time.time()-t0:.1f}s "
              f"({len(candidatos_test)} filas, {len(usuarios_test)} usuarios)", flush=True)

    candidatos_test = candidatos_test.copy()
    candidatos_test["_score"] = modelo_ranker.predict(candidatos_test[features])
    relevantes_por_usuario = test_final.groupby("id_lector")["id_libro"].agg(set).to_dict()
    candidatos_por_usuario = {u: g for u, g in candidatos_test.groupby("id_lector", sort=False, observed=True)}
    ranking_global = ranking_global_test

    ndcg_por_usuario: dict = {}
    recs_ranker: dict = {}
    recs_ranker_pool: dict = {}
    for id_lector in usuarios_test:
        grupo = candidatos_por_usuario.get(id_lector)
        if grupo is None or len(grupo) == 0:
            pool = []
        else:
            pool = list(grupo["id_libro"].to_numpy()[(-grupo["_score"].to_numpy()).argsort()][:K_POOL])
        recs_ranker_pool[id_lector] = pool  # SIN padding de popularidad -- ranking crudo del modelo, para fusión

        recomendados = pool[:K]
        if len(recomendados) < K:
            vistos = set(libros_leidos_hasta_t2.get(id_lector, set())) | set(recomendados)
            extra = [libro for libro in ranking_global if libro not in vistos]
            recomendados = recomendados + extra[: K - len(recomendados)]
        recs_ranker[id_lector] = recomendados
        ndcg_por_usuario[id_lector] = ndcg_at_k(recomendados, relevantes_por_usuario.get(id_lector, set()), K)

    ndcg_ranker = evaluar_ndcg_personalizado(test_final, recs_ranker, K)

    recs_pop = {
        u: [libro for libro in ranking_global_test if libro not in libros_leidos_hasta_t2.get(u, set())][:K]
        for u in usuarios_test
    }
    ndcg_pop = evaluar_ndcg_personalizado(test_final, recs_pop, K)

    recs_als = recomendar_als(
        usuarios=usuarios_test,
        modelo=args_candidatos_test["modelo_als"],
        matriz_usuario_libro=args_candidatos_test["matriz_usuario_libro"],
        fila_por_usuario=args_candidatos_test["fila_por_usuario"],
        libros_por_columna=args_candidatos_test["libros_por_columna"],
        ranking_global=ranking_global_test,
        libros_leidos=libros_leidos_hasta_t2,
        k=K,
    )
    ndcg_als = evaluar_ndcg_personalizado(test_final, recs_als, K)

    # Pool ALS más grande que K, SIN padding de popularidad (mismo criterio que
    # recs_ranker_pool) -- para blend/fusión de rankings (scripts/probar_blend_global.py),
    # no para la evaluación oficial de ALS-solo (eso sigue siendo recs_als/ndcg_als arriba).
    recs_als_pool_con_padding = recomendar_als(
        usuarios=usuarios_test,
        modelo=args_candidatos_test["modelo_als"],
        matriz_usuario_libro=args_candidatos_test["matriz_usuario_libro"],
        fila_por_usuario=args_candidatos_test["fila_por_usuario"],
        libros_por_columna=args_candidatos_test["libros_por_columna"],
        ranking_global=ranking_global_test,
        libros_leidos=libros_leidos_hasta_t2,
        k=K_POOL,
    )

    if verbose:
        print(f"tiempo total: {time.time()-t_inicio:.1f}s", flush=True)

    return {
        "ndcg_pop": ndcg_pop,
        "ndcg_als": ndcg_als,
        "ndcg_ranker": ndcg_ranker,
        "ndcg_por_usuario": ndcg_por_usuario,
        "recs_ranker_pool": recs_ranker_pool,
        "recs_als_pool": recs_als_pool_con_padding,
        "relevantes_por_usuario": relevantes_por_usuario,
        "libros_leidos_hasta_t2": libros_leidos_hasta_t2,
        "ranking_global": ranking_global,
        "n_train": len(X),
        "n_grupos": len(group),
        "n_test": len(usuarios_test),
    }


def main() -> None:
    r = construir_y_evaluar()
    print(f"\n=== NDCG@{K} bajo corte temporal GLOBAL (ventanas={CORTES_VENTANA}) ===")
    print(f"popularidad global: {r['ndcg_pop']:.6f}")
    print(f"ALS solo:           {r['ndcg_als']:.6f}")
    print(f"ranker (2 etapas):  {r['ndcg_ranker']:.6f}  (entrenado con {r['n_grupos']} grupos/{r['n_train']} filas)")


if __name__ == "__main__":
    main()
