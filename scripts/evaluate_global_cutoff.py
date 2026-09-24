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

**Ventana rodante de cortes globales** (`CORTES_VENTANA`, mismo espíritu que
`N_CORTES_RANKER`/`_ensamblar_dataset_ventana_rodante` en `ranker.py`, pero con
fechas de calendario en vez de un corte relativo por usuario): cada ventana
`(cortes[i], cortes[i+1]]` fitea su PROPIA etapa 1 solo con datos `<= cortes[i]` y
etiqueta con la primera interacción de cada usuario en esa ventana. La última frontera
(`FECHA_CORTE_T2`) separa todo lo anterior de `test_final` (la primera interacción de
cada usuario posterior a T2 -- el hold-out real, elegido porque da una población de
test con perfil de actividad muy parecido al de `ejemplo.csv` de Kaggle sin haber
ajustado a mano, ver `estado_del_arte.md`). Antes de generar los candidatos de test, la
etapa 1 se REFITEA sobre todo `interacciones` con `fecha <= T2`, mismo criterio que
`submit.py` en producción.

**Diseño en dos fases** (2026-09-24, mismo patrón que `ranker.preparar_pipeline` /
`ranker.evaluar_con_params`): armar la ventana rodante completa (~7-8 min) es caro y
NO depende de los hiperparámetros de LightGBM -- separarlo de la parte barata
(fitear+evaluar el `LGBMRanker`, ~30-60s) es lo que hace viable un tuning real
(`scripts/tune_ranker_global.py`) sin repetir el armado en cada trial.

- `preparar_contexto_global(seed, fuentes_activas)` arma la ventana rodante +
  `candidatos_test`, cachea el resultado a DISCO (`data/cache/global_ctx_*.pkl`,
  mismo criterio de invalidación por hash de código que
  `ranker.preparar_pipeline_cacheado`: hash de ESTE archivo + `ranker.py`).
- `evaluar_con_params_global(contexto, features, lgbm_params)` fitea+evalúa sobre un
  contexto ya armado -- la parte que sí varía con `lgbm_params`.
- `construir_y_evaluar(...)` sigue existiendo como atajo de conveniencia (una sola
  config) para no romper `scripts/comparar_global_pareado.py` /
  `scripts/probar_blend_global.py`, que ya lo usaban.

`seed`, a diferencia del split por usuario, NO cambia qué datos caen en cada partición
(`CORTES_VENTANA` son fechas fijas) -- alimenta el `random_state` de ALS en cada
ventana (`fit_als`) y el desempate de `split_temporal_global` (fechas iguales dentro
de un usuario). Variar `seed` acá mide varianza de ENTRENAMIENTO (ALS +, si se pasa en
`lgbm_params`, LightGBM), no varianza de qué usuarios/eventos caen en train vs val --
distinto de lo que mide el CV de 3 seeds del split por usuario. Ver
`scripts/evaluate_ranker_global.py` (CV) y `experiments/estado_del_arte.md`.

Uso: uv run python scripts/evaluate_global_cutoff.py
"""

from __future__ import annotations

import hashlib
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import (
    canonicalizar_libros_duplicados,
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
"""Tamaño del pool sin padding (`recs_ranker_pool`/`recs_als_pool`) que se expone para
blend/fusión de rankings entre familias de modelos distintas
(`scripts/probar_blend_global.py`) -- más grande que `K` para que la fusión tenga
margen real, no solo reordenar la intersección de dos top-20."""
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

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
"""Mismo directorio que `ranker.CACHE_DIR` (gitignored) -- los contextos de este
script y los de `preparar_pipeline_cacheado` conviven ahí, prefijos de archivo
distintos (`global_ctx_*` vs `ranker_ctx_*`)."""

# Cache en memoria de proceso (no a disco): interacciones/libros/lectores no cambian
# entre llamadas de `preparar_contexto_global` dentro de una misma corrida (tuning,
# CV multi-seed) -- evita releer sqlite una vez por seed/trial.
_DATOS_CACHE: dict = {}


def _cargar_datos():
    if not _DATOS_CACHE:
        _DATOS_CACHE["interacciones"] = load_interacciones()
        _DATOS_CACHE["libros"] = load_libros()
        _DATOS_CACHE["lectores"] = load_lectores()
    return _DATOS_CACHE["interacciones"], _DATOS_CACHE["libros"], _DATOS_CACHE["lectores"]


def _fit_stage1(train, libros, lectores, fuentes_activas=None, seed: int = 42, normalizar_autor_editorial: bool = False):
    """Fitea ALS (BM25) + popularidad + género + features auxiliares sobre `train` --
    mismo bloque que arma `preparar_pipeline`/`submit.py` para un corte dado, con los
    mismos hiperparámetros de producción (n_por_fuente/n_por_autor/n_por_fuente_autor).
    `fuentes_activas` (default `None` = todas) se pasa tal cual a
    `generar_candidatos_con_features` vía `args_candidatos` -- ver `ranker.FUENTES_CANDIDATOS`.
    `seed` alimenta `fit_als(..., seed=seed)` -- ver docstring del módulo.
    `normalizar_autor_editorial` se pasa tal cual a `calcular_features_auxiliares`
    -- ver `preparar_contexto_global`/`ranker.calcular_features_auxiliares`."""
    stats_popularidad = fit_popularity(train)
    stats_por_genero = fit_popularity_por_genero(train, libros)
    genero_por_usuario = genero_preferido_por_usuario(train, libros)
    modelo_als, matriz, fila_por_usuario, libros_por_columna = fit_als(train, bm25=BM25_ALS, seed=seed)
    features_auxiliares = calcular_features_auxiliares(
        train,
        libros,
        lectores,
        matriz,
        fila_por_usuario,
        libros_por_columna,
        normalizar_autor_editorial=normalizar_autor_editorial,
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


def _hash_codigo() -> str:
    """Hash corto de ESTE archivo + `ranker.py` -- cualquier cambio en la lógica de
    armado de ventana/candidatos invalida el cache de `preparar_contexto_global`,
    mismo criterio que `ranker.preparar_pipeline_cacheado`."""
    ranker_path = Path(__file__).resolve().parents[1] / "src" / "recsys" / "models" / "ranker.py"
    contenido = Path(__file__).read_bytes() + ranker_path.read_bytes()
    return hashlib.sha256(contenido).hexdigest()[:12]


def preparar_contexto_global(
    seed: int = 42,
    fuentes_activas=None,
    cache_dir: str | Path | None = None,
    verbose: bool = True,
    sanear: bool = False,
) -> dict:
    """Arma la ventana rodante de cortes globales + `candidatos_test` -- la parte cara
    (~7-8 min) del pipeline, la que NO depende de los hiperparámetros de LightGBM.
    Cachea el resultado a disco (ver `_hash_codigo`) para no repetir el armado al
    tunear hiperparámetros o correr CV multi-seed sobre el mismo `seed`/`fuentes_activas`.

    `sanear` (default `False`, opt-in -- 2026-09-24, ver "Sanear los tres hallazgos
    de la Parte I" en `experiments/estado_del_arte.md`): si `True`, aplica
    `recsys.data.canonicalizar_libros_duplicados` sobre `interacciones`/`libros`
    ANTES de armar la ventana rodante (afecta a TODO lo que sigue: splits,
    popularidad, ALS, candidatos, features) y pasa `normalizar_autor_editorial=True`
    a `calcular_features_auxiliares` en cada `_fit_stage1` -- los dos hallazgos
    reales de saneamiento de catálogo (duplicados de libro, fragmentación de
    autor/editorial), aplicados juntos como un solo eje "datos crudos vs.
    saneados" sobre la MISMA arquitectura sin cambios, para aislar el efecto de
    limpiar el dato del de cualquier otro cambio. El tercer hallazgo (strings
    vacíos) no requiere fix: confirmado en 0% sobre título/género/autor.

    Barre del `cache_dir` los `global_ctx_*.pkl` con hash de código VIEJO en cada
    llamada, mismo criterio que `preparar_pipeline_cacheado` (si no, cada edición de
    este archivo o de `ranker.py` deja una copia de varios GB sin techo).
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    hash_codigo = _hash_codigo()

    for viejo in cache_dir.glob("global_ctx_*.pkl"):
        if not viejo.name.endswith(f"_{hash_codigo}.pkl"):
            try:
                viejo.unlink()
            except OSError:
                pass

    fuentes_label = "todas" if fuentes_activas is None else "+".join(sorted(fuentes_activas))
    nombre = f"global_ctx_seed{seed}_fuentes-{fuentes_label}_sanear-{sanear}_{hash_codigo}.pkl"
    ruta = cache_dir / nombre
    if ruta.exists():
        if verbose:
            print(f"contexto cacheado: {ruta.name}", flush=True)
        with open(ruta, "rb") as f:
            return pickle.load(f)

    t_inicio = time.time()
    interacciones, libros, lectores = _cargar_datos()
    if sanear:
        libros, interacciones = canonicalizar_libros_duplicados(libros, interacciones)
        if verbose:
            print(f"[{time.time()-t_inicio:6.1f}s] datos saneados: {len(libros)} libros, "
                  f"{len(interacciones)} interacciones", flush=True)

    hasta_t2, test_final = split_temporal_global(interacciones, FECHA_CORTE_T2, seed=seed)
    if verbose:
        print(f"test_final (primeras interacciones post {FECHA_CORTE_T2}): {len(test_final)} etiquetas", flush=True)

    X_partes: list = []
    y_partes: list = []
    group: list = []
    for i in range(len(CORTES_VENTANA) - 1):
        b_izq, b_der = CORTES_VENTANA[i], CORTES_VENTANA[i + 1]
        hasta_der, _ = split_temporal_global(interacciones, b_der, seed=seed)
        train_ventana, etiquetas_ventana = split_temporal_global(hasta_der, b_izq, seed=seed)

        t0 = time.time()
        libros_leidos_ventana = libros_leidos_por_usuario(train_ventana)
        args_candidatos, _ = _fit_stage1(
            train_ventana, libros, lectores, fuentes_activas=fuentes_activas, seed=seed, normalizar_autor_editorial=sanear
        )
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
    libros_leidos_hasta_t2 = libros_leidos_por_usuario(hasta_t2)
    args_candidatos_test, ranking_global_test = _fit_stage1(
        hasta_t2, libros, lectores, fuentes_activas=fuentes_activas, seed=seed, normalizar_autor_editorial=sanear
    )
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

    relevantes_por_usuario = test_final.groupby("id_lector")["id_libro"].agg(set).to_dict()
    ranking_global = ranking_global_test

    recs_pop = {
        u: [libro for libro in ranking_global if libro not in libros_leidos_hasta_t2.get(u, set())][:K]
        for u in usuarios_test
    }
    ndcg_pop = evaluar_ndcg_personalizado(test_final, recs_pop, K)

    recs_als = recomendar_als(
        usuarios=usuarios_test,
        modelo=args_candidatos_test["modelo_als"],
        matriz_usuario_libro=args_candidatos_test["matriz_usuario_libro"],
        fila_por_usuario=args_candidatos_test["fila_por_usuario"],
        libros_por_columna=args_candidatos_test["libros_por_columna"],
        ranking_global=ranking_global,
        libros_leidos=libros_leidos_hasta_t2,
        k=K,
    )
    ndcg_als = evaluar_ndcg_personalizado(test_final, recs_als, K)

    # Pool ALS más grande que K, para blend/fusión (scripts/probar_blend_global.py) --
    # no para la evaluación oficial de ALS-solo (eso es ndcg_als arriba).
    recs_als_pool = recomendar_als(
        usuarios=usuarios_test,
        modelo=args_candidatos_test["modelo_als"],
        matriz_usuario_libro=args_candidatos_test["matriz_usuario_libro"],
        fila_por_usuario=args_candidatos_test["fila_por_usuario"],
        libros_por_columna=args_candidatos_test["libros_por_columna"],
        ranking_global=ranking_global,
        libros_leidos=libros_leidos_hasta_t2,
        k=K_POOL,
    )

    if verbose:
        print(f"[{time.time()-t_inicio:6.1f}s] contexto completo (sin fitear el LGBMRanker final)", flush=True)

    contexto = {
        "X": X,
        "y": y,
        "group": group,
        "candidatos_test": candidatos_test,
        "test_final": test_final,
        "usuarios_test": usuarios_test,
        "libros_leidos_hasta_t2": libros_leidos_hasta_t2,
        "ranking_global": ranking_global,
        "relevantes_por_usuario": relevantes_por_usuario,
        "ndcg_pop": ndcg_pop,
        "ndcg_als": ndcg_als,
        "recs_als_pool": recs_als_pool,
        "n_train": len(X),
        "n_grupos": len(group),
        "n_test": len(usuarios_test),
    }
    with open(ruta, "wb") as f:
        pickle.dump(contexto, f, protocol=pickle.HIGHEST_PROTOCOL)
    return contexto


def evaluar_con_params_global(contexto: dict, features: list[str] | None = None, lgbm_params: dict | None = None) -> dict:
    """Fitea un `LGBMRanker` (rápido, ~30-60s) con `lgbm_params` sobre `contexto`
    (`preparar_contexto_global`) y evalúa NDCG@K en `test_final` -- pensada para
    llamarse muchas veces con distintos `lgbm_params`/`features` contra el mismo
    contexto, igual que `ranker.evaluar_con_params`.

    Devuelve `{"ndcg_ranker", "ndcg_por_usuario" (dict id_lector -> NDCG@k, para tests
    pareados), "recs_ranker_pool" (top-K_POOL sin padding, para blend), "modelo_ranker"}`.
    """
    features = FEATURES if features is None else features
    modelo_ranker = fit_ranker(
        contexto["X"][features], contexto["y"], contexto["group"], n_bag=N_BAG_RANKER, **(lgbm_params or {})
    )

    candidatos_test = contexto["candidatos_test"].copy()
    candidatos_test["_score"] = modelo_ranker.predict(candidatos_test[features])
    candidatos_por_usuario = {u: g for u, g in candidatos_test.groupby("id_lector", sort=False, observed=True)}
    libros_leidos_hasta_t2 = contexto["libros_leidos_hasta_t2"]
    ranking_global = contexto["ranking_global"]
    relevantes_por_usuario = contexto["relevantes_por_usuario"]

    ndcg_por_usuario: dict = {}
    recs_ranker: dict = {}
    recs_ranker_pool: dict = {}
    for id_lector in contexto["usuarios_test"]:
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

    ndcg_ranker = evaluar_ndcg_personalizado(contexto["test_final"], recs_ranker, K)

    return {
        "ndcg_ranker": ndcg_ranker,
        "ndcg_por_usuario": ndcg_por_usuario,
        "recs_ranker_pool": recs_ranker_pool,
        "modelo_ranker": modelo_ranker,
    }


def construir_y_evaluar(
    fuentes_activas=None,
    features: list[str] | None = None,
    seed: int = 42,
    lgbm_params: dict | None = None,
    verbose: bool = True,
    sanear: bool = False,
) -> dict:
    """Atajo de conveniencia para el caso de una sola config: `preparar_contexto_global`
    + `evaluar_con_params_global`, mismo patrón que `ranker.evaluar_pipeline`. Usada por
    `scripts/comparar_global_pareado.py`/`scripts/probar_blend_global.py`.

    `sanear` se pasa tal cual a `preparar_contexto_global` -- ver su docstring.

    Devuelve `{"ndcg_pop", "ndcg_als", "ndcg_ranker", "ndcg_por_usuario", "recs_ranker_pool",
    "recs_als_pool", "relevantes_por_usuario", "libros_leidos_hasta_t2", "ranking_global",
    "n_train", "n_grupos", "n_test"}`.
    """
    contexto = preparar_contexto_global(seed=seed, fuentes_activas=fuentes_activas, verbose=verbose, sanear=sanear)
    r = evaluar_con_params_global(contexto, features=features, lgbm_params=lgbm_params)
    if verbose:
        print(f"NDCG@{K} ranker: {r['ndcg_ranker']:.6f}", flush=True)
    return {
        "ndcg_pop": contexto["ndcg_pop"],
        "ndcg_als": contexto["ndcg_als"],
        "ndcg_ranker": r["ndcg_ranker"],
        "ndcg_por_usuario": r["ndcg_por_usuario"],
        "recs_ranker_pool": r["recs_ranker_pool"],
        "recs_als_pool": contexto["recs_als_pool"],
        "relevantes_por_usuario": contexto["relevantes_por_usuario"],
        "libros_leidos_hasta_t2": contexto["libros_leidos_hasta_t2"],
        "ranking_global": contexto["ranking_global"],
        "n_train": contexto["n_train"],
        "n_grupos": contexto["n_grupos"],
        "n_test": contexto["n_test"],
    }


def main() -> None:
    r = construir_y_evaluar()
    print(f"\n=== NDCG@{K} bajo corte temporal GLOBAL (ventanas={CORTES_VENTANA}) ===")
    print(f"popularidad global: {r['ndcg_pop']:.6f}")
    print(f"ALS solo:           {r['ndcg_als']:.6f}")
    print(f"ranker (2 etapas):  {r['ndcg_ranker']:.6f}  (entrenado con {r['n_grupos']} grupos/{r['n_train']} filas)")


if __name__ == "__main__":
    main()
