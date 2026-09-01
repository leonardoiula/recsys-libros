"""Compara dos subconjuntos de `FEATURES` del ranker con un test PAREADO
por usuario, sobre el MISMO contexto/seed (mismo split, mismo fit de
ALS/popularidad/género) -- mucho más poder estadístico que comparar
promedios de NDCG@k entre seeds independientes (el criterio que venía
usando el proyecto).

Uso: uv run python scripts/comparar_features_pareado.py

Por qué existe: al reimplementar así la comparación de las 26 features
actuales contra las 23 previas (sin las 3 señales cruzadas lector↔libro
que esta sesión confirmó en Kaggle con +0.5%), el efecto resultó
estadísticamente indistinguible de ruido (bootstrap 95% CI de la
diferencia incluye el 0) -- pese a que esas 3 features sí habían
"pasado" el criterio de "casi positivo en los 3 seeds". Ver
`experiments/modelo_actual.md`, sección "Recomendación: ¿cambiar de
paradigma?", para el detalle completo y por qué el criterio anterior
(desvío entre 3 seeds) tiene ~5x menos poder que este test pareado.

Edita `FEATURES_A`/`FEATURES_B` de abajo para comparar otro par de
subconjuntos (ej. una feature nueva vs. sin ella, o dos variantes de
una misma idea). Con el mismo contexto ya armado, dos configuraciones
se comparan en ~1 minuto en vez de repetir dos corridas completas de
`scripts/evaluate_ranker.py` (~20-25 min cada una).

Ojo -- el ENTRENAMIENTO de LightGBM no es 100% determinista pese a
`random_state` fijo (el orden de agregación entre threads en modo
multi-thread introduce ruido de punto flotante que se acumula en 200
rondas de boosting): dos corridas de este mismo script con el mismo
`SEED`/`FEATURES_A`/`FEATURES_B` dieron diferencia pareada +0.00058 y
-0.00009 respectivamente -- distinto en signo, pero ambas veces muy
por debajo de 1 sigma (0.85 y 0.11 sigma). Es un eje de ruido más,
además del muestral -- no cambia la conclusión (el efecto sigue siendo
indistinguible de cero), pero no esperes el mismo número exacto entre
corridas.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_lectores, load_libros
from recsys.evaluation import ndcg_at_k
from recsys.models import ranker as R

K = 20
N_POR_FUENTE = 150
SEED = 42
N_BOOTSTRAP = 2000

# Comparación activa: las 32 features actuales (incluye la 5a fuente de
# candidatos por similitud de resumen) vs. esas mismas 32 sin las 3
# features de esa fuente -- para confirmar con rigor la 5a fuente (ver
# experiments/modelo_actual.md, "Recomendación: ¿cambiar de paradigma?").
# OJO: esto solo compara las FEATURES del ranker, no la generación de
# candidatos en sí -- la fuente de resumen igual aporta candidatos
# nuevos en ambos casos (A y B comparten el mismo contexto, incluida la
# unión de las 5 fuentes); lo que se aísla acá es si el ranker aprende
# algo de los 3 campos score/rank/en_resumen_candidato -- que se solapan
# con sim_resumen_historial, ya presente en ambos -- no si el recall
# extra ayuda (eso ya lo mide `recall_candidatos.py` + el CV de 3 seeds).
FEATURES_A = R.FEATURES
FEATURES_EXCLUIR_EN_B = [
    "score_resumen_candidato",
    "rank_resumen_candidato",
    "en_resumen_candidato",
]
FEATURES_B = [f for f in R.FEATURES if f not in FEATURES_EXCLUIR_EN_B]


def _ndcg_por_usuario(ctx: dict, features: list) -> dict:
    """Entrena un `LGBMRanker` sobre `features` (subconjunto de las
    columnas ya armadas en `ctx["X"]`) y devuelve NDCG@k por usuario de
    `ctx["usuarios_test"]` -- no el promedio, para poder parear después.
    """
    modelo = R.fit_ranker(ctx["X"][features], ctx["y"], ctx["group"])
    candidatos_por_usuario = {u: g for u, g in ctx["candidatos_test"].groupby("id_lector", sort=False)}
    relevantes_por_usuario = ctx["test_final"].groupby("id_lector")["id_libro"].agg(set).to_dict()
    libros_leidos = ctx["libros_leidos_hasta_ranker"]
    ranking_global = ctx["ranking_global"]

    resultado = {}
    for id_lector in ctx["usuarios_test"]:
        grupo = candidatos_por_usuario.get(id_lector)
        if grupo is None or len(grupo) == 0:
            recomendados = []
        else:
            scores = modelo.predict(grupo[features])
            recomendados = list(grupo["id_libro"].to_numpy()[np.argsort(-scores)][:K])
        if len(recomendados) < K:
            vistos = set(libros_leidos.get(id_lector, set())) | set(recomendados)
            extra = [libro for libro in ranking_global if libro not in vistos]
            recomendados = recomendados + extra[: K - len(recomendados)]
        resultado[id_lector] = ndcg_at_k(recomendados, relevantes_por_usuario.get(id_lector, set()), K)
    return resultado


def main() -> None:
    interacciones = load_interacciones()
    libros = load_libros()
    lectores = load_lectores()

    t0 = time.time()
    ctx = R.preparar_pipeline(interacciones, libros, lectores, SEED, n_por_fuente=N_POR_FUENTE, k=K)
    print(f"contexto listo en {time.time()-t0:.0f}s", flush=True)

    ndcg_a = _ndcg_por_usuario(ctx, FEATURES_A)
    ndcg_b = _ndcg_por_usuario(ctx, FEATURES_B)
    print(f"listo en {time.time()-t0:.0f}s", flush=True)

    usuarios = ctx["usuarios_test"]
    valores_a = np.array([ndcg_a[u] for u in usuarios])
    valores_b = np.array([ndcg_b[u] for u in usuarios])
    diferencia = valores_a - valores_b
    n = len(diferencia)

    print(f"\nn usuarios de test: {n}")
    print(f"NDCG@{K} FEATURES_A ({len(FEATURES_A)} features): {valores_a.mean():.6f}")
    print(f"NDCG@{K} FEATURES_B ({len(FEATURES_B)} features): {valores_b.mean():.6f}")

    print("\n--- comparación NO pareada (promedios independientes) ---")
    se_no_pareado = math.sqrt(valores_a.var(ddof=1) / n + valores_b.var(ddof=1) / n)
    sigma_no_pareado = diferencia.mean() / se_no_pareado if se_no_pareado else float("nan")
    print(f"diferencia de medias: {diferencia.mean():+.6f}   SE no pareado: {se_no_pareado:.6f}   -> {sigma_no_pareado:.2f} sigma")

    print("\n--- comparación PAREADA por usuario (recomendada) ---")
    se_pareado = diferencia.std(ddof=1) / math.sqrt(n)
    sigma_pareado = diferencia.mean() / se_pareado if se_pareado else float("nan")
    print(f"diferencia media pareada: {diferencia.mean():+.6f}   sd de la diferencia: {diferencia.std(ddof=1):.6f}   SE pareado: {se_pareado:.6f}")
    print(f"-> {sigma_pareado:.2f} sigma")
    print(
        f"usuarios donde cambia el NDCG: {(diferencia != 0).mean():.4f}"
        f"  (mejora {(diferencia > 0).sum()}, empeora {(diferencia < 0).sum()})"
    )

    rng = np.random.default_rng(0)
    bootstrap = np.array(
        [rng.choice(diferencia, size=n, replace=True).mean() for _ in range(N_BOOTSTRAP)]
    )
    print(
        f"bootstrap 95% CI de la diferencia: "
        f"[{np.percentile(bootstrap, 2.5):+.6f}, {np.percentile(bootstrap, 97.5):+.6f}]"
    )
    print(f"P(diferencia > 0) bootstrap: {(bootstrap > 0).mean():.4f}")
    print(f"\nganancia de poder: SE no pareado / SE pareado = {se_no_pareado/se_pareado:.1f}x")


if __name__ == "__main__":
    main()
