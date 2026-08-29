"""Filtrado colaborativo con ALS (Alternating Least Squares) sobre la matriz
usuario-libro.

`implicit.als.AlternatingLeastSquares` está pensado para feedback
implícito (una matriz de "confianza", no de rating explícito). Acá el
dato real es un rating explícito de 1 a 10 (siempre positivo, ver EDA):
en vez de binarizar "leyó o no leyó" y perder la señal de intensidad, se
arma la matriz sparse usuario x libro con `confianza = 1 + alpha * rating`
(`alpha=None` usa el rating crudo tal cual) — la fórmula estándar de
Hu/Koren/Volinsky para feedback implícito ponderado, con `alpha` como
hiperparámetro que controla cuánta más confianza da cada punto de rating
(en vez de una relación 1:1 fija con el rating crudo). Las celdas sin
interacción quedan en 0, consistente con lo que espera `implicit`.

**Historial de hiperparámetros, confirmado con Kaggle real (no solo
NDCG local) -- ver `experiments/bitacora.md`:**
- `factors=128, regularization=0.1`, rating crudo (`alpha=None`):
  **0.03864 en Kaggle, el mejor confirmado hasta ahora.** Es el default
  actual.
- `factors=256, regularization=0.128, alpha=4.718` (`1+alpha*rating`),
  elegido por una búsqueda con `optuna` (30 trials) sobre un único
  split local: mejoraba el NDCG local +11.5%, pero dio **0.03341 en
  Kaggle, peor** que la config de arriba -- sobreajuste al split fijo
  usado en el sweep. Se dejó de usar como default por esto.

Para usuarios sin ninguna fila en la matriz (cold start real, sin
historial) se cae a `ranking_global` de popularidad, igual que el
fallback final de v1. Además, `recomendar_hibrido` rutea a los usuarios
con poca actividad hacia la popularidad por género de v1 en vez de
confiar en un embedding de ALS mal condicionado por pocos datos (ver su
docstring).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from recsys.models.popularity_segmentada import recomendar_por_usuario as _recomendar_genero_global


def construir_matriz_usuario_libro(
    interacciones: pd.DataFrame, alpha: float | None = None
) -> tuple[sp.csr_matrix, dict, list]:
    """Arma la matriz sparse usuario x libro pesada por confianza implícita.

    `confianza = 1 + alpha * rating` (Hu/Koren/Volinsky) si `alpha` es un
    número; `confianza = rating` crudo si `alpha=None` (default -- es la
    config con mejor score confirmado en Kaggle, ver `fit_als`). `alpha`
    controla cuánta más confianza da cada punto de rating, independiente
    de la escala del rating en sí; se sweepeó junto con
    `factors`/`regularization` en `scripts/tune_als.py`, pero esa
    búsqueda resultó sobreajustada al split usado (ver `fit_als`).

    Devuelve (matriz, fila_por_usuario, libros_por_columna): `fila_por_usuario`
    mapea id_lector -> fila de la matriz, y `libros_por_columna` es la
    lista de id_libro en el orden de las columnas (índice de columna ->
    id_libro).
    """
    usuarios = interacciones["id_lector"].unique()
    libros = interacciones["id_libro"].unique()

    fila_por_usuario = {u: i for i, u in enumerate(usuarios)}
    columna_por_libro = {b: i for i, b in enumerate(libros)}

    filas = interacciones["id_lector"].map(fila_por_usuario).to_numpy()
    columnas = interacciones["id_libro"].map(columna_por_libro).to_numpy()
    if alpha is None:
        pesos = interacciones["rating"].to_numpy(dtype=np.float32)
    else:
        pesos = (1.0 + alpha * interacciones["rating"]).to_numpy(dtype=np.float32)

    matriz = sp.csr_matrix((pesos, (filas, columnas)), shape=(len(usuarios), len(libros)))
    return matriz, fila_por_usuario, list(libros)


def fit_als(
    interacciones: pd.DataFrame,
    factors: int = 128,
    regularization: float = 0.1,
    iterations: int = 20,
    alpha: float | None = None,
    seed: int = 42,
) -> tuple[AlternatingLeastSquares, sp.csr_matrix, dict, list]:
    """Entrena ALS sobre la matriz usuario-libro pesada por confianza implícita.

    Default = `factors=128, regularization=0.1, alpha=None` (rating
    crudo como confianza): es la config con **mejor score confirmado en
    Kaggle** (0.03864) de todas las probadas hasta ahora.

    Se probó (y se descartó) `factors=256, regularization=0.128,
    alpha=4.718` -- encontrada con `optuna` (30 trials) sobre el split
    temporal corregido (`n_val=1`, `seed=42`, ver `scripts/tune_als.py`):
    mejoraba el NDCG@20 **local** +11.5%, pero dio **0.03341 en Kaggle
    real, peor** que el default actual. Es sobreajuste al único split
    usado en la búsqueda, no una mejora genuina -- ver
    `experiments/bitacora.md`, sección "Regresión en Kaggle". Sirve como
    lección para cualquier tuneo futuro de este módulo: validar siempre
    con varios seeds (`evaluation.evaluar_multisplit`), nunca un solo
    split, antes de confiar en una mejora local.

    Devuelve el modelo entrenado junto con la matriz usuario-libro y los
    mapeos de índice necesarios para pedir recomendaciones por usuario.
    """
    matriz, fila_por_usuario, libros_por_columna = construir_matriz_usuario_libro(interacciones, alpha=alpha)

    modelo = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=seed,
    )
    modelo.fit(matriz)

    return modelo, matriz, fila_por_usuario, libros_por_columna


def recomendar_por_usuario(
    usuarios: list,
    modelo: AlternatingLeastSquares,
    matriz_usuario_libro: sp.csr_matrix,
    fila_por_usuario: dict,
    libros_por_columna: list,
    ranking_global: list,
    libros_leidos: dict,
    k: int,
) -> dict:
    """Arma el top-k de ALS por usuario, con fallback a popularidad global.

    `modelo.recommend` puntúa densamente todo el catálogo (no solo lo
    observado) y ya excluye, vía `filter_already_liked_items=True`, los
    libros con peso > 0 en la fila de `matriz_usuario_libro` de ese
    usuario -- por eso alcanza con pedir N=k. Igual se re-filtra
    explícitamente contra `libros_leidos` para mantener la misma garantía
    que el resto de los modelos del proyecto (por si esa matriz no
    coincide exactamente con `libros_leidos`, p.ej. en evaluación local
    donde la matriz sale de train) y porque, si el catálogo activo de un
    usuario fuera tan chico que no alcanzan k libros no vistos (no pasa
    con este dataset: cada usuario tiene miles de libros sin leer), la
    implementación de `implicit` no rellena esos huecos con -1 sino
    repitiendo ids de libros ya vistos (con score inválido) -- por eso el
    filtro contra `libros_leidos` es la protección real, no un chequeo de
    índice.

    Usuarios sin fila en la matriz (sin ningún dato de entrenamiento) se
    completan enteramente con `ranking_global`.
    """
    recomendaciones: dict = {}

    usuarios_conocidos = [u for u in usuarios if u in fila_por_usuario]
    if usuarios_conocidos:
        filas = np.array([fila_por_usuario[u] for u in usuarios_conocidos])
        ids_items, _scores = modelo.recommend(
            filas,
            matriz_usuario_libro[filas],
            N=k,
            filter_already_liked_items=True,
        )

        for id_lector, fila_ids in zip(usuarios_conocidos, ids_items):
            leidos = libros_leidos.get(id_lector, set())
            candidatos = [
                libros_por_columna[idx]
                for idx in fila_ids
                if libros_por_columna[idx] not in leidos
            ]
            recomendaciones[id_lector] = candidatos

    for id_lector in usuarios:
        candidatos = recomendaciones.get(id_lector, [])
        if len(candidatos) < k:
            vistos = set(libros_leidos.get(id_lector, set())) | set(candidatos)
            extra = [libro for libro in ranking_global if libro not in vistos]
            candidatos = candidatos + extra[: k - len(candidatos)]
        recomendaciones[id_lector] = candidatos[:k]

    return recomendaciones


def recomendar_hibrido(
    usuarios: list,
    n_train_por_usuario: dict,
    umbral: int,
    modelo: AlternatingLeastSquares,
    matriz_usuario_libro: sp.csr_matrix,
    fila_por_usuario: dict,
    libros_por_columna: list,
    ranking_por_genero: dict,
    genero_por_usuario: dict,
    ranking_global: list,
    libros_leidos: dict,
    k: int,
) -> dict:
    """Rutea cada usuario a ALS o a popularidad por género según su actividad.

    **No se usa en producción (`submit.py` sigue con ALS puro).** Se
    construyó para probar la hipótesis de que el embedding de ALS es poco
    confiable para usuarios con poca actividad (con pocas interacciones
    queda mal condicionado, alta varianza) y que convendría, para esos
    casos, caer a la cadena de popularidad por género de v1 en vez de
    confiar en ALS. Medido bajo el split corregido (temporal, n_val=1):
    **ALS le gana a género en todos los buckets de actividad probados**,
    incluidos usuarios con una sola interacción en train (NDCG@20:
    0.086 ALS vs 0.016 género con n=1; la brecha se mantiene o crece con
    más actividad). No hay ningún `umbral` donde rutear a género mejore
    el resultado -- ver `experiments/bitacora.md` para la tabla completa.
    Se deja esta función implementada y testeada por si el escenario
    cambia (ej. un dataset con usuarios de mucha menos actividad
    promedio, donde el argumento original sí podría sostenerse), pero hoy
    usarla con cualquier `umbral > 0` empeora el modelo.

    Cuando se usa, para usuarios con menos de `umbral` interacciones en
    train (incluidos los que no tienen ninguna) cae a la cadena de
    popularidad por género de v1, reusando
    `popularity_segmentada.recomendar_por_usuario` tal cual pero con
    `ranking_por_franja={}` y `franja_por_usuario={}` -- se confirmó que
    franja de nacimiento aporta candidatos a ~0.03% de los usuarios en la
    práctica (ver `experiments/decisiones.md`), así que no vale la
    complejidad de llevarla a este ruteo; con esos diccionarios vacíos la
    cadena colapsa directo a género -> global.
    """
    calidos = [u for u in usuarios if n_train_por_usuario.get(u, 0) >= umbral]
    livianos = [u for u in usuarios if n_train_por_usuario.get(u, 0) < umbral]

    recomendaciones: dict = {}

    if calidos:
        recomendaciones.update(
            recomendar_por_usuario(
                usuarios=calidos,
                modelo=modelo,
                matriz_usuario_libro=matriz_usuario_libro,
                fila_por_usuario=fila_por_usuario,
                libros_por_columna=libros_por_columna,
                ranking_global=ranking_global,
                libros_leidos=libros_leidos,
                k=k,
            )
        )

    if livianos:
        recomendaciones.update(
            _recomendar_genero_global(
                usuarios=livianos,
                ranking_por_genero=ranking_por_genero,
                genero_por_usuario=genero_por_usuario,
                ranking_por_franja={},
                franja_por_usuario={},
                ranking_global=ranking_global,
                libros_leidos=libros_leidos,
                k=k,
            )
        )

    return recomendaciones
