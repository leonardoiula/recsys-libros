"""Filtrado colaborativo con ALS (Alternating Least Squares) sobre la matriz
usuario-libro.

`implicit.als.AlternatingLeastSquares` está pensado para feedback
implícito (una matriz de "confianza", no de rating explícito). Acá el
dato real es un rating explícito de 1 a 10 (siempre positivo, ver EDA):
en vez de binarizar "leyó o no leyó" y perder la señal de intensidad, se
arma la matriz sparse usuario x libro usando el rating directamente como
peso/confianza de esa celda — patrón estándar para aplicar ALS de
feedback implícito sobre datos que en el fondo son explícitos pero muy
sparse. Las celdas sin interacción quedan en 0, consistente con lo que
espera `implicit`.

Para usuarios sin ninguna fila en la matriz (cold start real, sin
historial) se cae a `ranking_global` de popularidad, igual que el
fallback final de v1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


def construir_matriz_usuario_libro(interacciones: pd.DataFrame) -> tuple[sp.csr_matrix, dict, list]:
    """Arma la matriz sparse usuario x libro pesada por rating.

    Devuelve (matriz, fila_por_usuario, libros_por_columna): `matriz` usa
    el rating como confianza implícita (siempre > 0 en estos datos),
    `fila_por_usuario` mapea id_lector -> fila de la matriz, y
    `libros_por_columna` es la lista de id_libro en el orden de las
    columnas (índice de columna -> id_libro).
    """
    usuarios = interacciones["id_lector"].unique()
    libros = interacciones["id_libro"].unique()

    fila_por_usuario = {u: i for i, u in enumerate(usuarios)}
    columna_por_libro = {b: i for i, b in enumerate(libros)}

    filas = interacciones["id_lector"].map(fila_por_usuario).to_numpy()
    columnas = interacciones["id_libro"].map(columna_por_libro).to_numpy()
    pesos = interacciones["rating"].to_numpy(dtype=np.float32)

    matriz = sp.csr_matrix((pesos, (filas, columnas)), shape=(len(usuarios), len(libros)))
    return matriz, fila_por_usuario, list(libros)


def fit_als(
    interacciones: pd.DataFrame,
    factors: int = 128,
    regularization: float = 0.1,
    iterations: int = 20,
    seed: int = 42,
) -> tuple[AlternatingLeastSquares, sp.csr_matrix, dict, list]:
    """Entrena ALS sobre la matriz usuario-libro pesada por rating.

    Hiperparámetros elegidos con un sweep local sobre el split
    frac_val=0.2/seed=42 (ver experiments/bitacora.md, sección v2).

    Devuelve el modelo entrenado junto con la matriz usuario-libro y los
    mapeos de índice necesarios para pedir recomendaciones por usuario.
    """
    matriz, fila_por_usuario, libros_por_columna = construir_matriz_usuario_libro(interacciones)

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
