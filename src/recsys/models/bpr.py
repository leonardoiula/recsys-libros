"""Bayesian Personalized Ranking (BPR) sobre la matriz usuario-libro.

Alternativa a ALS: en vez de reconstruir la matriz de confianza (mínimos
cuadrados), BPR optimiza directamente el *orden relativo* entre pares de
ítems interactuados/no-interactuados (para cada usuario, un libro que
leyó debería puntuar más alto que uno que no leyó) usando descenso de
gradiente estocástico. Al optimizar orden en vez de reconstrucción, en
teoría se ajusta mejor a una métrica de ranking como NDCG@k.

BPR es un modelo de ranking *pairwise* sobre interactuado/no-interactuado
-- a diferencia de ALS acá no se usa el rating como peso, la matriz es
binaria (interactuó o no). `implicit.bpr.BayesianPersonalizedRanking`
expone la misma API que `AlternatingLeastSquares` (`.fit`, `.recommend`
con la misma firma), así que `als.recomendar_por_usuario` se reusa tal
cual para armar las recomendaciones -- confirmado probando la librería
instalada directamente, no hace falta duplicar esa lógica acá.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.bpr import BayesianPersonalizedRanking


def construir_matriz_binaria(interacciones: pd.DataFrame) -> tuple[sp.csr_matrix, dict, list]:
    """Arma la matriz sparse usuario x libro binaria (interactuó = 1).

    Mismo mapeo de índices que `als.construir_matriz_usuario_libro`, pero
    sin usar el rating como peso -- BPR es un ranking pairwise sobre
    interactuado/no-interactuado, no una reconstrucción ponderada.

    Devuelve (matriz, fila_por_usuario, libros_por_columna).
    """
    usuarios = interacciones["id_lector"].unique()
    libros = interacciones["id_libro"].unique()

    fila_por_usuario = {u: i for i, u in enumerate(usuarios)}
    columna_por_libro = {b: i for i, b in enumerate(libros)}

    filas = interacciones["id_lector"].map(fila_por_usuario).to_numpy()
    columnas = interacciones["id_libro"].map(columna_por_libro).to_numpy()
    pesos = np.ones(len(interacciones), dtype=np.float32)

    matriz = sp.csr_matrix((pesos, (filas, columnas)), shape=(len(usuarios), len(libros)))
    return matriz, fila_por_usuario, list(libros)


def fit_bpr(
    interacciones: pd.DataFrame,
    factors: int = 128,
    regularization: float = 0.01,
    learning_rate: float = 0.01,
    iterations: int = 100,
    seed: int = 42,
) -> tuple[BayesianPersonalizedRanking, sp.csr_matrix, dict, list]:
    """Entrena BPR sobre la matriz binaria usuario-libro.

    Hiperparámetros sweepeados con `scripts/tune_als.py` (BPR es
    optimización por SGD, converge distinto a ALS -- no asumir los
    mismos valores por defecto).

    Devuelve el modelo entrenado junto con la matriz usuario-libro y los
    mapeos de índice necesarios para pedir recomendaciones por usuario
    (con `als.recomendar_por_usuario`).
    """
    matriz, fila_por_usuario, libros_por_columna = construir_matriz_binaria(interacciones)

    modelo = BayesianPersonalizedRanking(
        factors=factors,
        regularization=regularization,
        learning_rate=learning_rate,
        iterations=iterations,
        random_state=seed,
    )
    modelo.fit(matriz)

    return modelo, matriz, fila_por_usuario, libros_por_columna
