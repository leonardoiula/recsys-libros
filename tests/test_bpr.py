"""Tests para la construcción de la matriz binaria de BPR.

`fit_bpr` en sí (entrenar un `BayesianPersonalizedRanking` real) no se
testea acá por la misma razón que ALS: es responsabilidad de `implicit`,
ya probada por esa librería. La lógica de armado de recomendaciones se
reusa de `als.recomendar_por_usuario` -- ya testeada en `tests/test_als.py`
con un stub que respeta la misma API (`.recommend(...)`) que expone
`BayesianPersonalizedRanking`, así que no hace falta duplicar esos tests
acá.
"""

import pandas as pd

from recsys.models.bpr import construir_matriz_binaria


def test_construir_matriz_binaria_ignora_el_rating():
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1", "u2"],
            "id_libro": ["a", "b", "a"],
            "rating": [10, 1, 5],  # BPR no debe usar esto como peso
        }
    )

    matriz, fila_por_usuario, libros_por_columna = construir_matriz_binaria(interacciones)

    assert matriz.shape == (2, 2)
    assert fila_por_usuario == {"u1": 0, "u2": 1}
    assert libros_por_columna == ["a", "b"]

    columna_a = libros_por_columna.index("a")
    columna_b = libros_por_columna.index("b")
    # todas las celdas con interaccion valen 1, sin importar el rating
    assert matriz[fila_por_usuario["u1"], columna_a] == 1
    assert matriz[fila_por_usuario["u1"], columna_b] == 1
    assert matriz[fila_por_usuario["u2"], columna_a] == 1


def test_construir_matriz_binaria_celdas_sin_interaccion_son_cero():
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1"],
            "id_libro": ["a"],
            "rating": [7],
        }
    )

    matriz, fila_por_usuario, libros_por_columna = construir_matriz_binaria(interacciones)

    assert matriz.shape == (1, 1)
    assert matriz.toarray().sum() == 1  # una sola celda con dato
