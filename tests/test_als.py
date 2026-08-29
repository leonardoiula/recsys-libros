"""Tests para el modelo ALS: construcción de la matriz usuario-libro y el
armado del ranking por usuario (con fallback a popularidad global).

`recomendar_por_usuario` se testea con un modelo ALS *stub* (no un
`AlternatingLeastSquares` real) porque lo que hay que verificar acá es la
lógica propia del proyecto -- traducción columna->id_libro, filtro contra
libros ya leídos, truncado a k, fallback a ranking_global -- no el
comportamiento interno de `implicit`, que ya viene testeado por esa
librería.
"""

import numpy as np
import pandas as pd

from recsys.models.als import construir_matriz_usuario_libro, recomendar_hibrido, recomendar_por_usuario


class _ModeloALSFalso:
    """Stub de AlternatingLeastSquares.recommend: ignora los pesos reales
    y devuelve, para cada fila pedida, un top-N fijo de índices de columna
    (los primeros N que no estén ya "vistos" según `matriz_usuario_libro`)."""

    def __init__(self, columnas_por_fila: dict):
        self._columnas_por_fila = columnas_por_fila

    def recommend(self, userids, user_items, N, filter_already_liked_items):
        assert filter_already_liked_items is True
        ids_items = np.array(
            [self._columnas_por_fila[uid][:N] for uid in userids]
        )
        scores = np.ones_like(ids_items, dtype=float)
        return ids_items, scores


def test_construir_matriz_usuario_libro():
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1", "u2"],
            "id_libro": ["a", "b", "a"],
            "rating": [8, 5, 3],
        }
    )

    matriz, fila_por_usuario, libros_por_columna = construir_matriz_usuario_libro(interacciones)

    assert matriz.shape == (2, 2)
    assert fila_por_usuario == {"u1": 0, "u2": 1}
    assert libros_por_columna == ["a", "b"]

    columna_a = libros_por_columna.index("a")
    columna_b = libros_por_columna.index("b")
    assert matriz[fila_por_usuario["u1"], columna_a] == 8
    assert matriz[fila_por_usuario["u1"], columna_b] == 5
    assert matriz[fila_por_usuario["u2"], columna_a] == 3


def test_recomendar_usa_al_modelo_cuando_alcanza():
    modelo = _ModeloALSFalso({0: [1, 2, 3]})
    matriz = np.zeros((1, 4))

    recomendaciones = recomendar_por_usuario(
        usuarios=["u1"],
        modelo=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u1": 0},
        libros_por_columna=["a", "b", "c", "d"],
        ranking_global=["x1", "x2"],
        libros_leidos={},
        k=2,
    )

    assert recomendaciones["u1"] == ["b", "c"]


def test_recomendar_filtra_libros_ya_leidos():
    modelo = _ModeloALSFalso({0: [0, 1]})
    matriz = np.zeros((1, 2))

    recomendaciones = recomendar_por_usuario(
        usuarios=["u1"],
        modelo=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u1": 0},
        libros_por_columna=["a", "b"],
        ranking_global=["x1"],
        libros_leidos={"u1": {"a"}},
        k=2,
    )

    # "a" viene del modelo pero ya está leído -> se filtra y se completa
    # con el fallback global.
    assert recomendaciones["u1"] == ["b", "x1"]


def test_recomendar_cae_a_global_para_usuario_sin_fila():
    modelo = _ModeloALSFalso({})
    matriz = np.zeros((0, 0))

    recomendaciones = recomendar_por_usuario(
        usuarios=["u_nuevo"],
        modelo=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={},
        libros_por_columna=[],
        ranking_global=["x1", "x2"],
        libros_leidos={},
        k=2,
    )

    assert recomendaciones["u_nuevo"] == ["x1", "x2"]


def test_hibrido_rutea_usuario_activo_a_als():
    modelo = _ModeloALSFalso({0: [0, 1]})
    matriz = np.zeros((1, 2))

    recomendaciones = recomendar_hibrido(
        usuarios=["u_activo"],
        n_train_por_usuario={"u_activo": 50},
        umbral=10,
        modelo=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u_activo": 0},
        libros_por_columna=["als1", "als2"],
        ranking_por_genero={"terror": pd.DataFrame({"id_libro": ["g1", "g2"]})},
        genero_por_usuario={"u_activo": "terror"},
        ranking_global=["x1"],
        libros_leidos={},
        k=2,
    )

    # >= umbral -> ALS, no la cadena de genero
    assert recomendaciones["u_activo"] == ["als1", "als2"]


def test_hibrido_rutea_usuario_liviano_a_genero():
    modelo = _ModeloALSFalso({0: [0, 1]})
    matriz = np.zeros((1, 2))

    recomendaciones = recomendar_hibrido(
        usuarios=["u_liviano"],
        n_train_por_usuario={"u_liviano": 3},
        umbral=10,
        modelo=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u_liviano": 0},
        libros_por_columna=["als1", "als2"],
        ranking_por_genero={"terror": pd.DataFrame({"id_libro": ["g1", "g2"]})},
        genero_por_usuario={"u_liviano": "terror"},
        ranking_global=["x1"],
        libros_leidos={},
        k=2,
    )

    # < umbral -> cadena de genero (via popularity_segmentada), no ALS
    assert recomendaciones["u_liviano"] == ["g1", "g2"]


def test_hibrido_usuario_sin_actividad_registrada_va_a_genero():
    modelo = _ModeloALSFalso({})
    matriz = np.zeros((0, 0))

    recomendaciones = recomendar_hibrido(
        usuarios=["u_frio"],
        n_train_por_usuario={},  # no aparece -> se trata como 0 interacciones
        umbral=10,
        modelo=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={},
        libros_por_columna=[],
        ranking_por_genero={},
        genero_por_usuario={},
        ranking_global=["x1", "x2"],
        libros_leidos={},
        k=2,
    )

    assert recomendaciones["u_frio"] == ["x1", "x2"]
