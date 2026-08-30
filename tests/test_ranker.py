"""Tests para la generación de candidatos/features del ranker y el armado
del dataset de entrenamiento (group/inyección de positivos faltantes).

No se testea `fit_ranker` en sí (entrenar un `LGBMRanker` real): es
responsabilidad de `lightgbm`, ya probada por esa librería -- mismo
criterio que con `implicit` en `test_als.py`/`test_bpr.py`.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys.models.popularity_segmentada import MACRO_GENERO_DEFAULT
from recsys.models.ranker import (
    FEATURES,
    SENTINEL_DIAS_DESCONOCIDO,
    armar_dataset_entrenamiento,
    calcular_features_auxiliares,
    generar_candidatos_con_features,
)


class _ModeloALSFalso:
    def __init__(self, columnas_y_scores_por_fila: dict):
        self._datos = columnas_y_scores_por_fila

    def recommend(self, userids, user_items, N, filter_already_liked_items):
        assert filter_already_liked_items is True
        ids_items = np.array([self._datos[uid][0][:N] for uid in userids])
        scores = np.array([self._datos[uid][1][:N] for uid in userids])
        return ids_items, scores


def _stats_popularidad(ids: list, scores: list) -> pd.DataFrame:
    return pd.DataFrame({"id_libro": ids, "n": [10] * len(ids), "avg_rating": [7.0] * len(ids), "score": scores})


def _features_auxiliares_vacias() -> dict:
    """Bundle vacío de `calcular_features_auxiliares`, para tests que no
    ejercitan las features de autor/editorial/año/género/recencia/
    co-lectura/resumen."""
    return {
        "autor_por_libro": {},
        "anio_edicion_por_libro": {},
        "n_libros_autor_leidos_por_usuario": {},
        "editorial_por_libro": {},
        "n_libros_editorial_leidos_por_usuario": {},
        "n_libros_por_editorial": {},
        "anio_edicion_promedio_por_usuario": {},
        "n_generos_distintos_por_usuario": {},
        "dias_desde_ultima_interaccion_por_usuario": {},
        "cooc": None,
        "columna_por_libro": {},
        "tfidf_norm": None,
        "fila_por_libro_texto": {},
        "perfil_usuario_norm": None,
        "genero_macro_por_libro": {},
        "score_por_libro_genero_macro": {},
        "frecuencia_genero_macro_por_usuario": {},
    }


def test_candidato_de_una_sola_fuente_tiene_sentinel_en_las_otras():
    modelo = _ModeloALSFalso({0: ([0, 1], [0.9, 0.5])})  # columnas -> libros "a","b"
    matriz = np.zeros((1, 2))
    stats_pop = _stats_popularidad(["x1"], [5.0])  # no incluye "a"/"b"

    candidatos = generar_candidatos_con_features(
        usuarios=["u1"],
        modelo_als=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u1": 0},
        libros_por_columna=["a", "b"],
        stats_popularidad=stats_pop,
        stats_por_genero={},
        genero_por_usuario={},
        libros_leidos={},
        n_interacciones_por_usuario={"u1": 3},
        features_auxiliares=_features_auxiliares_vacias(),
        n_por_fuente=150,
    )

    fila_a = candidatos[candidatos["id_libro"] == "a"].iloc[0]
    assert fila_a["en_als"] == 1
    assert fila_a["score_als"] == 0.9
    assert fila_a["en_popularidad"] == 0
    assert fila_a["rank_popularidad"] == 150  # sentinel = n_por_fuente
    assert fila_a["n_interacciones_usuario"] == 3


def test_candidato_de_varias_fuentes_se_une_en_una_sola_fila():
    modelo = _ModeloALSFalso({0: ([0], [0.7])})
    matriz = np.zeros((1, 1))
    stats_pop = _stats_popularidad(["a"], [5.0])  # "a" tambien es columna 0

    candidatos = generar_candidatos_con_features(
        usuarios=["u1"],
        modelo_als=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u1": 0},
        libros_por_columna=["a"],
        stats_popularidad=stats_pop,
        stats_por_genero={},
        genero_por_usuario={},
        libros_leidos={},
        n_interacciones_por_usuario={},
        features_auxiliares=_features_auxiliares_vacias(),
        n_por_fuente=150,
    )

    assert len(candidatos) == 1  # una sola fila para "a", no duplicada
    fila = candidatos.iloc[0]
    assert fila["en_als"] == 1 and fila["en_popularidad"] == 1


def test_libros_ya_leidos_se_excluyen_de_todas_las_fuentes():
    modelo = _ModeloALSFalso({0: ([0, 1], [0.9, 0.5])})
    matriz = np.zeros((1, 2))
    stats_pop = _stats_popularidad(["a", "b"], [9.0, 5.0])

    candidatos = generar_candidatos_con_features(
        usuarios=["u1"],
        modelo_als=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u1": 0},
        libros_por_columna=["a", "b"],
        stats_popularidad=stats_pop,
        stats_por_genero={},
        genero_por_usuario={},
        libros_leidos={"u1": {"a"}},
        n_interacciones_por_usuario={},
        features_auxiliares=_features_auxiliares_vacias(),
        n_por_fuente=150,
    )

    assert "a" not in set(candidatos["id_libro"])
    assert "b" in set(candidatos["id_libro"])


def test_calcular_features_auxiliares():
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1", "u1"],
            "id_libro": ["a", "b", "c"],
            "fecha": ["01-01-2020", "01-01-2021", "01-01-2022"],
            "rating": [8, 8, 8],
        }
    )
    libros = pd.DataFrame(
        {
            "id_libro": ["a", "b", "c", "d"],
            "autor": ["KING, STEPHEN", "KING, STEPHEN", "OTRO", None],
            "genero": ["Terror", "terror", "Novela negra", "Ensayo"],
            "anio_edicion": ["2000", "2010", "2020", "1990"],
            "editorial": ["PLANETA", "PLANETA", "SUDAMERICANA", None],
            "resumen": [
                "dragones y magia en un reino lejano",
                "mas dragones y aventuras de magia",
                "un asesinato en la gran ciudad",
                None,
            ],
        }
    )
    # u1 con fila 0, leyo a/b/c (columnas 0/1/2); "d" (columna 3) sin leer
    matriz = np.array([[1, 1, 1, 0]])
    fila_por_usuario = {"u1": 0}
    libros_por_columna = ["a", "b", "c", "d"]

    aux = calcular_features_auxiliares(interacciones, libros, matriz, fila_por_usuario, libros_por_columna)

    # u1 leyo 2 libros de "KING, STEPHEN" (a, b) y 1 de "OTRO" (c)
    assert aux["n_libros_autor_leidos_por_usuario"]["u1"]["KING, STEPHEN"] == 2
    assert aux["n_libros_autor_leidos_por_usuario"]["u1"]["OTRO"] == 1
    # mismo patron para editorial: 2 de "PLANETA" (a, b), 1 de "SUDAMERICANA" (c)
    assert aux["n_libros_editorial_leidos_por_usuario"]["u1"]["PLANETA"] == 2
    assert aux["n_libros_editorial_leidos_por_usuario"]["u1"]["SUDAMERICANA"] == 1
    # tamano de catalogo por editorial: PLANETA tiene 2 libros (a, b) en TODO
    # el catalogo (no solo los leidos), SUDAMERICANA tiene 1 (c)
    assert aux["n_libros_por_editorial"]["PLANETA"] == 2
    assert aux["n_libros_por_editorial"]["SUDAMERICANA"] == 1
    # anio promedio de a,b,c = (2000+2010+2020)/3
    assert aux["anio_edicion_promedio_por_usuario"]["u1"] == 2010.0
    # generos normalizados: "terror" (a,b, normalizado igual) + "novela negra" (c) = 2 distintos
    assert aux["n_generos_distintos_por_usuario"]["u1"] == 2
    # la interaccion mas reciente de todo el dataset es "c" (01-01-2022) -> 0 dias
    assert aux["dias_desde_ultima_interaccion_por_usuario"]["u1"] == 0

    # co-lectura: u1 leyo a/b/c juntos -> cooc entre esas 3 columnas > 0, "d" en 0
    assert aux["columna_por_libro"] == {"a": 0, "b": 1, "c": 2, "d": 3}
    cooc = aux["cooc"]
    assert cooc[0, 1] > 0  # a y b co-leidos por u1
    assert cooc[0, 3] == 0  # d nunca co-leido

    # perfil de texto: "d" no tiene resumen, no entra al indice de texto
    assert "d" not in aux["fila_por_libro_texto"]
    assert set(aux["fila_por_libro_texto"]) == {"a", "b", "c"}
    perfil_u1 = aux["perfil_usuario_norm"][fila_por_usuario["u1"]]
    norma_u1 = np.sqrt(perfil_u1.multiply(perfil_u1).sum())
    assert norma_u1 == pytest.approx(1.0, abs=1e-6)  # normalizado L2, u1 tiene señal real (leyo a y b)

    # macro-genero: "terror" (a, b) no esta en el mapa explicito -> catch-all;
    # "novela negra" (c) si esta mapeado a su propia familia
    assert aux["genero_macro_por_libro"]["a"] == MACRO_GENERO_DEFAULT
    assert aux["genero_macro_por_libro"]["c"] == "novela negra y suspenso"
    # frecuencia: u1 leyo 2/3 catch-all (a,b) + 1/3 "novela negra y suspenso" (c)
    assert aux["frecuencia_genero_macro_por_usuario"]["u1"][MACRO_GENERO_DEFAULT] == pytest.approx(2 / 3)
    assert aux["frecuencia_genero_macro_por_usuario"]["u1"]["novela negra y suspenso"] == pytest.approx(1 / 3)
    # popularidad pooleada por macro-genero: existe un score para cada libro con interaccion
    assert aux["score_por_libro_genero_macro"]["c"] == pytest.approx(8.0)


def test_generar_candidatos_incluye_features_de_autor_y_recencia():
    modelo = _ModeloALSFalso({0: ([0], [0.7])})
    matriz = np.zeros((1, 1))
    stats_pop = _stats_popularidad(["a"], [5.0])
    aux = {
        "autor_por_libro": {"a": "KING, STEPHEN"},
        "anio_edicion_por_libro": {"a": 2015.0},
        "n_libros_autor_leidos_por_usuario": {"u1": {"KING, STEPHEN": 3}},
        "editorial_por_libro": {"a": "PLANETA"},
        "n_libros_editorial_leidos_por_usuario": {"u1": {"PLANETA": 2}},
        "anio_edicion_promedio_por_usuario": {"u1": 2000.0},
        "n_generos_distintos_por_usuario": {"u1": 4},
        "dias_desde_ultima_interaccion_por_usuario": {"u1": 30},
    }

    candidatos = generar_candidatos_con_features(
        usuarios=["u1"],
        modelo_als=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u1": 0},
        libros_por_columna=["a"],
        stats_popularidad=stats_pop,
        stats_por_genero={},
        genero_por_usuario={},
        libros_leidos={},
        n_interacciones_por_usuario={},
        features_auxiliares=aux,
        n_por_fuente=150,
    )

    fila = candidatos.iloc[0]
    assert fila["en_autor_leido"] == 1
    assert fila["n_libros_autor_leidos"] == 3
    assert fila["anio_edicion_dif"] == 2015.0 - 2000.0
    assert fila["n_generos_distintos_usuario"] == 4
    assert fila["dias_desde_ultima_interaccion_usuario"] == 30
    assert fila["en_editorial_leida"] == 1
    assert fila["n_libros_editorial_leidos"] == 2


def test_generar_candidatos_sentinel_cuando_no_hay_dato_de_recencia():
    modelo = _ModeloALSFalso({0: ([0], [0.7])})
    matriz = np.zeros((1, 1))
    stats_pop = _stats_popularidad(["a"], [5.0])

    candidatos = generar_candidatos_con_features(
        usuarios=["u_sin_fecha"],
        modelo_als=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u_sin_fecha": 0},
        libros_por_columna=["a"],
        stats_popularidad=stats_pop,
        stats_por_genero={},
        genero_por_usuario={},
        libros_leidos={},
        n_interacciones_por_usuario={},
        features_auxiliares=_features_auxiliares_vacias(),
        n_por_fuente=150,
    )

    fila = candidatos.iloc[0]
    assert fila["dias_desde_ultima_interaccion_usuario"] == SENTINEL_DIAS_DESCONOCIDO
    assert fila["en_autor_leido"] == 0
    assert fila["anio_edicion_dif"] == 0.0
    assert fila["en_editorial_leida"] == 0
    assert fila["score_coleido"] == 0.0
    assert fila["sim_resumen_historial"] == 0.0


def test_generar_candidatos_incluye_score_coleido():
    # u1 tiene fila ALS 0; matriz binaria u1 x [a, b, c] = leyo a y b
    modelo = _ModeloALSFalso({0: ([2], [0.4])})  # candidato propuesto por ALS: "c"
    matriz = np.array([[1, 1, 0]])
    libros_por_columna = ["a", "b", "c"]
    # cooc: a y c fueron co-leidos por otro usuario 5 veces, b y c nunca
    cooc = sp.csr_matrix(np.array([[0, 0, 5], [0, 0, 0], [5, 0, 0]]))
    aux = {**_features_auxiliares_vacias(), "cooc": cooc, "columna_por_libro": {"a": 0, "b": 1, "c": 2}}

    candidatos = generar_candidatos_con_features(
        usuarios=["u1"],
        modelo_als=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u1": 0},
        libros_por_columna=libros_por_columna,
        stats_popularidad=_stats_popularidad(["c"], [5.0]),
        stats_por_genero={},
        genero_por_usuario={},
        libros_leidos={},
        n_interacciones_por_usuario={},
        features_auxiliares=aux,
        n_por_fuente=150,
    )

    fila_c = candidatos[candidatos["id_libro"] == "c"].iloc[0]
    # score_coleido(u1, c) = cooc[a,c] + cooc[b,c] (u1 leyo a y b) = 5 + 0
    assert fila_c["score_coleido"] == 5.0


def test_generar_candidatos_incluye_sim_resumen_historial():
    modelo = _ModeloALSFalso({0: ([1], [0.4])})  # candidato propuesto por ALS: "b"
    matriz = np.array([[1, 0]])  # u1 leyo "a" (columna 0)
    tfidf_norm = sp.csr_matrix(np.array([[1.0, 0.0], [0.8, 0.6]]))  # filas: "a", "b" (ya L2-normalizadas)
    aux = {
        **_features_auxiliares_vacias(),
        "tfidf_norm": tfidf_norm,
        "fila_por_libro_texto": {"a": 0, "b": 1},
        "perfil_usuario_norm": sp.csr_matrix(np.array([[1.0, 0.0]])),  # perfil de u1 = vector de "a"
    }

    candidatos = generar_candidatos_con_features(
        usuarios=["u1"],
        modelo_als=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u1": 0},
        libros_por_columna=["a", "b"],
        stats_popularidad=_stats_popularidad(["b"], [5.0]),
        stats_por_genero={},
        genero_por_usuario={},
        libros_leidos={},
        n_interacciones_por_usuario={},
        features_auxiliares=aux,
        n_por_fuente=150,
    )

    fila_b = candidatos[candidatos["id_libro"] == "b"].iloc[0]
    # sim(u1, b) = perfil_u1 . tfidf["b"] = [1,0] . [0.8,0.6] = 0.8
    assert fila_b["sim_resumen_historial"] == pytest.approx(0.8)


def test_generar_candidatos_incluye_features_de_genero_macro():
    modelo = _ModeloALSFalso({0: ([0], [0.7])})
    matriz = np.zeros((1, 1))
    stats_pop = _stats_popularidad(["a"], [5.0])
    aux = {
        **_features_auxiliares_vacias(),
        "genero_macro_por_libro": {"a": "novela negra y suspenso"},
        "score_por_libro_genero_macro": {"a": 6.5},
        "frecuencia_genero_macro_por_usuario": {"u1": {"novela negra y suspenso": 0.75}},
    }

    candidatos = generar_candidatos_con_features(
        usuarios=["u1"],
        modelo_als=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u1": 0},
        libros_por_columna=["a"],
        stats_popularidad=stats_pop,
        stats_por_genero={},
        genero_por_usuario={},
        libros_leidos={},
        n_interacciones_por_usuario={},
        features_auxiliares=aux,
        n_por_fuente=150,
    )

    fila = candidatos.iloc[0]
    assert fila["popularidad_genero_macro_candidato"] == pytest.approx(6.5)
    assert fila["frecuencia_genero_macro_usuario"] == pytest.approx(0.75)


def test_generar_candidatos_sentinel_cuando_no_hay_dato_de_genero_macro():
    modelo = _ModeloALSFalso({0: ([0], [0.7])})
    matriz = np.zeros((1, 1))
    stats_pop = _stats_popularidad(["a"], [5.0])
    # "a" no tiene macro-genero conocido (libro sin genero en libros.genero)
    aux = {**_features_auxiliares_vacias(), "genero_macro_por_libro": {"a": None}}

    candidatos = generar_candidatos_con_features(
        usuarios=["u1"],
        modelo_als=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u1": 0},
        libros_por_columna=["a"],
        stats_popularidad=stats_pop,
        stats_por_genero={},
        genero_por_usuario={},
        libros_leidos={},
        n_interacciones_por_usuario={},
        features_auxiliares=aux,
        n_por_fuente=150,
    )

    fila = candidatos.iloc[0]
    assert fila["popularidad_genero_macro_candidato"] == 0.0
    assert fila["frecuencia_genero_macro_usuario"] == 0.0


def test_generar_candidatos_incluye_tamano_catalogo_editorial():
    modelo = _ModeloALSFalso({0: ([0], [0.7])})
    matriz = np.zeros((1, 1))
    stats_pop = _stats_popularidad(["a"], [5.0])
    aux = {
        **_features_auxiliares_vacias(),
        "editorial_por_libro": {"a": "PLANETA"},
        "n_libros_por_editorial": {"PLANETA": 1874},
    }

    candidatos = generar_candidatos_con_features(
        usuarios=["u1"],
        modelo_als=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u1": 0},
        libros_por_columna=["a"],
        stats_popularidad=stats_pop,
        stats_por_genero={},
        genero_por_usuario={},
        libros_leidos={},
        n_interacciones_por_usuario={},
        features_auxiliares=aux,
        n_por_fuente=150,
    )

    assert candidatos.iloc[0]["n_libros_editorial_catalogo"] == 1874


def test_generar_candidatos_sentinel_cuando_no_hay_editorial():
    modelo = _ModeloALSFalso({0: ([0], [0.7])})
    matriz = np.zeros((1, 1))
    stats_pop = _stats_popularidad(["a"], [5.0])
    # "a" no tiene editorial conocida
    aux = {**_features_auxiliares_vacias(), "editorial_por_libro": {"a": None}}

    candidatos = generar_candidatos_con_features(
        usuarios=["u1"],
        modelo_als=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario={"u1": 0},
        libros_por_columna=["a"],
        stats_popularidad=stats_pop,
        stats_por_genero={},
        genero_por_usuario={},
        libros_leidos={},
        n_interacciones_por_usuario={},
        features_auxiliares=aux,
        n_por_fuente=150,
    )

    assert candidatos.iloc[0]["n_libros_editorial_catalogo"] == 0


def test_armar_dataset_marca_el_positivo_y_arma_group():
    candidatos_df = pd.DataFrame(
        {
            "id_lector": ["u1", "u1"],
            "id_libro": ["a", "b"],
            **{f: [0, 0] for f in FEATURES},
        }
    )
    etiquetas_df = pd.DataFrame({"id_lector": ["u1"], "id_libro": ["b"]})

    X, y, group = armar_dataset_entrenamiento(candidatos_df, etiquetas_df)

    assert group == [2]
    assert y.tolist() == [0, 1]
    assert list(X.columns) == FEATURES


def test_armar_dataset_inyecta_positivo_faltante():
    candidatos_df = pd.DataFrame(
        {
            "id_lector": ["u1"],
            "id_libro": ["a"],
            **{f: [0] for f in FEATURES},
        }
    )
    # "z" (la etiqueta real) no esta entre los candidatos generados de u1
    etiquetas_df = pd.DataFrame({"id_lector": ["u1"], "id_libro": ["z"]})

    X, y, group = armar_dataset_entrenamiento(candidatos_df, etiquetas_df, n_por_fuente=150)

    assert group == [2]  # "a" + "z" inyectado
    assert y.sum() == 1
    fila_inyectada = X[y == 1].iloc[0]
    assert fila_inyectada["rank_als"] == 150  # sentinel


def test_armar_dataset_usuario_sin_candidatos_igual_aporta_el_positivo():
    candidatos_df = pd.DataFrame(columns=["id_lector", "id_libro"] + FEATURES)
    etiquetas_df = pd.DataFrame({"id_lector": ["u_frio"], "id_libro": ["z"]})

    X, y, group = armar_dataset_entrenamiento(candidatos_df, etiquetas_df)

    assert group == [1]
    assert y.tolist() == [1]
