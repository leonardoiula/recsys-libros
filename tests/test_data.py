"""Tests para split_train_val (leave-one-out temporal por usuario) y
split_temporal_global (corte de calendario único para todo el dataset)."""

import pandas as pd

from recsys.data import split_temporal_global, split_train_val


def test_retiene_las_n_val_interacciones_mas_recientes():
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1", "u1", "u1"],
            "id_libro": ["a", "b", "c", "d"],
            "fecha": ["01-01-2020", "01-01-2022", "01-01-2021", "01-01-2023"],
            "rating": [8, 8, 8, 8],
        }
    )

    train, val = split_train_val(interacciones, n_val=1, seed=42)

    # "d" (01-01-2023) es la mas reciente -> va a val
    assert val["id_libro"].tolist() == ["d"]
    assert set(train["id_libro"]) == {"a", "b", "c"}


def test_n_val_mayor_a_uno_retiene_las_ultimas_n():
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1"] * 5,
            "id_libro": ["a", "b", "c", "d", "e"],
            "fecha": ["01-01-2020", "01-01-2021", "01-01-2022", "01-01-2023", "01-01-2024"],
            "rating": [8] * 5,
        }
    )

    train, val = split_train_val(interacciones, n_val=2, seed=42)

    assert set(val["id_libro"]) == {"d", "e"}
    assert set(train["id_libro"]) == {"a", "b", "c"}


def test_usuario_con_una_sola_interaccion_queda_entero_en_train():
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1"],
            "id_libro": ["a"],
            "fecha": ["01-01-2020"],
            "rating": [8],
        }
    )

    train, val = split_train_val(interacciones, n_val=1, seed=42)

    assert len(val) == 0
    assert train["id_libro"].tolist() == ["a"]


def test_nunca_vacia_el_train_de_un_usuario():
    # n_val=3 pero el usuario solo tiene 3 interacciones -> como maximo
    # se retiene len(idx)-1 = 2 a val, siempre queda >=1 en train.
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1", "u1"],
            "id_libro": ["a", "b", "c"],
            "fecha": ["01-01-2020", "01-01-2021", "01-01-2022"],
            "rating": [8, 8, 8],
        }
    )

    train, val = split_train_val(interacciones, n_val=3, seed=42)

    assert len(train) == 1
    assert len(val) == 2
    assert train["id_libro"].tolist() == ["a"]


def test_fechas_no_parseables_se_tratan_como_las_mas_antiguas():
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1", "u1"],
            "id_libro": ["a", "b", "c"],
            "fecha": ["no-es-fecha", "01-01-2021", "01-01-2020"],
            "rating": [8, 8, 8],
        }
    )

    train, val = split_train_val(interacciones, n_val=1, seed=42)

    # "b" (01-01-2021) es la mas reciente -> va a val; "a" (fecha invalida)
    # se trata como la mas antigua, nunca termina en val.
    assert val["id_libro"].tolist() == ["b"]
    assert set(train["id_libro"]) == {"a", "c"}


def test_split_es_deterministico_dado_el_mismo_seed():
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1", "u2", "u2"],
            "id_libro": ["a", "b", "c", "d"],
            "fecha": ["01-01-2020", "01-01-2020", "01-01-2022", "01-01-2022"],
            "rating": [8, 8, 8, 8],
        }
    )

    train1, val1 = split_train_val(interacciones, n_val=1, seed=7)
    train2, val2 = split_train_val(interacciones, n_val=1, seed=7)

    assert val1["id_libro"].tolist() == val2["id_libro"].tolist()
    assert train1["id_libro"].tolist() == train2["id_libro"].tolist()


def test_global_fecha_corte_string_se_parsea_DD_MM_YYYY_no_MM_DD():
    # Regresion: pd.Timestamp("01-07-2024") sin formato explicito lo lee como
    # 7 de enero (MM-DD, convencion US) en vez de 1 de julio (DD-MM, el
    # formato real de la columna `fecha` en todo el proyecto). "10-03-2020"
    # es ambiguo entre 10 de marzo (DD-MM) y 3 de octubre (MM-DD) -- un
    # libro fechado "15-03-2020" (15 de marzo) tiene que caer del lado
    # correcto segun cual de las dos interpretaciones se use.
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1"],
            "id_libro": ["a"],
            "fecha": ["15-03-2020"],  # 15 de marzo de 2020
            "rating": [8],
        }
    )

    # Corte "10-03-2020" = 10 de marzo (DD-MM) -> "a" (15 de marzo) es POSTERIOR -> val.
    # Si se leyera como 3 de octubre (MM-DD), "a" seria ANTERIOR -> train (mal).
    train, val = split_temporal_global(interacciones, fecha_corte="10-03-2020", seed=42)

    assert val["id_libro"].tolist() == ["a"]
    assert len(train) == 0


def test_global_train_es_todo_lo_anterior_o_igual_al_corte():
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1", "u2"],
            "id_libro": ["a", "b", "c"],
            "fecha": ["01-01-2020", "01-01-2022", "01-01-2021"],
            "rating": [8, 8, 8],
        }
    )

    train, val = split_temporal_global(interacciones, fecha_corte="01-06-2021", seed=42)

    # "b" (2022) es la unica posterior al corte -> a val; el resto a train.
    assert val["id_libro"].tolist() == ["b"]
    assert set(train["id_libro"]) == {"a", "c"}


def test_global_toma_la_interaccion_MAS_TEMPRANA_despues_del_corte():
    # A diferencia de split_train_val (que retiene la MAS RECIENTE de cada
    # usuario), split_temporal_global busca el PRIMER "proximo libro" que el
    # usuario lee despues del corte -- no el ultimo.
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1", "u1"],
            "id_libro": ["temprana", "tardia", "antes_del_corte"],
            "fecha": ["01-02-2021", "01-06-2021", "01-01-2020"],
            "rating": [8, 8, 8],
        }
    )

    train, val = split_temporal_global(interacciones, fecha_corte="01-01-2021", seed=42)

    assert val["id_libro"].tolist() == ["temprana"]
    assert set(train["id_libro"]) == {"antes_del_corte"}


def test_global_usuario_sin_interacciones_posteriores_queda_entero_en_train():
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1"],
            "id_libro": ["a", "b"],
            "fecha": ["01-01-2019", "01-01-2020"],
            "rating": [8, 8],
        }
    )

    train, val = split_temporal_global(interacciones, fecha_corte="01-01-2021", seed=42)

    assert len(val) == 0
    assert set(train["id_libro"]) == {"a", "b"}


def test_global_fechas_no_parseables_nunca_van_a_val():
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1"],
            "id_libro": ["sin_fecha", "antes_del_corte"],
            "fecha": ["no-es-fecha", "01-01-2019"],
            "rating": [8, 8],
        }
    )

    train, val = split_temporal_global(interacciones, fecha_corte="01-01-2020", seed=42)

    assert len(val) == 0
    assert set(train["id_libro"]) == {"sin_fecha", "antes_del_corte"}


def test_global_es_deterministico_dado_el_mismo_seed():
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1", "u2", "u2"],
            "id_libro": ["a", "b", "c", "d"],
            "fecha": ["01-06-2021", "01-06-2021", "01-07-2021", "01-07-2021"],
            "rating": [8, 8, 8, 8],
        }
    )

    train1, val1 = split_temporal_global(interacciones, fecha_corte="01-01-2021", seed=7)
    train2, val2 = split_temporal_global(interacciones, fecha_corte="01-01-2021", seed=7)

    assert val1["id_libro"].tolist() == val2["id_libro"].tolist()
    assert train1["id_libro"].tolist() == train2["id_libro"].tolist()


def test_global_no_todos_los_usuarios_aparecen_en_val():
    # A diferencia de split_train_val, no hay garantia de cobertura total en
    # val -- es intencional (Kaggle tampoco evalua a todo el mundo).
    interacciones = pd.DataFrame(
        {
            "id_lector": ["activo", "inactivo"],
            "id_libro": ["a", "b"],
            "fecha": ["01-06-2021", "01-01-2010"],
            "rating": [8, 8],
        }
    )

    train, val = split_temporal_global(interacciones, fecha_corte="01-01-2021", seed=42)

    assert val["id_lector"].tolist() == ["activo"]
    assert "inactivo" not in set(val["id_lector"])
    assert "inactivo" in set(train["id_lector"])
