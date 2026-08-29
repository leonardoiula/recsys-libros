"""Tests para split_train_val (leave-one-out temporal por usuario)."""

import pandas as pd

from recsys.data import split_train_val


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
