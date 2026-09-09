import pandas as pd

from comics_recsys.data import comics_calificados_por_usuario, split_train_val


def _interacciones(filas):
    return pd.DataFrame(filas, columns=["id_usuario", "id_comic", "fecha", "rating"])


def test_retiene_la_interaccion_mas_reciente_en_validacion():
    inter = _interacciones(
        [
            ("u1", "c1", "2024-01-01", 8.0),
            ("u1", "c2", "2024-06-01", 9.0),
            ("u1", "c3", "2024-03-01", 7.0),
        ]
    )
    train, val = split_train_val(inter, n_val=1)
    assert val["id_comic"].tolist() == ["c2"]
    assert set(train["id_comic"]) == {"c1", "c3"}


def test_nunca_vacia_el_train_de_un_usuario():
    inter = _interacciones([("u1", "c1", "2024-01-01", 8.0)])
    train, val = split_train_val(inter, n_val=1)
    assert len(train) == 1
    assert len(val) == 0


def test_fechas_no_parseables_se_tratan_como_las_mas_antiguas():
    inter = _interacciones(
        [
            ("u1", "c1", "fecha-invalida", 8.0),
            ("u1", "c2", "2024-06-01", 9.0),
        ]
    )
    train, val = split_train_val(inter, n_val=1)
    assert val["id_comic"].tolist() == ["c2"]
    assert train["id_comic"].tolist() == ["c1"]


def test_split_es_deterministico_dado_el_mismo_seed():
    inter = _interacciones(
        [(f"u{i}", f"c{i}", "2024-01-01", 8.0) for i in range(20)]
        + [(f"u{i}", f"c{i}b", "2024-06-01", 9.0) for i in range(20)]
    )
    train1, val1 = split_train_val(inter, n_val=1, seed=7)
    train2, val2 = split_train_val(inter, n_val=1, seed=7)
    assert train1.equals(train2)
    assert val1.equals(val2)


def test_comics_calificados_por_usuario():
    inter = _interacciones(
        [
            ("u1", "c1", "2024-01-01", 8.0),
            ("u1", "c2", "2024-01-02", 7.0),
            ("u2", "c1", "2024-01-01", 6.0),
        ]
    )
    resultado = comics_calificados_por_usuario(inter)
    assert resultado == {"u1": {"c1", "c2"}, "u2": {"c1"}}
