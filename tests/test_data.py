"""Tests para split_train_val (leave-one-out temporal por usuario),
split_temporal_global (corte de calendario único para todo el dataset) y las
utilidades de saneamiento de catálogo (normalizar_texto, canonicalizar_libros_duplicados)."""

import pandas as pd

from recsys.data import (
    canonicalizar_libros_duplicados,
    normalizar_texto,
    split_temporal_global,
    split_train_val,
)


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


def test_normalizar_texto_ignora_acentos_mayusculas_y_espacios():
    assert normalizar_texto("GARCÍA MÁRQUEZ, GABRIEL") == normalizar_texto("Garcia Marquez, Gabriel")
    assert normalizar_texto("EDICIONES B") == normalizar_texto("EDICIONES  B")  # espacio doble


def test_normalizar_texto_ignora_puntuacion():
    assert normalizar_texto("ARROBA@BOOKS") == normalizar_texto("ARROBABOOKS")


def test_normalizar_texto_nulo_da_none():
    assert normalizar_texto(None) is None
    assert normalizar_texto(pd.NA) is None
    assert normalizar_texto(float("nan")) is None


def test_normalizar_texto_vacio_da_none_no_string_vacio():
    assert normalizar_texto("   ") is None
    assert normalizar_texto("---") is None  # normaliza a "" (solo puntuacion)


def _libro(id_libro, titulo=None, autor=None, isbn=None, editorial=None, anio_edicion=None, resumen=None):
    return {
        "id_libro": id_libro,
        "titulo": titulo,
        "autor": autor,
        "genero": None,
        "editorial": editorial,
        "anio_edicion": anio_edicion,
        "isbn": isbn,
        "resumen": resumen,
        "img_src": None,
    }


def test_canonicaliza_por_isbn_compartido():
    libros = pd.DataFrame(
        [
            _libro("a", titulo="El Principito", autor="Saint-Exupery", isbn="123"),
            _libro("b", titulo="El Principito (reed.)", autor="Saint-Exupery", isbn="123"),
        ]
    )
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u2"],
            "id_libro": ["a", "b"],
            "fecha": ["01-01-2020", "01-01-2021"],
            "rating": [8, 9],
        }
    )

    libros_canon, interacciones_canon = canonicalizar_libros_duplicados(libros, interacciones)

    assert len(libros_canon) == 1
    assert set(interacciones_canon["id_libro"]) == {libros_canon["id_libro"].iloc[0]}


def test_canonicaliza_por_titulo_autor_normalizado_sin_isbn_compartido():
    libros = pd.DataFrame(
        [
            _libro("a", titulo="Rayuela", autor="CORTAZAR, JULIO", isbn="111"),
            _libro("b", titulo="RAYUELA", autor="Cortazar, Julio", isbn="222"),
        ]
    )
    interacciones = pd.DataFrame(
        {"id_lector": ["u1"], "id_libro": ["a"], "fecha": ["01-01-2020"], "rating": [8]}
    )

    libros_canon, _ = canonicalizar_libros_duplicados(libros, interacciones)

    assert len(libros_canon) == 1


def test_canonicaliza_encadena_transitivamente():
    # a-b comparten isbn; b-c comparten titulo+autor normalizado; a y c no
    # comparten ninguna clave entre si directamente, pero deben quedar en el
    # mismo grupo via el union-find (a-b-c).
    libros = pd.DataFrame(
        [
            _libro("a", titulo="Titulo A", autor="Autor A", isbn="999"),
            _libro("b", titulo="Titulo B Distinto", autor="Autor B Distinto", isbn="999"),
            _libro("c", titulo="Titulo B Distinto", autor="Autor B Distinto", isbn="888"),
        ]
    )
    interacciones = pd.DataFrame(
        {"id_lector": ["u1"], "id_libro": ["a"], "fecha": ["01-01-2020"], "rating": [8]}
    )

    libros_canon, _ = canonicalizar_libros_duplicados(libros, interacciones)

    assert len(libros_canon) == 1


def test_canonicaliza_no_toca_libros_sin_duplicado():
    libros = pd.DataFrame(
        [
            _libro("a", titulo="Libro Uno", autor="Autor Uno", isbn="111"),
            _libro("b", titulo="Libro Dos", autor="Autor Dos", isbn="222"),
        ]
    )
    interacciones = pd.DataFrame(
        {"id_lector": ["u1", "u1"], "id_libro": ["a", "b"], "fecha": ["01-01-2020", "01-01-2021"], "rating": [8, 8]}
    )

    libros_canon, interacciones_canon = canonicalizar_libros_duplicados(libros, interacciones)

    assert set(libros_canon["id_libro"]) == {"a", "b"}
    assert set(interacciones_canon["id_libro"]) == {"a", "b"}


def test_canonico_es_el_de_mas_interacciones():
    libros = pd.DataFrame(
        [
            _libro("poco_leido", titulo="Cien Anios de Soledad", autor="Garcia Marquez", isbn="1"),
            _libro("muy_leido", titulo="Cien Anios de Soledad", autor="Garcia Marquez", isbn="2"),
        ]
    )
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u2", "u3"],
            "id_libro": ["poco_leido", "muy_leido", "muy_leido"],
            "fecha": ["01-01-2020", "01-01-2020", "01-01-2021"],
            "rating": [8, 8, 8],
        }
    )

    libros_canon, interacciones_canon = canonicalizar_libros_duplicados(libros, interacciones)

    assert libros_canon["id_libro"].iloc[0] == "muy_leido"
    assert set(interacciones_canon["id_libro"]) == {"muy_leido"}


def test_canonico_completa_metadata_nula_desde_el_resto_del_grupo():
    libros = pd.DataFrame(
        [
            _libro("canon", titulo="Libro X", autor="Autor X", isbn="1", resumen=None, anio_edicion=2020),
            _libro("dup", titulo="Libro X", autor="Autor X", isbn="2", resumen="un resumen", anio_edicion=None),
        ]
    )
    interacciones = pd.DataFrame(
        {"id_lector": ["u1", "u2"], "id_libro": ["canon", "canon"], "fecha": ["01-01-2020", "01-01-2021"], "rating": [8, 8]}
    )
    # "canon" ya tiene mas interacciones (2 vs 0 de "dup"), asi que queda como el id
    # canonico -- pero le falta resumen, que "dup" si tiene.

    libros_canon, _ = canonicalizar_libros_duplicados(libros, interacciones)

    assert libros_canon["id_libro"].iloc[0] == "canon"
    assert libros_canon["resumen"].iloc[0] == "un resumen"
    assert libros_canon["anio_edicion"].iloc[0] == 2020


def test_canonicaliza_deja_intacto_un_id_libro_huerfano_sin_fila_en_libros():
    # Gap de integridad referencial preexistente (confirmado en los datos reales,
    # ver experiments/estado_del_arte.md): un id_libro de interacciones sin fila
    # correspondiente en libros no tiene grupo que asignarle -- debe quedar TAL
    # CUAL (mismo comportamiento que sin canonicalizar), nunca convertirse en NaN.
    libros = pd.DataFrame([_libro("a", titulo="Libro A", autor="Autor A", isbn="1")])
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1"],
            "id_libro": ["a", "huerfano-sin-metadata"],
            "fecha": ["01-01-2020", "01-01-2020"],
            "rating": [8, 8],
        }
    )

    _, interacciones_canon = canonicalizar_libros_duplicados(libros, interacciones)

    assert set(interacciones_canon["id_libro"]) == {"a", "huerfano-sin-metadata"}
    assert interacciones_canon["id_libro"].notna().all()


def test_canonicaliza_deduplica_interacciones_repetidas_tras_el_remapeo():
    # Mismo usuario leyo dos ediciones distintas del mismo libro -- tras
    # remapear ambas al mismo id_libro canonico, debe quedar UNA sola
    # interaccion (la de fecha mas temprana), no dos lecturas del mismo libro.
    libros = pd.DataFrame(
        [
            _libro("ed1", titulo="Libro Y", autor="Autor Y", isbn="1"),
            _libro("ed2", titulo="Libro Y", autor="Autor Y", isbn="2"),
        ]
    )
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1"],
            "id_libro": ["ed1", "ed2"],
            "fecha": ["01-06-2021", "01-01-2020"],
            "rating": [7, 9],
        }
    )

    _, interacciones_canon = canonicalizar_libros_duplicados(libros, interacciones)

    assert len(interacciones_canon) == 1
    assert interacciones_canon["fecha"].iloc[0] == "01-01-2020"  # la mas temprana
    assert interacciones_canon["rating"].iloc[0] == 9
