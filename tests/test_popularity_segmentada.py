"""Tests para popularidad segmentada por género/franja de nacimiento y su
cadena de fallback (género -> franja de nacimiento -> global)."""

import pandas as pd

from recsys.models.popularity_segmentada import (
    franja_nacimiento_por_usuario,
    genero_preferido_por_usuario,
    recomendar_por_usuario,
)


def test_genero_preferido_ignora_capitalizacion():
    libros = pd.DataFrame(
        {
            "id_libro": ["a", "b", "c", "d"],
            "genero": ["Terror", "Fantástica", "FANTÁSTICA", "fantástica"],
        }
    )
    # u1 lee 1 libro de terror y 3 de fantástica (con distinta
    # capitalización) -> debería preferir "fantástica", no "terror" ni
    # tratarlas como géneros separados.
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1", "u1", "u1"],
            "id_libro": ["a", "b", "c", "d"],
            "rating": [8, 8, 8, 8],
        }
    )

    preferido = genero_preferido_por_usuario(interacciones, libros)

    assert preferido["u1"] == "fantástica"


def test_genero_preferido_usuario_sin_generos_conocidos_no_aparece():
    libros = pd.DataFrame({"id_libro": ["a"], "genero": [None]})
    interacciones = pd.DataFrame({"id_lector": ["u1"], "id_libro": ["a"], "rating": [7]})

    preferido = genero_preferido_por_usuario(interacciones, libros)

    assert "u1" not in preferido


def test_franja_nacimiento_por_usuario():
    lectores = pd.DataFrame(
        {
            "id_lector": ["u1", "u2", "u3"],
            "nacimiento": ["1985", "", "no-valido"],
        }
    )

    franjas = franja_nacimiento_por_usuario(lectores)

    assert franjas["u1"] == "1980s"
    assert "u2" not in franjas
    assert "u3" not in franjas


def test_recomendar_usa_genero_cuando_alcanza():
    ranking_por_genero = {
        "terror": pd.DataFrame({"id_libro": ["g1", "g2", "g3"]}),
    }
    ranking_por_franja = {
        "1980s": pd.DataFrame({"id_libro": ["f1", "f2", "f3"]}),
    }
    ranking_global = ["x1", "x2", "x3"]

    recomendaciones = recomendar_por_usuario(
        usuarios=["u1"],
        ranking_por_genero=ranking_por_genero,
        genero_por_usuario={"u1": "terror"},
        ranking_por_franja=ranking_por_franja,
        franja_por_usuario={"u1": "1980s"},
        ranking_global=ranking_global,
        libros_leidos={},
        k=2,
    )

    # alcanza con el género -> nunca debería tocar franja ni global
    assert recomendaciones["u1"] == ["g1", "g2"]


def test_recomendar_cae_a_franja_si_genero_no_alcanza():
    ranking_por_genero = {
        "terror": pd.DataFrame({"id_libro": ["g1"]}),
    }
    ranking_por_franja = {
        "1980s": pd.DataFrame({"id_libro": ["f1", "f2"]}),
    }
    ranking_global = ["x1", "x2"]

    recomendaciones = recomendar_por_usuario(
        usuarios=["u1"],
        ranking_por_genero=ranking_por_genero,
        genero_por_usuario={"u1": "terror"},
        ranking_por_franja=ranking_por_franja,
        franja_por_usuario={"u1": "1980s"},
        ranking_global=ranking_global,
        libros_leidos={},
        k=3,
    )

    # g1 (género, único candidato) + f1, f2 (franja, completa el resto)
    assert recomendaciones["u1"] == ["g1", "f1", "f2"]


def test_recomendar_cae_a_global_sin_genero_ni_franja():
    ranking_global = ["x1", "x2", "x3"]

    recomendaciones = recomendar_por_usuario(
        usuarios=["u1"],
        ranking_por_genero={},
        genero_por_usuario={},
        ranking_por_franja={},
        franja_por_usuario={},
        ranking_global=ranking_global,
        libros_leidos={},
        k=2,
    )

    assert recomendaciones["u1"] == ["x1", "x2"]


def test_recomendar_no_repite_libros_ya_leidos_ni_entre_fuentes():
    ranking_por_genero = {
        "terror": pd.DataFrame({"id_libro": ["g1", "g2"]}),
    }
    # "g2" también aparece en el ranking global: no debería duplicarse
    ranking_global = ["g2", "x1"]

    recomendaciones = recomendar_por_usuario(
        usuarios=["u1"],
        ranking_por_genero=ranking_por_genero,
        genero_por_usuario={"u1": "terror"},
        ranking_por_franja={},
        franja_por_usuario={},
        ranking_global=ranking_global,
        libros_leidos={"u1": {"g1"}},  # "g1" ya leído -> se filtra
        k=2,
    )

    assert recomendaciones["u1"] == ["g2", "x1"]
