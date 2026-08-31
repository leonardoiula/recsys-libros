"""Tests para popularidad segmentada por género/franja de nacimiento y su
cadena de fallback (género -> franja de nacimiento -> global)."""

import pandas as pd

from recsys.models.popularity_segmentada import (
    FRANJA_DESCONOCIDA,
    GENERO_LECTOR_DESCONOCIDO,
    MACRO_GENERO_DEFAULT,
    PAIS_DESCONOCIDO,
    fit_popularidad_por_genero_macro,
    fit_popularity_por_franja_nacimiento,
    fit_popularity_por_genero_lector,
    fit_popularity_por_pais,
    franja_nacimiento_por_usuario,
    genero_lector_por_usuario,
    genero_macro,
    genero_preferido_por_usuario,
    normalizar_genero_macro,
    pais_por_usuario,
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

    # sin tilde: _normalizar_genero ahora también ignora acentos (ver
    # test_normalizar_genero_ignora_acentos), no solo capitalización.
    assert preferido["u1"] == "fantastica"


def test_normalizar_genero_ignora_acentos():
    # "clásicos"/"clasicos" son el mismo género real, separado en el dato
    # crudo por una tilde inconsistente (ver bitacora) -- deberían unirse
    # en una sola categoría, no quedar como géneros distintos.
    libros = pd.DataFrame(
        {
            "id_libro": ["a", "b", "c"],
            "genero": ["Clásicos de la literatura", "clasicos de la literatura", "Clasicos De La Literatura"],
        }
    )
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u1", "u1"],
            "id_libro": ["a", "b", "c"],
            "rating": [8, 8, 8],
        }
    )

    preferido = genero_preferido_por_usuario(interacciones, libros)

    assert preferido["u1"] == "clasicos de la literatura"


def test_genero_macro_mapea_categoria_conocida():
    assert genero_macro("narrativa") == "narrativa y clasicos"
    assert genero_macro("novela negra, intriga, terror") == "novela negra y suspenso"


def test_genero_macro_usa_catchall_para_categoria_desconocida():
    assert genero_macro("cocina") == MACRO_GENERO_DEFAULT


def test_genero_macro_preserva_none():
    assert genero_macro(None) is None
    assert genero_macro(pd.NA) is None


def test_normalizar_genero_macro_de_extremo_a_extremo():
    generos = pd.Series(["Narrativa", "Cocina", None])
    macro = normalizar_genero_macro(generos)

    assert macro.tolist()[0] == "narrativa y clasicos"
    assert macro.tolist()[1] == MACRO_GENERO_DEFAULT
    assert pd.isna(macro.tolist()[2])


def test_fit_popularidad_por_genero_macro_agrupa_por_macro_no_por_categoria_granular():
    libros = pd.DataFrame(
        {
            "id_libro": ["a", "b", "c"],
            "genero": ["Narrativa", "Novela", "Cocina"],  # narrativa/novela -> mismo macro; cocina -> otro
        }
    )
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u2", "u3"],
            "id_libro": ["a", "b", "c"],
            "rating": [8, 9, 7],
        }
    )

    stats = fit_popularidad_por_genero_macro(interacciones, libros)

    assert set(stats) == {"narrativa y clasicos", MACRO_GENERO_DEFAULT}
    # "a" y "b" (narrativa/novela) caen juntos en la misma tabla de popularidad
    assert set(stats["narrativa y clasicos"]["id_libro"]) == {"a", "b"}


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
    # a diferencia de la version anterior (que descartaba estos lectores),
    # nacimiento invalido/vacio cae en su propia categoria "desconocido"
    assert franjas["u2"] == FRANJA_DESCONOCIDA
    assert franjas["u3"] == FRANJA_DESCONOCIDA


def test_franja_nacimiento_trata_sentinel_1910_como_desconocido():
    lectores = pd.DataFrame(
        {
            "id_lector": ["u1", "u2"],
            "nacimiento": ["1910", "1911"],  # 1910 es el sentinel, 1911 es una decada real
        }
    )

    franjas = franja_nacimiento_por_usuario(lectores)

    assert franjas["u1"] == FRANJA_DESCONOCIDA
    assert franjas["u2"] == "1910s"


def test_fit_popularity_por_franja_nacimiento_agrupa_desconocido_por_separado():
    lectores = pd.DataFrame(
        {
            "id_lector": ["u1", "u2", "u3"],
            "nacimiento": ["1985", "", "1910"],  # u2/u3 caen ambos en "desconocido"
        }
    )
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u2", "u3"],
            "id_libro": ["a", "b", "c"],
            "rating": [8, 6, 7],
        }
    )

    stats = fit_popularity_por_franja_nacimiento(interacciones, lectores)

    assert set(stats) == {"1980s", FRANJA_DESCONOCIDA}
    assert set(stats["1980s"]["id_libro"]) == {"a"}
    # u2 (vacio) y u3 (sentinel 1910) se agrupan juntos en "desconocido"
    assert set(stats[FRANJA_DESCONOCIDA]["id_libro"]) == {"b", "c"}


def test_genero_lector_por_usuario_mapea_guion_a_desconocido():
    lectores = pd.DataFrame(
        {
            "id_lector": ["u1", "u2", "u3"],
            "genero": ["Mujer", "Hombre", "-"],
        }
    )

    generos = genero_lector_por_usuario(lectores)

    assert generos["u1"] == "Mujer"
    assert generos["u2"] == "Hombre"
    assert generos["u3"] == GENERO_LECTOR_DESCONOCIDO


def test_fit_popularity_por_genero_lector_agrupa_por_genero_declarado():
    lectores = pd.DataFrame(
        {
            "id_lector": ["u1", "u2", "u3"],
            "genero": ["Mujer", "Hombre", "-"],
        }
    )
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u2", "u3"],
            "id_libro": ["a", "b", "c"],
            "rating": [8, 6, 7],
        }
    )

    stats = fit_popularity_por_genero_lector(interacciones, lectores)

    assert set(stats) == {"Mujer", "Hombre", GENERO_LECTOR_DESCONOCIDO}
    assert set(stats["Mujer"]["id_libro"]) == {"a"}
    assert set(stats[GENERO_LECTOR_DESCONOCIDO]["id_libro"]) == {"c"}


def test_pais_por_usuario_extrae_pais_de_ciudad_pais():
    lectores = pd.DataFrame(
        {
            "id_lector": ["u1", "u2", "u3"],
            # "Ciudad - País", solo "País", y variantes de tildeo que deberian
            # colapsar en el mismo pais normalizado
            "vive_en": ["Vigo - España", "Mexico", "Bogota - Colombia"],
        }
    )

    paises = pais_por_usuario(lectores)

    assert paises["u1"] == "espana"
    assert paises["u2"] == "mexico"
    assert paises["u3"] == "colombia"


def test_pais_por_usuario_marca_desconocido_como_categoria_propia():
    lectores = pd.DataFrame(
        {
            "id_lector": ["u1", "u2", "u3"],
            # vacio, el placeholder "¿?" del dataset, y nulo real -- los tres
            # son "no especifica", pero a diferencia de franja de nacimiento
            # no se descartan: quedan con su propia categoria
            "vive_en": ["", "¿?", None],
        }
    )

    paises = pais_por_usuario(lectores)

    assert paises["u1"] == PAIS_DESCONOCIDO
    assert paises["u2"] == PAIS_DESCONOCIDO
    assert paises["u3"] == PAIS_DESCONOCIDO
    assert set(paises) == {"u1", "u2", "u3"}  # nadie se descarta


def test_fit_popularity_por_pais_agrupa_por_pais_del_lector():
    lectores = pd.DataFrame(
        {
            "id_lector": ["u1", "u2"],
            "vive_en": ["Madrid - España", "Lima - Peru"],
        }
    )
    interacciones = pd.DataFrame(
        {
            "id_lector": ["u1", "u2"],
            "id_libro": ["a", "b"],
            "rating": [8, 6],
        }
    )

    stats = fit_popularity_por_pais(interacciones, lectores)

    assert set(stats) == {"espana", "peru"}
    assert set(stats["espana"]["id_libro"]) == {"a"}
    assert set(stats["peru"]["id_libro"]) == {"b"}


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
