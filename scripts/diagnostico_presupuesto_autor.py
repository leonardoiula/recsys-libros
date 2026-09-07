"""Diagnóstico: ¿el tope TOTAL de la fuente de autor (`n_por_fuente=150`
candidatos por usuario, repartido priorizando a los autores MÁS leídos)
deja a usuarios sin candidatos de sus autores "secundarios"?

Uso: uv run python scripts/diagnostico_presupuesto_autor.py

Continúa la investigación abierta del límite del reranking (ver
`experiments/bitacora.md`). `scripts/diagnostico_cap_autor.py` ya midió el
OTRO tope de la fuente (`n_por_autor=20` libros por autor) y encontró que
explica <½ del subgrupo puntual. Acá se mira el tope total: la fuente
recorre los autores del usuario de más leído a menos leído y corta al
llegar a `n_por_fuente` candidatos acumulados (ver
`generar_candidatos_con_features` en `ranker.py`, ~línea 837) -- un
usuario con muchos autores leídos puede agotar el presupuesto en sus
favoritos y no recibir NADA de los demás.

Reproduce el split de 3 niveles y las piezas de la fuente de autor
(`n_libros_autor_leidos_por_usuario`, `libros_por_autor_ordenados`) a
mano, sin correr el pipeline completo. Aproximación (misma lógica que la
nota de precisión de `diagnostico_cap_autor.py`): el `vistos` real
incluye lo que ya propusieron ALS/popularidad/género, que la fuente de
autor SALTEA sin gastar presupuesto -- acá `vistos` es solo lo ya leído,
así que esto SOBREESTIMA cuánto muerde el tope de 150. Es una cota
superior del problema, no el número exacto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import (
    libros_leidos_por_usuario,
    load_interacciones,
    load_libros,
    split_train_val,
)
from recsys.models.popularity import fit_popularity

N_POR_FUENTE = 150
N_POR_AUTOR = 20
SEED = 42


def candidatos_fuente_autor(
    autores_ordenados: list[str],
    libros_por_autor: dict[str, list[str]],
    ya_vistos: set[str],
    tope_total: int,
) -> list[str]:
    """Reproduce el bucle de la fuente de autor: recorre autores de más a
    menos leído, hasta `tope_total` candidatos sin leer en total."""
    out: list[str] = []
    for autor in autores_ordenados:
        if len(out) >= tope_total:
            break
        for id_libro in libros_por_autor.get(autor, []):
            if len(out) >= tope_total:
                break
            if id_libro in ya_vistos or id_libro in out:
                continue
            out.append(id_libro)
    return out


def main() -> None:
    interacciones = load_interacciones()
    libros = load_libros()

    train_candidatos_full, test_final = split_train_val(interacciones, n_val=1, seed=SEED)
    train_candidatos, _train_ranker = split_train_val(train_candidatos_full, n_val=1, seed=SEED + 1000)

    autor_por_libro = libros.set_index("id_libro")["autor"].to_dict()

    # n_libros_autor_leidos_por_usuario (igual que calcular_features_auxiliares)
    con_autor = train_candidatos.assign(autor=train_candidatos["id_libro"].map(autor_por_libro))
    agg = con_autor.dropna(subset=["autor"]).groupby(["id_lector", "autor"]).size()
    autores_por_usuario: dict[str, dict[str, int]] = {}
    for (id_lector, autor), n in agg.items():
        autores_por_usuario.setdefault(id_lector, {})[autor] = int(n)

    # libros_por_autor_ordenados: top-N_POR_AUTOR por popularidad global
    ranking_global_ids = fit_popularity(train_candidatos)["id_libro"].tolist()
    libros_por_autor: dict[str, list[str]] = {}
    for id_libro in ranking_global_ids:
        autor = autor_por_libro.get(id_libro)
        if autor is None or pd.isna(autor):
            continue
        lista = libros_por_autor.setdefault(autor, [])
        if len(lista) < N_POR_AUTOR:
            lista.append(id_libro)

    libros_leidos = libros_leidos_por_usuario(train_candidatos_full)
    objetivo_por_usuario = dict(zip(test_final["id_lector"], test_final["id_libro"]))
    n_interacciones_libro = interacciones.groupby("id_libro").size().to_dict()

    filas = []
    for id_lector in test_final["id_lector"]:
        conteo = autores_por_usuario.get(id_lector, {})
        if not conteo:
            continue
        autores_ordenados = sorted(conteo, key=lambda a: -conteo[a])
        vistos = libros_leidos.get(id_lector, set())

        cand_sin_tope = candidatos_fuente_autor(autores_ordenados, libros_por_autor, vistos, 10**9)
        cand_con_tope = candidatos_fuente_autor(autores_ordenados, libros_por_autor, vistos, N_POR_FUENTE)

        # ¿Cuántos autores quedaron sin ningún candidato por agotar el presupuesto?
        set_con_tope = set(cand_con_tope)
        autores_servidos = {
            autor
            for autor in autores_ordenados
            for id_libro in libros_por_autor.get(autor, [])
            if id_libro in set_con_tope
        }
        autores_con_algun_candidato_posible = {
            autor
            for autor in autores_ordenados
            if any(b not in vistos for b in libros_por_autor.get(autor, []))
        }

        objetivo = objetivo_por_usuario.get(id_lector)
        obj_de_autor_leido = objetivo is not None and autor_por_libro.get(objetivo) in conteo
        obj_en_sin_tope = objetivo in cand_sin_tope if objetivo is not None else False
        obj_en_con_tope = objetivo in set_con_tope if objetivo is not None else False

        filas.append(
            {
                "id_lector": id_lector,
                "n_autores_leidos": len(conteo),
                "n_autores_con_candidato_posible": len(autores_con_algun_candidato_posible),
                "cand_sin_tope": len(cand_sin_tope),
                "cand_con_tope": len(cand_con_tope),
                "hit_cap": len(cand_sin_tope) > N_POR_FUENTE,
                "autores_sin_presupuesto": len(autores_con_algun_candidato_posible) - len(autores_servidos),
                "obj_de_autor_leido": obj_de_autor_leido,
                "obj_perdido_por_tope_total": obj_de_autor_leido and obj_en_sin_tope and not obj_en_con_tope,
                "obj_no_alcanzable_por_fuente": obj_de_autor_leido and not obj_en_sin_tope,
                "n_interacciones_objetivo": n_interacciones_libro.get(objetivo, 0) if objetivo is not None else 0,
            }
        )

    df = pd.DataFrame(filas)
    n = len(df)
    print(f"usuarios en test_final con al menos un autor leído: {n}\n")

    print("=== Tamaño de la fuente de autor por usuario ===")
    print(f"n_autores_leidos          media={df.n_autores_leidos.mean():.1f}  mediana={df.n_autores_leidos.median():.0f}  p90={df.n_autores_leidos.quantile(.9):.0f}  max={df.n_autores_leidos.max()}")
    print(f"candidatos SIN tope total media={df.cand_sin_tope.mean():.1f}  mediana={df.cand_sin_tope.median():.0f}  p90={df.cand_sin_tope.quantile(.9):.0f}  max={df.cand_sin_tope.max()}")
    print(f"candidatos CON tope 150   media={df.cand_con_tope.mean():.1f}  mediana={df.cand_con_tope.median():.0f}")
    print(f"usuarios que TOCAN el tope de {N_POR_FUENTE}: {df.hit_cap.sum()} ({df.hit_cap.mean():.1%})")

    tope = df[df.hit_cap]
    if len(tope):
        print(f"\n=== De los {len(tope)} usuarios que tocan el tope ===")
        perdidos = tope.cand_sin_tope - tope.cand_con_tope
        print(f"candidatos de autor perdidos por el tope  media={perdidos.mean():.1f}  mediana={perdidos.median():.0f}  max={perdidos.max()}")
        print(f"autores que quedan SIN NINGÚN candidato   media={tope.autores_sin_presupuesto.mean():.1f}  mediana={tope.autores_sin_presupuesto.median():.0f}  p90={tope.autores_sin_presupuesto.quantile(.9):.0f}")

    print("\n=== Impacto en el OBJETIVO (el libro que el usuario leyó después) ===")
    de_autor = df[df.obj_de_autor_leido]
    print(f"objetivo de un autor ya leído: {len(de_autor)} usuarios ({len(de_autor)/n:.1%} de los {n})")
    print(f"  de esos, NO alcanzable por la fuente ni sin tope (no está entre los top-{N_POR_AUTOR} populares de ningún autor leído): "
          f"{df.obj_no_alcanzable_por_fuente.sum()} ({df.obj_no_alcanzable_por_fuente.sum()/max(len(de_autor),1):.1%})")
    print(f"  de esos, PERDIDO específicamente por el tope total de {N_POR_FUENTE} "
          f"(estaría sin el tope, no está con el tope): {df.obj_perdido_por_tope_total.sum()} "
          f"({df.obj_perdido_por_tope_total.sum()/max(len(de_autor),1):.1%})")
    perdidos_obj = df[df.obj_perdido_por_tope_total]
    if len(perdidos_obj):
        print(f"    popularidad de esos objetivos perdidos -- n_interacciones mediana: {perdidos_obj.n_interacciones_objetivo.median():.0f} "
              f"(vs {df[df.obj_de_autor_leido].n_interacciones_objetivo.median():.0f} de todos los objetivos de autor leído)")
        print(f"    esos {len(perdidos_obj)} usuarios son el {len(perdidos_obj)/n:.2%} del total de test_final con autor leído")


if __name__ == "__main__":
    main()
