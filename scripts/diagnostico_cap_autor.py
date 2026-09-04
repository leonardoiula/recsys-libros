"""Diagnóstico puntual: cuando un objetivo de popularidad media es de un
autor ya leído pero la fuente de autor NO lo propuso, ¿es porque el tope
`n_por_autor=20` (ranking por popularidad GLOBAL dentro del autor) lo
dejó afuera?

Uso: uv run python scripts/diagnostico_cap_autor.py

Continúa `scripts/diagnostico_franja_media.py`: ese script encontró que,
en la franja de popularidad media (deciles 3-4), el éxito en el top-20
correlaciona fuerte con que el objetivo llegue TAMBIÉN corroborado por
una fuente específica (autor/resumen/co-lectura ya leídos), no solo por
ALS. Acá se chequea la hipótesis puntual sobre la fuente de autor: el
código (`generar_candidatos_con_features`) arma, por autor, hasta
`n_por_autor=20` libros SIN LEER rankeados por popularidad GLOBAL (no un
score por autor) -- si el libro objetivo no está entre los 20 más
populares de TODO lo que escribió ese autor, nunca se propone como
candidato vía esta fuente, aunque el usuario haya leído mucho de ese
autor.

Nota de precisión: acá se calcula el rank del objetivo dentro del
catálogo COMPLETO del autor (sin excluir los libros que el usuario en
cuestión ya leyó) -- es una cota, no el número exacto que usa el código
real. Si el rank crudo ya es < 20, el objetivo iba a entrar sí o sí
(sacar libros ya leídos del autor solo puede mejorar su rank, nunca
empeorarlo). Si es >= 20, puede que el tope real (después de sacar los
ya leídos de esa autora) lo hubiera dejado pasar igual -- este script
mide el caso más simple/común, no la lista exacta que ve cada usuario.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data import load_interacciones, load_lectores, load_libros
from recsys.models.ranker import FEATURES, evaluar_con_params, preparar_pipeline_cacheado

K = 20
N_POR_FUENTE = 150
N_POR_AUTOR = 20  # mismo default que generar_candidatos_con_features/submit.py
SEED = 42


def main() -> None:
    interacciones = load_interacciones()
    libros = load_libros()
    lectores = load_lectores()

    t0 = time.time()
    ctx = preparar_pipeline_cacheado(interacciones, libros, lectores, SEED, n_por_fuente=N_POR_FUENTE, k=K)
    print(f"contexto listo en {time.time() - t0:.0f}s", flush=True)

    resultado = evaluar_con_params(ctx, None)
    modelo = resultado["modelo_ranker"]

    candidatos = ctx["candidatos_test"]
    test_final = ctx["test_final"]
    objetivo_por_usuario = dict(zip(test_final["id_lector"], test_final["id_libro"]))

    # Mismo ranking_global que usó el código real para armar candidatos_test
    # -- reconstruye el rank de cada libro DENTRO del catálogo de su autor,
    # ordenado igual que `libros_por_autor_ordenados` en generar_candidatos_con_features.
    autor_por_libro = libros.set_index("id_libro")["autor"].to_dict()
    rank_dentro_de_autor: dict[str, int] = {}
    contador_por_autor: dict = {}
    for id_libro in ctx["ranking_global"]:
        autor = autor_por_libro.get(id_libro)
        if autor is None or pd.isna(autor):
            continue
        rank_dentro_de_autor[id_libro] = contador_por_autor.get(autor, 0)
        contador_por_autor[autor] = contador_por_autor.get(autor, 0) + 1

    filas = []
    for id_lector, grupo in candidatos.groupby("id_lector", sort=False):
        objetivo = objetivo_por_usuario.get(id_lector)
        if objetivo is None:
            continue
        fila_objetivo = grupo[grupo["id_libro"] == objetivo]
        if fila_objetivo.empty:
            continue

        scores = modelo.predict(grupo[FEATURES])
        orden = grupo["id_libro"].to_numpy()[np.argsort(-scores)]
        posicion = int(np.where(orden == objetivo)[0][0])

        fila = fila_objetivo.iloc[0].to_dict()
        fila["id_lector"] = id_lector
        fila["id_libro"] = objetivo
        fila["posicion"] = posicion
        fila["rank_dentro_de_autor"] = rank_dentro_de_autor.get(objetivo)
        filas.append(fila)

    df = pd.DataFrame(filas)
    df["decil"] = pd.qcut(df["n_interacciones_libro"], 10, labels=False, duplicates="drop")
    df["en_top_k"] = df["posicion"] < K

    franja_media = df[df["decil"].isin([3, 4])]
    caso = franja_media[(franja_media["en_autor_leido"] == 1) & (franja_media["en_autor_candidato"] == 0)]

    print(f"\nfranja media (deciles 3-4): {len(franja_media)} usuarios")
    print(
        f"de esos, objetivo de un autor ya leido PERO no propuesto por la fuente de autor: "
        f"{len(caso)} ({len(caso) / len(franja_media):.1%})"
    )

    con_rank = caso.dropna(subset=["rank_dentro_de_autor"])
    print(f"con autor identificable en metadata: {len(con_rank)}")
    fuera_del_tope = con_rank["rank_dentro_de_autor"] >= N_POR_AUTOR
    print(
        f"\nde esos, con rank dentro del catalogo del autor >= {N_POR_AUTOR} "
        f"(el tope n_por_autor lo hubiera dejado afuera aunque no hubiera leido nada mas de ese autor): "
        f"{fuera_del_tope.sum()} ({fuera_del_tope.mean():.1%})"
    )
    print(f"rank dentro del autor -- mediana: {con_rank['rank_dentro_de_autor'].median():.0f}, "
          f"percentiles 25/75: {con_rank['rank_dentro_de_autor'].quantile(0.25):.0f}/{con_rank['rank_dentro_de_autor'].quantile(0.75):.0f}")

    print(f"\nP(en_top_{K}) para este grupo (autor leido, no propuesto por la fuente autor): {caso['en_top_k'].mean():.4f}")
    print(f"vs. P(en_top_{K}) de la franja media completa: {franja_media['en_top_k'].mean():.4f}")


if __name__ == "__main__":
    main()
