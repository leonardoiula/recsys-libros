"""Diagnóstico: ¿qué distingue a los objetivos de popularidad MEDIA que
el reranker sí sube al top-k de los que no, si es que algo lo hace?

Uso: uv run python scripts/diagnostico_franja_media.py

Continúa `scripts/diagnostico_posicion_popularidad.py`: ese script
encontró que la relación entre popularidad del objetivo y P(top-20) NO
es monótona -- tiene forma de U. Los objetivos muy poco populares (decil
0, ~4 interacciones) rankean casi tan bien como los muy populares
(decil 9), pero los de popularidad MEDIA (deciles 3-4, ~100-250
interacciones) son los que peor rankean. Hipótesis a chequear acá: los
objetivos raros llegan como candidatos vía una señal *específica y
fuerte* (autor/resumen/co-lectura ya leídos, que son features de peso
alto), los populares tienen la popularidad misma como señal directa,
pero los de popularidad media no tienen ninguna de las dos cosas a
favor -- no son tan raros como para depender de una coincidencia
específica, ni tan populares como para que el score global los empuje.

Mide, para la franja media (deciles 3-4 de `diagnostico_posicion_popularidad.py`):
1. Cuántas fuentes DISTINTAS proponen al objetivo como candidato
   (`en_als`+`en_popularidad`+`en_genero`+`en_autor_candidato`+
   `en_resumen_candidato`+`en_coleido_candidato`), comparado con los
   extremos (decil 0 y decil 9).
2. Qué features difieren más entre los objetivos que SÍ llegan al top-20
   y los que no, dentro de la franja media -- si ninguna difiere, es
   evidencia de que falta señal, no que el modelo la esté ignorando.
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
SEED = 42

COLUMNAS_EN = ["en_als", "en_popularidad", "en_genero", "en_autor_candidato", "en_resumen_candidato", "en_coleido_candidato"]


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
        fila["posicion"] = posicion
        filas.append(fila)

    df = pd.DataFrame(filas)
    df["decil"] = pd.qcut(df["n_interacciones_libro"], 10, labels=False, duplicates="drop")
    df["n_fuentes"] = df[COLUMNAS_EN].sum(axis=1)
    df["en_top_k"] = df["posicion"] < K

    print(f"\nusuarios con objetivo alcanzable: {len(df)}")

    print("\ncantidad de fuentes distintas que proponen al objetivo, por decil de popularidad:")
    print(df.groupby("decil")["n_fuentes"].mean().to_string(float_format=lambda x: f"{x:.2f}"))

    franja_media = df[df["decil"].isin([3, 4])]
    print(f"\n=== franja media (deciles 3-4, n={len(franja_media)}) ===")
    print(f"n_fuentes promedio: {franja_media['n_fuentes'].mean():.2f}")
    print(f"n_fuentes promedio -- en top-{K}: {franja_media.loc[franja_media['en_top_k'], 'n_fuentes'].mean():.2f}")
    print(f"n_fuentes promedio -- fuera del top-{K}: {franja_media.loc[~franja_media['en_top_k'], 'n_fuentes'].mean():.2f}")

    print(f"\ncomparación de fuentes que proponen al objetivo (franja media, fracción de usuarios):")
    comparacion_en = franja_media.groupby("en_top_k")[COLUMNAS_EN].mean().T
    comparacion_en.columns = ["fuera_del_top_k", "en_top_k"]
    print(comparacion_en.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\ndiferencia de medianas por feature (en_top_k - fuera_del_top_k), franja media, ordenado por |diferencia|:")
    medianas = franja_media.groupby("en_top_k")[FEATURES].median().T
    medianas.columns = ["fuera_del_top_k", "en_top_k"]
    medianas["diferencia"] = medianas["en_top_k"] - medianas["fuera_del_top_k"]
    print(medianas.reindex(medianas["diferencia"].abs().sort_values(ascending=False).index).to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
