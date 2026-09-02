"""Arma el csv de entrega para Kaggle."""

from __future__ import annotations

import argparse
import gc
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from recsys.data import (
    libros_leidos_por_usuario,
    load_interacciones,
    load_lectores,
    load_libros,
    split_train_val,
)
from recsys.models.als import fit_als
from recsys.models.als import recomendar_por_usuario as recomendar_por_usuario_als
from recsys.models.popularity import fit_popularity
from recsys.models.popularity_segmentada import (
    fit_popularity_por_franja_nacimiento,
    fit_popularity_por_genero,
    franja_nacimiento_por_usuario,
    genero_preferido_por_usuario,
    recomendar_por_usuario,
)
from recsys.models.ranker import (
    armar_dataset_entrenamiento_por_lotes,
    calcular_features_auxiliares,
    fit_ranker,
    recomendar_por_usuario_por_lotes,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
EJEMPLO_PATH = ROOT_DIR / "data" / "raw" / "ejemplo.csv"
SUBMISSIONS_DIR = ROOT_DIR / "outputs" / "submissions"


def _recomendaciones_popularity(usuarios: list, k: int) -> dict:
    """v0: mismo ranking global de popularidad para todos los usuarios."""
    interacciones = load_interacciones()
    libros_leidos = libros_leidos_por_usuario(interacciones)
    ranking_global = fit_popularity(interacciones)["id_libro"].tolist()

    return {
        id_lector: [
            libro
            for libro in ranking_global
            if libro not in libros_leidos.get(id_lector, set())
        ][:k]
        for id_lector in usuarios
    }


def _recomendaciones_popularity_segmentada(usuarios: list, k: int) -> dict:
    """v1: ranking por usuario con fallback género -> franja de nacimiento -> global."""
    interacciones = load_interacciones()
    lectores = load_lectores()
    libros = load_libros()
    libros_leidos = libros_leidos_por_usuario(interacciones)

    return recomendar_por_usuario(
        usuarios=usuarios,
        ranking_por_genero=fit_popularity_por_genero(interacciones, libros),
        genero_por_usuario=genero_preferido_por_usuario(interacciones, libros),
        ranking_por_franja=fit_popularity_por_franja_nacimiento(interacciones, lectores),
        franja_por_usuario=franja_nacimiento_por_usuario(lectores),
        ranking_global=fit_popularity(interacciones)["id_libro"].tolist(),
        libros_leidos=libros_leidos,
        k=k,
    )


def _recomendaciones_als(usuarios: list, k: int) -> dict:
    """v2: filtrado colaborativo ALS sobre ratings, fallback a popularidad global.

    Se evaluó (y se descartó) rutear usuarios livianos hacia la
    popularidad por género de v1 en vez de ALS, bajo la hipótesis de que
    el embedding de ALS para poca actividad sería poco confiable. Bajo el
    split corregido (temporal, n_val=1) los datos no lo respaldan: ALS le
    gana a género en *todos* los buckets de actividad medidos, incluso
    con una sola interacción de historial (ver
    `recsys.models.als.recomendar_hibrido` y `experiments/bitacora.md`
    para el detalle). Se mantiene ALS puro acá.
    """
    interacciones = load_interacciones()
    libros_leidos = libros_leidos_por_usuario(interacciones)
    modelo, matriz, fila_por_usuario, libros_por_columna = fit_als(interacciones)

    return recomendar_por_usuario_als(
        usuarios=usuarios,
        modelo=modelo,
        matriz_usuario_libro=matriz,
        fila_por_usuario=fila_por_usuario,
        libros_por_columna=libros_por_columna,
        ranking_global=fit_popularity(interacciones)["id_libro"].tolist(),
        libros_leidos=libros_leidos,
        k=k,
    )


N_POR_FUENTE_RANKER = 150
N_POR_AUTOR_RANKER = 20


def _recomendaciones_ranker(usuarios: list, k: int) -> dict:
    """v3: ranker de dos etapas -- candidatos de ALS + popularidad por
    género + popularidad global + libros de autores ya leídos,
    reordenados por un `LGBMRanker` (LightGBM) entrenado para combinar
    esas señales.

    Validado con validación cruzada sobre 3 seeds antes de wirear acá:
    le ganó a ALS solo en los 3 seeds (+3.9% de NDCG@20 en promedio, con
    menor desvío entre seeds que ALS) -- ver `scripts/evaluate_ranker.py`
    y `experiments/bitacora.md`.

    Las señales de etapa 1 (ALS/popularidad/género/`calcular_features_auxiliares`)
    se fittean sobre `train_candidatos` para entrenar el ranker (evita que
    vea, como features, scores calculados con la misma etiqueta que tiene
    que predecir) pero se REFITEAN sobre `interacciones` completo -- todos
    los datos disponibles, no hay un "futuro" que reservar acá como en la
    evaluación local -- para generar los candidatos finales de la
    submission real. Confirmado localmente antes de wirear acá
    (`scripts/comparar_refit_etapa1.py`, `ranker.preparar_pipeline(...,
    refit_para_test=True)`): +12.4% de NDCG@20 en promedio, positivo en
    los 3 seeds, muy por encima del desvío entre seeds -- refitear con la
    interacción más reciente de cada usuario (que antes solo se usaba
    para filtrar libros ya leídos, nunca como señal) aporta bastante. Ver
    `experiments/bitacora.md`.

    Tanto el armado del dataset de entrenamiento como la generación de
    los candidatos finales van por lotes de usuarios
    (`armar_dataset_entrenamiento_por_lotes`/`recomendar_por_usuario_por_lotes`,
    ver `TAMANO_LOTE_USUARIOS` en `ranker.py`) -- generar la unión de
    candidatos de los ~11k lectores de una sola vez llegó a fallar por
    `ArrayMemoryError` pese a tener RAM de sobra en la máquina.
    """
    interacciones = load_interacciones()
    libros = load_libros()
    lectores = load_lectores()
    train_candidatos, train_ranker = split_train_val(interacciones, n_val=1, seed=42)

    libros_leidos_stage1 = libros_leidos_por_usuario(train_candidatos)
    n_interacciones_por_usuario = train_candidatos.groupby("id_lector").size().to_dict()
    stats_popularidad = fit_popularity(train_candidatos)
    stats_por_genero = fit_popularity_por_genero(train_candidatos, libros)
    genero_por_usuario = genero_preferido_por_usuario(train_candidatos, libros)
    modelo_als, matriz, fila_por_usuario, libros_por_columna = fit_als(train_candidatos)
    features_auxiliares = calcular_features_auxiliares(
        train_candidatos, libros, lectores, matriz, fila_por_usuario, libros_por_columna
    )

    args_candidatos = dict(
        modelo_als=modelo_als,
        matriz_usuario_libro=matriz,
        fila_por_usuario=fila_por_usuario,
        libros_por_columna=libros_por_columna,
        stats_popularidad=stats_popularidad,
        stats_por_genero=stats_por_genero,
        genero_por_usuario=genero_por_usuario,
        n_interacciones_por_usuario=n_interacciones_por_usuario,
        features_auxiliares=features_auxiliares,
        n_por_fuente=N_POR_FUENTE_RANKER,
        n_por_autor=N_POR_AUTOR_RANKER,
    )

    usuarios_ranker = train_ranker["id_lector"].unique().tolist()
    X, y, group = armar_dataset_entrenamiento_por_lotes(
        usuarios_ranker,
        train_ranker[["id_lector", "id_libro"]],
        libros_leidos_stage1,
        args_candidatos,
        n_por_fuente=N_POR_FUENTE_RANKER,
        n_por_autor=N_POR_AUTOR_RANKER,
    )
    modelo_ranker = fit_ranker(X, y, group)

    # La etapa 1 fiteada sobre train_candidatos ya no hace falta -- se
    # refitea sobre todos los datos más abajo. Liberarla antes evita
    # tener las dos versiones (cada una con su propia matriz de
    # co-ocurrencia/TF-IDF) vivas a la vez (mismo ajuste que
    # `ranker.preparar_pipeline` con `refit_para_test=True`).
    del args_candidatos, features_auxiliares, matriz, modelo_als, fila_por_usuario, libros_por_columna
    gc.collect()

    # Refit de etapa 1 sobre TODOS los datos (no solo train_candidatos)
    # para generar los candidatos finales -- ver docstring.
    n_interacciones_por_usuario_completo = interacciones.groupby("id_lector").size().to_dict()
    stats_popularidad_completo = fit_popularity(interacciones)
    ranking_global = stats_popularidad_completo["id_libro"].tolist()
    stats_por_genero_completo = fit_popularity_por_genero(interacciones, libros)
    genero_por_usuario_completo = genero_preferido_por_usuario(interacciones, libros)
    modelo_als_completo, matriz_completo, fila_por_usuario_completo, libros_por_columna_completo = fit_als(
        interacciones
    )
    features_auxiliares_completo = calcular_features_auxiliares(
        interacciones, libros, lectores, matriz_completo, fila_por_usuario_completo, libros_por_columna_completo
    )
    args_candidatos_finales = dict(
        modelo_als=modelo_als_completo,
        matriz_usuario_libro=matriz_completo,
        fila_por_usuario=fila_por_usuario_completo,
        libros_por_columna=libros_por_columna_completo,
        stats_popularidad=stats_popularidad_completo,
        stats_por_genero=stats_por_genero_completo,
        genero_por_usuario=genero_por_usuario_completo,
        n_interacciones_por_usuario=n_interacciones_por_usuario_completo,
        features_auxiliares=features_auxiliares_completo,
        n_por_fuente=N_POR_FUENTE_RANKER,
        n_por_autor=N_POR_AUTOR_RANKER,
    )

    libros_leidos_completo = libros_leidos_por_usuario(interacciones)
    return recomendar_por_usuario_por_lotes(
        usuarios,
        modelo_ranker,
        libros_leidos=libros_leidos_completo,
        ranking_global=ranking_global,
        args_candidatos=args_candidatos_finales,
        k=k,
    )


# Cada modelo mapea a una función (usuarios, k) -> {id_lector: [id_libro, ...]}
# ya entrenada con todos los datos y con los libros ya leídos filtrados.
MODELOS = {
    "popularity": _recomendaciones_popularity,
    "popularity_segmentada": _recomendaciones_popularity_segmentada,
    "als": _recomendaciones_als,
    "ranker": _recomendaciones_ranker,
}


def armar_submission(ejemplo_df: pd.DataFrame, recomendaciones: dict) -> pd.DataFrame:
    """Arma el DataFrame de entrega a partir de recomendaciones ya armadas por usuario.

    Para cada usuario de `ejemplo_df` (en el orden en que aparecen) toma
    tantas filas de `recomendaciones[id_lector]` como filas tenga ese
    usuario en `ejemplo_df` (su k), preservando el orden del ranking.
    """
    filas = []
    for id_lector, grupo in ejemplo_df.groupby("id_lector", sort=False):
        k = len(grupo)
        candidatos = recomendaciones.get(id_lector, [])[:k]
        filas.extend({"id_lector": id_lector, "id_libro": libro} for libro in candidatos)

    return pd.DataFrame(filas, columns=["id_lector", "id_libro"])


def _nombre_submission(model: str, tag: str | None) -> str:
    """Arma un nombre de archivo único con timestamp -- nunca pisa una
    submission anterior (cada corrida es una versión distinta, y las que
    ya se subieron a Kaggle no se pueden regenerar del código actual sin
    volver al commit correspondiente)."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sufijo_tag = ""
    if tag:
        tag_limpio = re.sub(r"[^a-zA-Z0-9._-]+", "-", tag).strip("-")
        if tag_limpio:
            sufijo_tag = f"_{tag_limpio}"

    base = f"{model}_{timestamp}{sufijo_tag}"
    nombre = f"{base}.csv"
    contador = 2
    while (SUBMISSIONS_DIR / nombre).exists():
        nombre = f"{base}-{contador}.csv"
        contador += 1
    return nombre


def generar_submission(model: str, tag: str | None = None) -> Path:
    """Entrena `model` con todos los datos y guarda el csv de entrega.

    El archivo se guarda como `{model}_{timestamp}[_{tag}].csv` -- nunca
    con el nombre pelado `{model}.csv`, para no pisar submissions
    anteriores (en particular, las que ya se subieron a Kaggle).
    """
    if model not in MODELOS:
        raise ValueError(f"Modelo desconocido: {model!r}. Opciones: {sorted(MODELOS)}")

    ejemplo_df = pd.read_csv(EJEMPLO_PATH)
    usuarios = ejemplo_df["id_lector"].unique().tolist()
    k = int(ejemplo_df.groupby("id_lector").size().max())

    recomendaciones = MODELOS[model](usuarios, k)
    submission = armar_submission(ejemplo_df, recomendaciones)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS_DIR / _nombre_submission(model, tag)
    submission.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un csv de entrega para Kaggle.")
    parser.add_argument("--model", choices=sorted(MODELOS), required=True)
    parser.add_argument("--tag", default=None, help="Sufijo descriptivo opcional para el nombre del archivo.")
    args = parser.parse_args()

    out_path = generar_submission(args.model, tag=args.tag)
    print(f"Submission guardada en {out_path}")


if __name__ == "__main__":
    main()
