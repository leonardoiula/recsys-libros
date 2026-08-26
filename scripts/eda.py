"""Análisis exploratorio de datos (EDA) de interacciones/lectores/libros.

Uso: uv run python scripts/eda.py

Imprime un resumen por stdout y guarda los gráficos en docs/eda/. Los
hallazgos completos quedan resumidos en experiments/eda.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsys.data import load_interacciones, load_lectores, load_libros

ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "docs" / "eda"

# Paleta y estilo tomados de la guía de dataviz del proyecto: un solo hue
# (azul, slot categórico 1) para magnitudes de una sola serie, tinta y grid
# recesivos.
BLUE = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def _style_ax(ax, horizontal_grid: bool = True) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.title.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.set_axisbelow(True)
    if horizontal_grid:
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    else:
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=1)


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  guardado: {path.relative_to(ROOT_DIR)}")


def resumen_general(inter: pd.DataFrame, lectores: pd.DataFrame, libros: pd.DataFrame) -> None:
    print("=== Resumen general ===")
    print(f"interacciones: {len(inter):,}")
    print(f"lectores:      {len(lectores):,}")
    print(f"libros:        {len(libros):,}")

    n_usuarios_activos = inter["id_lector"].nunique()
    n_libros_leidos = inter["id_libro"].nunique()
    sparsity = 1 - len(inter) / (n_usuarios_activos * n_libros_leidos)
    print(f"usuarios con >=1 interaccion: {n_usuarios_activos:,}")
    print(f"libros con >=1 interaccion:   {n_libros_leidos:,}")
    print(f"sparsity de la matriz usuario-libro: {sparsity:.6%}")

    fechas = pd.to_datetime(inter["fecha"], format="%d-%m-%Y", errors="coerce")
    print(
        f"fecha: {fechas.notna().mean():.4%} parseable, "
        f"rango {fechas.min().date()} a {fechas.max().date()}"
    )
    print(f"rating: min={inter['rating'].min()} max={inter['rating'].max()} "
          f"media={inter['rating'].mean():.2f} std={inter['rating'].std():.2f}")


def calidad_datos(inter: pd.DataFrame, lectores: pd.DataFrame, libros: pd.DataFrame) -> None:
    print("\n=== Calidad de datos ===")

    nulos_libros = libros["genero"].isna().mean()
    print(f"libros con metadata nula (titulo/genero/autor/...): {nulos_libros:.2%}")

    libros_en_inter = set(inter["id_libro"].unique())
    libros_con_genero = set(libros.loc[libros["genero"].notna(), "id_libro"])
    cobertura = len(libros_en_inter & libros_con_genero) / len(libros_en_inter)
    print(f"de los libros con >=1 interaccion, % con genero conocido: {cobertura:.2%}")
    print(
        "  -> la metadata nula es casi toda de libros del catálogo que nadie "
        "leyó (scrape incompleto), no afecta al grueso de las interacciones."
    )

    sin_libro = (~inter["id_libro"].isin(libros["id_libro"])).mean()
    sin_lector = (~inter["id_lector"].isin(lectores["id_lector"])).mean()
    print(f"interacciones con id_libro sin match en libros:  {sin_libro:.4%}")
    print(f"interacciones con id_lector sin match en lectores: {sin_lector:.4%}")

    nac_vacio = (lectores["nacimiento"].astype(str).str.strip() == "").mean()
    genero_desconocido = (lectores["genero"] == "-").mean()
    vive_en_vacio = (lectores["vive_en"].astype(str).str.strip() == "").mean()
    print(f"lectores sin nacimiento informado: {nac_vacio:.2%}")
    print(f"lectores con genero '-' (desconocido): {genero_desconocido:.2%}")
    print(f"lectores sin vive_en informado:    {vive_en_vacio:.2%}")


def plot_interacciones_por_usuario(inter: pd.DataFrame) -> None:
    counts = inter.groupby("id_lector").size()
    print("\n=== Interacciones por usuario ===")
    print(counts.describe(percentiles=[0.5, 0.9, 0.99]))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(0, np.log10(counts.max()), 40)
    ax.hist(counts, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("interacciones por usuario (escala log)")
    ax.set_ylabel("cantidad de usuarios")
    ax.set_title("Distribución de interacciones por usuario — cola larga")
    _style_ax(ax)
    _save(fig, "interacciones_por_usuario.png")


def plot_interacciones_por_libro(inter: pd.DataFrame) -> None:
    counts = inter.groupby("id_libro").size()
    print("\n=== Interacciones por libro ===")
    print(counts.describe(percentiles=[0.5, 0.9, 0.99]))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(0, np.log10(counts.max()), 40)
    ax.hist(counts, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("interacciones por libro (escala log)")
    ax.set_ylabel("cantidad de libros")
    ax.set_title("Distribución de interacciones por libro — cola larga")
    _style_ax(ax)
    _save(fig, "interacciones_por_libro.png")


def plot_distribucion_ratings(inter: pd.DataFrame) -> None:
    counts = inter["rating"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color=BLUE, width=0.7)
    ax.set_xticks(counts.index)
    ax.set_xlabel("rating")
    ax.set_ylabel("cantidad de interacciones")
    ax.set_title("Distribución de ratings")
    _style_ax(ax)
    _save(fig, "distribucion_ratings.png")


def plot_generos_mas_leidos(inter: pd.DataFrame, libros: pd.DataFrame, top_n: int = 10) -> None:
    merged = inter.merge(libros[["id_libro", "genero"]], on="id_libro", how="left")
    counts = merged["genero"].value_counts(dropna=True)

    top = counts.head(top_n)
    otros = counts.iloc[top_n:].sum()
    if otros > 0:
        top = pd.concat([top, pd.Series({"Otros": otros})])

    fig, ax = plt.subplots(figsize=(7, 5))
    orden = top.sort_values()
    ax.barh(orden.index, orden.values, color=BLUE)
    ax.set_xlabel("cantidad de interacciones")
    ax.set_title(f"Géneros literarios más leídos (top {top_n} + Otros)")
    _style_ax(ax, horizontal_grid=False)
    _save(fig, "generos_mas_leidos.png")


def plot_rating_promedio_por_genero(
    inter: pd.DataFrame, libros: pd.DataFrame, top_n: int = 10, min_interacciones: int = 100
) -> None:
    merged = inter.merge(libros[["id_libro", "genero"]], on="id_libro", how="left")
    stats = merged.groupby("genero")["rating"].agg(n="count", avg="mean")
    stats = stats[stats["n"] >= min_interacciones].sort_values("n", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    orden = stats["avg"].sort_values()
    ax.barh(orden.index, orden.values, color=BLUE)
    ax.set_xlim(0, 10)
    ax.set_xlabel("rating promedio")
    ax.set_title(f"Rating promedio por género (top {top_n} géneros por volumen)")
    _style_ax(ax, horizontal_grid=False)
    _save(fig, "rating_promedio_por_genero.png")


def plot_lectores_por_genero(lectores: pd.DataFrame) -> None:
    counts = lectores["genero"].replace({"-": "Desconocido"}).value_counts()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=BLUE, width=0.5)
    ax.set_ylabel("cantidad de lectores")
    ax.set_title("Lectores por género")
    _style_ax(ax)
    _save(fig, "lectores_por_genero.png")


def plot_decada_nacimiento(lectores: pd.DataFrame) -> None:
    nacimiento = pd.to_numeric(lectores["nacimiento"], errors="coerce")
    validos = nacimiento.dropna()
    decada = ((validos // 10) * 10).astype(int).astype(str) + "s"
    counts = decada.value_counts().sort_index()

    print("\n=== Década de nacimiento ===")
    print(f"nacimiento informado y numérico: {len(validos) / len(lectores):.2%} de los lectores")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color=BLUE, width=0.6)
    ax.set_ylabel("cantidad de lectores")
    ax.set_title("Década de nacimiento (sobre lectores con dato válido)")
    ax.tick_params(axis="x", rotation=45)
    _style_ax(ax)
    _save(fig, "decada_nacimiento.png")


def main() -> None:
    print("Cargando datos...")
    inter = load_interacciones()
    lectores = load_lectores()
    libros = load_libros()

    resumen_general(inter, lectores, libros)
    calidad_datos(inter, lectores, libros)

    print(f"\nGenerando gráficos en {OUT_DIR.relative_to(ROOT_DIR)}/ ...")
    plot_interacciones_por_usuario(inter)
    plot_interacciones_por_libro(inter)
    plot_distribucion_ratings(inter)
    plot_generos_mas_leidos(inter, libros)
    plot_rating_promedio_por_genero(inter, libros)
    plot_lectores_por_genero(lectores)
    plot_decada_nacimiento(lectores)

    print("\nListo.")


if __name__ == "__main__":
    main()
