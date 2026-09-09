"""
EDA de comics/data/raw/comics.db: genera un informe HTML autocontenido
(sin dependencias externas -- SVG inline, un poco de JS para tooltips, cero
llamadas a CDN) en comics/docs/eda/informe_eda.html.

Uso: uv run python comics/scripts/generar_eda.py

Todas las consultas son SQL directo (sqlite3/pandas.read_sql_query) sobre las
4 tablas de comics.db; los gráficos son SVG generado a mano siguiendo la guía
de estilo del proyecto (paleta y specs de marca en las referencias de la skill
de dataviz), no una librería de charting.
"""

from __future__ import annotations

import html
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from comics_recsys.evaluation import BINS_ACTIVIDAD_DEFAULT  # noqa: E402

COMICS_DIR = Path(__file__).resolve().parents[1]
DB_PATH = COMICS_DIR / "data" / "raw" / "comics.db"
OUT_PATH = COMICS_DIR / "docs" / "eda" / "informe_eda.html"

# Medido a mano el 2026-09-05 con comics/scripts/contar_poblacion_total.py
# (recorre solo páginas de listado semanal, no cada issue -- ver comics/CLAUDE.md).
TOTAL_ISSUES_CATALOGADOS_SITIO = 67_035

# ---------------------------------------------------------------------------
# Paleta y specs (ver skill de dataviz, references/palette.md y
# references/marks-and-anatomy.md) -- solo modo claro, para mantener acotado
# el alcance de este informe puntual.
# ---------------------------------------------------------------------------
COLOR_SERIE_1 = "#2a78d6"  # blue -- serie por default (magnitud, secuencial)
COLOR_SERIE_2 = "#eb6834"  # orange -- segunda serie cuando hace falta comparar 2
SURFACE = "#fcfcfb"
PAGE_PLANE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BORDER = "rgba(11,11,11,0.10)"


def consultar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def compacto(n: float) -> str:
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,.0f}" if n == int(n) else f"{n:,.2f}"


def con_comas(n: float) -> str:
    return f"{n:,.0f}"


# ---------------------------------------------------------------------------
# Componentes SVG
# ---------------------------------------------------------------------------


def _tip_attr(texto: str) -> str:
    return f'data-tip="{html.escape(texto, quote=True)}"'


def _path_barra_vertical(x0: float, ancho: float, y_top: float, y_base: float, radio: float) -> str:
    r = min(radio, ancho / 2, max(y_base - y_top, 0))
    if y_base - y_top <= 0:
        return ""
    if r <= 0:
        return f"M {x0},{y_base} L {x0},{y_top} L {x0 + ancho},{y_top} L {x0 + ancho},{y_base} Z"
    return (
        f"M {x0},{y_base} L {x0},{y_top + r} Q {x0},{y_top} {x0 + r},{y_top} "
        f"L {x0 + ancho - r},{y_top} Q {x0 + ancho},{y_top} {x0 + ancho},{y_top + r} "
        f"L {x0 + ancho},{y_base} Z"
    )


def _path_barra_horizontal(x0: float, y0: float, ancho: float, alto: float, radio: float) -> str:
    r = min(radio, ancho, alto / 2)
    if ancho <= 0:
        return ""
    if r <= 0:
        return f"M {x0},{y0} L {x0 + ancho},{y0} L {x0 + ancho},{y0 + alto} L {x0},{y0 + alto} Z"
    return (
        f"M {x0},{y0} L {x0 + ancho - r},{y0} Q {x0 + ancho},{y0} {x0 + ancho},{y0 + r} "
        f"L {x0 + ancho},{y0 + alto - r} Q {x0 + ancho},{y0 + alto} {x0 + ancho - r},{y0 + alto} "
        f"L {x0},{y0 + alto} Z"
    )


def svg_barras_verticales(
    datos: list[tuple[str, float]],
    *,
    titulo: str,
    subtitulo: str = "",
    color: str = COLOR_SERIE_1,
    ancho: int = 680,
    alto: int = 300,
    formato_valor=con_comas,
    formato_tip=None,
) -> str:
    m_izq, m_der, m_arr, m_ab = 56, 16, 16, 44
    aw, ah = ancho - m_izq - m_der, alto - m_arr - m_ab
    valor_max = max((v for _, v in datos), default=1) or 1
    n = len(datos)
    slot = aw / n if n else aw
    ancho_barra = min(24, slot * 0.62)
    gap_inset = (slot - ancho_barra) / 2

    y0 = m_arr

    # gridlines: 4 pasos limpios
    n_gridlines = 4
    gridlines = []
    for i in range(n_gridlines + 1):
        frac = i / n_gridlines
        y = y0 + ah * (1 - frac)
        valor_tick = valor_max * frac
        gridlines.append(
            f'<line x1="{m_izq}" y1="{y:.1f}" x2="{m_izq + aw}" y2="{y:.1f}" '
            f'stroke="{GRIDLINE}" stroke-width="1"/>'
            f'<text x="{m_izq - 8}" y="{y + 4:.1f}" text-anchor="end" class="viz-tick">{formato_valor(valor_tick)}</text>'
        )

    barras = []
    etiquetas = []
    for i, (label, valor) in enumerate(datos):
        x0 = m_izq + i * slot + gap_inset
        alto_barra = (valor / valor_max) * ah if valor_max else 0
        y_top = y0 + ah - alto_barra
        y_base = y0 + ah
        path = _path_barra_vertical(x0, ancho_barra, y_top, y_base, 4)
        tip = formato_tip(label, valor) if formato_tip else f"{label}: {con_comas(valor)}"
        if path:
            barras.append(
                f'<path d="{path}" fill="{color}"/>'
                f'<rect x="{x0 - 2:.1f}" y="{m_arr}" width="{ancho_barra + 4:.1f}" height="{ah:.1f}" '
                f'fill="transparent" class="viz-hit" {_tip_attr(tip)} '
                f'onmouseenter="vizShow(event)" onmousemove="vizMove(event)" onmouseleave="vizHide()"/>'
            )
        etiquetas.append(
            f'<text x="{x0 + ancho_barra / 2:.1f}" y="{y_base + 18:.1f}" text-anchor="middle" class="viz-tick">{html.escape(str(label))}</text>'
        )

    baseline = f'<line x1="{m_izq}" y1="{y0 + ah:.1f}" x2="{m_izq + aw}" y2="{y0 + ah:.1f}" stroke="{BASELINE}" stroke-width="1"/>'

    return _envolver_chart(
        titulo,
        subtitulo,
        ancho,
        alto,
        "".join(gridlines) + baseline + "".join(barras) + "".join(etiquetas),
    )


def svg_barras_horizontales(
    datos: list[tuple[str, float]],
    *,
    titulo: str,
    subtitulo: str = "",
    color: str = COLOR_SERIE_1,
    ancho: int = 680,
    formato_valor=con_comas,
    formato_tip=None,
) -> str:
    m_izq, m_der, m_arr = 160, 64, 16
    fila_alto = 30
    alto_barra = min(22, fila_alto - 8)
    n = len(datos)
    alto = m_arr + n * fila_alto + 12
    aw = ancho - m_izq - m_der
    valor_max = max((v for _, v in datos), default=1) or 1

    partes = []
    for i, (label, valor) in enumerate(datos):
        y0 = m_arr + i * fila_alto + (fila_alto - alto_barra) / 2
        ancho_barra = (valor / valor_max) * aw if valor_max else 0
        path = _path_barra_horizontal(m_izq, y0, ancho_barra, alto_barra, 4)
        tip = formato_tip(label, valor) if formato_tip else f"{label}: {con_comas(valor)}"
        etiqueta_valor_x = m_izq + ancho_barra + 8
        partes.append(
            f'<text x="{m_izq - 10}" y="{y0 + alto_barra / 2 + 4:.1f}" text-anchor="end" class="viz-tick">{html.escape(str(label))}</text>'
        )
        if path:
            partes.append(
                f'<path d="{path}" fill="{color}"/>'
                f'<text x="{etiqueta_valor_x:.1f}" y="{y0 + alto_barra / 2 + 4:.1f}" class="viz-tick-strong">{formato_valor(valor)}</text>'
                f'<rect x="{m_izq}" y="{y0 - 3:.1f}" width="{max(ancho_barra, 4) + 60:.1f}" height="{alto_barra + 6:.1f}" '
                f'fill="transparent" class="viz-hit" {_tip_attr(tip)} '
                f'onmouseenter="vizShow(event)" onmousemove="vizMove(event)" onmouseleave="vizHide()"/>'
            )

    return _envolver_chart(titulo, subtitulo, ancho, alto, "".join(partes))


def svg_barras_agrupadas(
    categorias: list[str],
    series: dict[str, list[float]],
    colores: dict[str, str],
    *,
    titulo: str,
    subtitulo: str = "",
    ancho: int = 680,
    alto: int = 300,
    formato_valor=con_comas,
) -> str:
    m_izq, m_der, m_arr, m_ab = 56, 16, 16, 44
    aw, ah = ancho - m_izq - m_der, alto - m_arr - m_ab
    nombres_series = list(series.keys())
    valor_max = max((v for vals in series.values() for v in vals), default=1) or 1
    n = len(categorias)
    slot = aw / n if n else aw
    n_series = len(nombres_series)
    ancho_barra = min(20, slot * 0.8 / n_series)
    gap_series = 2
    ancho_grupo = ancho_barra * n_series + gap_series * (n_series - 1)
    y0 = m_arr

    n_gridlines = 4
    gridlines = []
    for i in range(n_gridlines + 1):
        frac = i / n_gridlines
        y = y0 + ah * (1 - frac)
        valor_tick = valor_max * frac
        gridlines.append(
            f'<line x1="{m_izq}" y1="{y:.1f}" x2="{m_izq + aw}" y2="{y:.1f}" stroke="{GRIDLINE}" stroke-width="1"/>'
            f'<text x="{m_izq - 8}" y="{y + 4:.1f}" text-anchor="end" class="viz-tick">{formato_valor(valor_tick)}</text>'
        )

    barras = []
    etiquetas = []
    for i, cat in enumerate(categorias):
        x_grupo = m_izq + i * slot + (slot - ancho_grupo) / 2
        for j, nombre in enumerate(nombres_series):
            valor = series[nombre][i]
            x0 = x_grupo + j * (ancho_barra + gap_series)
            alto_barra = (valor / valor_max) * ah if valor_max else 0
            y_top = y0 + ah - alto_barra
            path = _path_barra_vertical(x0, ancho_barra, y_top, y0 + ah, 3)
            tip = f"{cat} · {nombre}: {con_comas(valor)}"
            if path:
                barras.append(
                    f'<path d="{path}" fill="{colores[nombre]}"/>'
                    f'<rect x="{x0 - 1:.1f}" y="{m_arr}" width="{ancho_barra + 2:.1f}" height="{ah:.1f}" '
                    f'fill="transparent" class="viz-hit" {_tip_attr(tip)} '
                    f'onmouseenter="vizShow(event)" onmousemove="vizMove(event)" onmouseleave="vizHide()"/>'
                )
        etiquetas.append(
            f'<text x="{x_grupo + ancho_grupo / 2:.1f}" y="{y0 + ah + 18:.1f}" text-anchor="middle" class="viz-tick">{html.escape(cat)}</text>'
        )

    baseline = f'<line x1="{m_izq}" y1="{y0 + ah:.1f}" x2="{m_izq + aw}" y2="{y0 + ah:.1f}" stroke="{BASELINE}" stroke-width="1"/>'
    leyenda = "".join(
        f'<span class="viz-legend-item"><span class="viz-swatch" style="background:{colores[n]}"></span>{html.escape(n)}</span>'
        for n in nombres_series
    )

    contenido = _envolver_chart(
        titulo, subtitulo, ancho, alto, "".join(gridlines) + baseline + "".join(barras) + "".join(etiquetas)
    )
    return contenido.replace("</div>\n", f'<div class="viz-legend">{leyenda}</div></div>\n', 1)


def svg_lineas(
    puntos: list[tuple[str, float]],
    *,
    titulo: str,
    subtitulo: str = "",
    color: str = COLOR_SERIE_1,
    ancho: int = 680,
    alto: int = 280,
    formato_valor=con_comas,
    marcar_cada: int = 6,
) -> str:
    m_izq, m_der, m_arr, m_ab = 56, 16, 16, 44
    aw, ah = ancho - m_izq - m_der, alto - m_arr - m_ab
    n = len(puntos)
    valor_max = max((v for _, v in puntos), default=1) or 1
    y0 = m_arr

    def xy(i, v):
        x = m_izq + (i / max(n - 1, 1)) * aw
        y = y0 + ah - (v / valor_max) * ah
        return x, y

    n_gridlines = 4
    gridlines = []
    for i in range(n_gridlines + 1):
        frac = i / n_gridlines
        y = y0 + ah * (1 - frac)
        valor_tick = valor_max * frac
        gridlines.append(
            f'<line x1="{m_izq}" y1="{y:.1f}" x2="{m_izq + aw}" y2="{y:.1f}" stroke="{GRIDLINE}" stroke-width="1"/>'
            f'<text x="{m_izq - 8}" y="{y + 4:.1f}" text-anchor="end" class="viz-tick">{formato_valor(valor_tick)}</text>'
        )

    coords = [xy(i, v) for i, (_, v) in enumerate(puntos)]
    linea = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area = linea + f" L {coords[-1][0]:.1f},{y0 + ah:.1f} L {coords[0][0]:.1f},{y0 + ah:.1f} Z"

    marcas = []
    for i, ((label, valor), (x, y)) in enumerate(zip(puntos, coords)):
        tip = f"{label}: {con_comas(valor)}"
        visible = i == 0 or i == n - 1 or i % marcar_cada == 0 or valor == valor_max
        marcas.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{5 if visible else 3}" fill="{color}" '
            f'stroke="{SURFACE}" stroke-width="2" class="viz-hit" {_tip_attr(tip)} '
            f'onmouseenter="vizShow(event)" onmousemove="vizMove(event)" onmouseleave="vizHide()"/>'
        )

    n_labels_x = min(8, n)
    paso_label = max(1, n // n_labels_x)
    etiquetas_x = "".join(
        f'<text x="{coords[i][0]:.1f}" y="{y0 + ah + 18:.1f}" text-anchor="middle" class="viz-tick">{html.escape(puntos[i][0])}</text>'
        for i in range(0, n, paso_label)
    )

    baseline = f'<line x1="{m_izq}" y1="{y0 + ah:.1f}" x2="{m_izq + aw}" y2="{y0 + ah:.1f}" stroke="{BASELINE}" stroke-width="1"/>'
    contenido = (
        "".join(gridlines)
        + baseline
        + f'<path d="{area}" fill="{color}" opacity="0.1"/>'
        + f'<path d="{linea}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        + "".join(marcas)
        + etiquetas_x
    )
    return _envolver_chart(titulo, subtitulo, ancho, alto, contenido)


def _envolver_chart(titulo: str, subtitulo: str, ancho: int, alto: int, contenido_svg: str) -> str:
    sub = f'<p class="viz-subtitulo">{html.escape(subtitulo)}</p>' if subtitulo else ""
    return (
        f'<div class="viz-card">'
        f'<h3 class="viz-titulo">{html.escape(titulo)}</h3>{sub}'
        f'<svg viewBox="0 0 {ancho} {alto}" width="100%" style="max-width:{ancho}px" role="img" aria-label="{html.escape(titulo)}">'
        f"{contenido_svg}"
        f"</svg>"
        f"</div>\n"
    )


def stat_tile(label: str, valor: str, sub: str = "") -> str:
    sub_html = f'<div class="viz-stat-sub">{html.escape(sub)}</div>' if sub else ""
    return (
        f'<div class="viz-stat-tile"><div class="viz-stat-label">{html.escape(label)}</div>'
        f'<div class="viz-stat-value">{html.escape(valor)}</div>{sub_html}</div>'
    )


def tabla_html(columnas: list[str], filas: list[tuple], *, titulo: str = "") -> str:
    thead = "".join(f"<th>{html.escape(c)}</th>" for c in columnas)
    tbody = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in fila) + "</tr>" for fila in filas
    )
    titulo_html = f'<h3 class="viz-titulo">{html.escape(titulo)}</h3>' if titulo else ""
    return (
        f'<div class="viz-card">{titulo_html}'
        f'<table class="viz-tabla"><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table></div>\n'
    )


# ---------------------------------------------------------------------------
# EDA propiamente dicho
# ---------------------------------------------------------------------------


def main() -> None:
    conn = sqlite3.connect(DB_PATH)

    n_comics = consultar(conn, "SELECT COUNT(*) n FROM comics").iloc[0]["n"]
    n_usuarios = consultar(conn, "SELECT COUNT(*) n FROM usuarios").iloc[0]["n"]
    n_interacciones = consultar(conn, "SELECT COUNT(*) n FROM interacciones").iloc[0]["n"]
    n_critic = consultar(conn, "SELECT COUNT(*) n FROM critic_reviews").iloc[0]["n"]
    rango_fechas = consultar(
        conn, "SELECT MIN(fecha) desde, MAX(fecha) hasta FROM interacciones WHERE fecha IS NOT NULL"
    ).iloc[0]

    densidad = n_interacciones / (n_comics * n_usuarios) * 100
    cobertura_sitio = n_comics / TOTAL_ISSUES_CATALOGADOS_SITIO * 100

    kpis = "".join(
        [
            stat_tile("Comics", con_comas(n_comics), "filas en la tabla comics"),
            stat_tile("Usuarios", con_comas(n_usuarios)),
            stat_tile("Interacciones", con_comas(n_interacciones), "reviews de usuario"),
            stat_tile("Critic reviews", con_comas(n_critic), "tabla separada, no son interacciones"),
            stat_tile("Densidad de la matriz", f"{densidad:.2f}%", "interacciones / (comics × usuarios)"),
            stat_tile(
                "Cobertura del sitio",
                f"{cobertura_sitio:.1f}%",
                f"sobre {con_comas(TOTAL_ISSUES_CATALOGADOS_SITIO)} issues catalogados (medido 2026-09-05)",
            ),
            stat_tile("Rango temporal", f"{rango_fechas['desde']} → {rango_fechas['hasta']}", "fecha de las reviews"),
        ]
    )

    # -- editoriales ---------------------------------------------------------
    editoriales_comics = consultar(
        conn, "SELECT editorial, COUNT(*) n FROM comics GROUP BY editorial ORDER BY n DESC LIMIT 10"
    )
    chart_editoriales = svg_barras_horizontales(
        list(editoriales_comics.itertuples(index=False, name=None)),
        titulo="Top 10 editoriales por cantidad de comics",
        subtitulo="SELECT editorial, COUNT(*) FROM comics GROUP BY editorial",
    )

    # -- interacciones por usuario (actividad) -------------------------------
    inter_por_usuario = consultar(
        conn, "SELECT id_usuario, COUNT(*) n FROM interacciones GROUP BY id_usuario"
    )
    bucket = pd.cut(inter_por_usuario["n"], bins=BINS_ACTIVIDAD_DEFAULT, right=False)
    conteo_bucket = bucket.value_counts().sort_index()
    datos_actividad = [(str(intervalo), int(v)) for intervalo, v in conteo_bucket.items()]
    chart_actividad = svg_barras_verticales(
        datos_actividad,
        titulo="Usuarios por bucket de actividad",
        subtitulo=(
            f"SELECT id_usuario, COUNT(*) FROM interacciones GROUP BY id_usuario -- "
            f"mediana={inter_por_usuario['n'].median():.0f}, media={inter_por_usuario['n'].mean():.1f}, "
            f"máximo={inter_por_usuario['n'].max()} (mismos buckets que recsys.evaluation.BINS_ACTIVIDAD_DEFAULT)"
        ),
    )

    # -- interacciones por comic (long tail) ----------------------------------
    inter_por_comic = consultar(conn, "SELECT id_comic, COUNT(*) n FROM interacciones GROUP BY id_comic")
    comics_sin_reviews = n_comics - len(inter_por_comic)
    bins_comic = [0, 1, 2, 3, 5, 10, 20, 50, float("inf")]
    bucket_comic = pd.cut(inter_por_comic["n"], bins=bins_comic, right=False)
    conteo_bucket_comic = bucket_comic.value_counts().sort_index()
    datos_comic = [("0 (sin reviews)", int(comics_sin_reviews))] + [
        (str(intervalo), int(v)) for intervalo, v in conteo_bucket_comic.items()
    ]
    chart_long_tail = svg_barras_verticales(
        datos_comic,
        titulo="Comics por cantidad de reviews de usuario recibidas",
        subtitulo=(
            f"{comics_sin_reviews} de {n_comics} comics ({comics_sin_reviews / n_comics * 100:.1f}%) "
            f"no tienen ninguna review de usuario -- long tail real, no es un error de scraping"
        ),
    )

    # -- serie temporal mensual -----------------------------------------------
    mensual = consultar(
        conn,
        """
        SELECT strftime('%Y-%m', fecha) mes, COUNT(*) n
        FROM interacciones
        WHERE fecha IS NOT NULL
        GROUP BY mes ORDER BY mes
        """,
    )
    chart_temporal = svg_lineas(
        list(mensual.itertuples(index=False, name=None)),
        titulo="Interacciones de usuario por mes",
        subtitulo="SELECT strftime('%Y-%m', fecha), COUNT(*) FROM interacciones GROUP BY 1 ORDER BY 1",
        marcar_cada=6,
    )

    # -- distribución de ratings: usuarios vs críticos ------------------------
    ratings_usuario = consultar(
        conn, "SELECT CAST(ROUND(rating) AS INT) r, COUNT(*) n FROM interacciones GROUP BY r ORDER BY r"
    )
    ratings_critico = consultar(
        conn, "SELECT CAST(ROUND(rating) AS INT) r, COUNT(*) n FROM critic_reviews GROUP BY r ORDER BY r"
    )
    escala = list(range(0, 11))
    mapa_usuario = dict(zip(ratings_usuario["r"], ratings_usuario["n"]))
    mapa_critico = dict(zip(ratings_critico["r"], ratings_critico["n"]))
    chart_ratings = svg_barras_agrupadas(
        [str(r) for r in escala],
        {
            "Usuarios (interacciones)": [int(mapa_usuario.get(r, 0)) for r in escala],
            "Críticos (critic_reviews)": [int(mapa_critico.get(r, 0)) for r in escala],
        },
        {"Usuarios (interacciones)": COLOR_SERIE_1, "Críticos (critic_reviews)": COLOR_SERIE_2},
        titulo="Distribución de ratings (redondeados a entero, escala 0-10)",
        subtitulo="Dos GROUP BY separados sobre interacciones y critic_reviews -- no se mezclan en ninguna tabla",
    )

    # -- correlación critico vs usuario por comic -----------------------------
    prom_usuario = consultar(
        conn, "SELECT id_comic, AVG(rating) avg_u, COUNT(*) n_u FROM interacciones GROUP BY id_comic HAVING n_u >= 3"
    )
    prom_critico = consultar(
        conn, "SELECT id_comic, AVG(rating) avg_c, COUNT(*) n_c FROM critic_reviews GROUP BY id_comic HAVING n_c >= 3"
    )
    comparacion = prom_usuario.merge(prom_critico, on="id_comic")
    correlacion = comparacion["avg_u"].corr(comparacion["avg_c"]) if len(comparacion) > 1 else float("nan")

    # -- overlap: series distintas por usuario --------------------------------
    series_por_usuario = consultar(
        conn,
        """
        SELECT i.id_usuario, c.serie
        FROM interacciones i JOIN comics c ON c.id_comic = i.id_comic
        """,
    )
    n_series_por_usuario = series_por_usuario.groupby("id_usuario")["serie"].nunique()
    pct_multi_serie = (n_series_por_usuario >= 2).mean() * 100

    # -- top comics más calificados --------------------------------------------
    top_comics = consultar(
        conn,
        """
        SELECT c.titulo, c.editorial, COUNT(*) n_reviews, ROUND(AVG(i.rating), 2) rating_promedio
        FROM interacciones i JOIN comics c ON c.id_comic = i.id_comic
        GROUP BY i.id_comic
        ORDER BY n_reviews DESC
        LIMIT 10
        """,
    )
    tabla_top_comics = tabla_html(
        ["Título", "Editorial", "N° reviews", "Rating promedio"],
        list(top_comics.itertuples(index=False, name=None)),
        titulo="Top 10 comics con más reviews de usuario",
    )

    # -- metadata faltante ------------------------------------------------------
    faltantes = consultar(
        conn,
        """
        SELECT
            SUM(CASE WHEN escritor IS NULL THEN 1 ELSE 0 END) sin_escritor,
            SUM(CASE WHEN dibujante IS NULL THEN 1 ELSE 0 END) sin_dibujante,
            SUM(CASE WHEN anio_edicion IS NULL THEN 1 ELSE 0 END) sin_anio,
            SUM(CASE WHEN img_src IS NULL THEN 1 ELSE 0 END) sin_imagen
        FROM comics
        """,
    ).iloc[0]

    tabla_faltantes = tabla_html(
        ["Campo", "Comics sin ese dato", "% del total"],
        [
            ("escritor", con_comas(faltantes["sin_escritor"]), f"{faltantes['sin_escritor'] / n_comics * 100:.1f}%"),
            ("dibujante", con_comas(faltantes["sin_dibujante"]), f"{faltantes['sin_dibujante'] / n_comics * 100:.1f}%"),
            ("anio_edicion", con_comas(faltantes["sin_anio"]), f"{faltantes['sin_anio'] / n_comics * 100:.1f}%"),
            ("img_src", con_comas(faltantes["sin_imagen"]), f"{faltantes['sin_imagen'] / n_comics * 100:.1f}%"),
        ],
        titulo="Metadata faltante en comics (NULL en la tabla)",
    )

    kpis_correlacion = "".join(
        [
            stat_tile(
                "Correlación crítica vs. usuarios",
                f"r = {correlacion:.2f}" if correlacion == correlacion else "sin datos suficientes",
                f"sobre {len(comparacion)} comics con ≥3 reviews de cada tipo",
            ),
            stat_tile(
                "Usuarios con ≥2 series distintas",
                f"{pct_multi_serie:.1f}%",
                "condición necesaria para que el filtrado colaborativo tenga señal cruzada",
            ),
        ]
    )

    html_final = _armar_html(
        kpis=kpis,
        kpis_correlacion=kpis_correlacion,
        chart_editoriales=chart_editoriales,
        chart_actividad=chart_actividad,
        chart_long_tail=chart_long_tail,
        chart_temporal=chart_temporal,
        chart_ratings=chart_ratings,
        tabla_top_comics=tabla_top_comics,
        tabla_faltantes=tabla_faltantes,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html_final, encoding="utf-8")
    print(f"Informe generado en {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


def _armar_html(**secciones: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>EDA -- comics.db</title>
<style>
  :root {{
    --surface-1: {SURFACE};
    --page-plane: {PAGE_PLANE};
    --text-primary: {INK_PRIMARY};
    --text-secondary: {INK_SECONDARY};
    --text-muted: {INK_MUTED};
    --border: {BORDER};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 24px 80px;
    background: var(--page-plane);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .viz-page {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 28px; margin-bottom: 4px; }}
  .viz-lead {{ color: var(--text-secondary); margin-top: 0; margin-bottom: 32px; }}
  h2 {{ font-size: 18px; margin: 40px 0 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
  .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
  .viz-charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
  .viz-stat-tile {{
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 18px;
  }}
  .viz-stat-label {{ font-size: 13px; color: var(--text-secondary); }}
  .viz-stat-value {{ font-size: 26px; font-weight: 600; margin-top: 4px; }}
  .viz-stat-sub {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}
  .viz-card {{
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 20px 12px;
  }}
  .viz-titulo {{ font-size: 15px; font-weight: 600; margin: 0 0 2px; }}
  .viz-subtitulo {{ font-size: 12px; color: var(--text-muted); margin: 0 0 8px; font-family: ui-monospace, monospace; }}
  .viz-tick {{ font-size: 11px; fill: var(--text-muted); }}
  .viz-tick-strong {{ font-size: 11px; fill: var(--text-secondary); font-weight: 600; }}
  .viz-legend {{ display: flex; gap: 16px; font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; }}
  .viz-legend-item {{ display: inline-flex; align-items: center; gap: 6px; }}
  .viz-swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
  .viz-hit {{ cursor: default; }}
  .viz-tabla {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .viz-tabla th {{ text-align: left; color: var(--text-secondary); font-weight: 600; padding: 6px 8px; border-bottom: 1px solid var(--border); }}
  .viz-tabla td {{ padding: 6px 8px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }}
  .viz-tooltip {{
    position: fixed; display: none; pointer-events: none; z-index: 100;
    background: var(--text-primary); color: #fff; font-size: 12px;
    padding: 6px 10px; border-radius: 6px; max-width: 260px;
  }}
  footer {{ margin-top: 48px; font-size: 12px; color: var(--text-muted); }}
</style>
</head>
<body>
<div class="viz-page">
  <h1>EDA -- comics.db</h1>
  <p class="viz-lead">
    Base de datos de comics scrapeada de comicbookroundup.com para el TP de sistema de
    recomendación. Generado con Python + SQL directo sobre <code>comics/data/raw/comics.db</code>
    (ver <code>comics/scripts/generar_eda.py</code> y <code>comics/CLAUDE.md</code> para el detalle
    del pipeline de scraping).
  </p>

  <h2>Panorama general</h2>
  <div class="viz-grid">{secciones['kpis']}</div>

  <h2>Editoriales</h2>
  <div class="viz-charts-grid">{secciones['chart_editoriales']}</div>

  <h2>Actividad de usuarios y long tail de ítems</h2>
  <div class="viz-charts-grid">{secciones['chart_actividad']}{secciones['chart_long_tail']}</div>

  <h2>Evolución temporal</h2>
  <div class="viz-charts-grid">{secciones['chart_temporal']}</div>

  <h2>Ratings: usuarios vs. crítica especializada</h2>
  <div class="viz-grid">{secciones['kpis_correlacion']}</div>
  <div class="viz-charts-grid">{secciones['chart_ratings']}</div>

  <h2>Tops y calidad de datos</h2>
  <div class="viz-charts-grid">{secciones['tabla_top_comics']}{secciones['tabla_faltantes']}</div>

  <footer>Generado automáticamente -- volver a correr <code>uv run python comics/scripts/generar_eda.py</code> después de cada tanda de scraping para refrescar este informe.</footer>
</div>
<div id="viz-tooltip" class="viz-tooltip" role="tooltip"></div>
<script>
function vizShow(evt) {{
  var tip = document.getElementById('viz-tooltip');
  tip.textContent = evt.currentTarget.getAttribute('data-tip');
  tip.style.display = 'block';
  vizMove(evt);
}}
function vizMove(evt) {{
  var tip = document.getElementById('viz-tooltip');
  var pad = 14;
  tip.style.left = (evt.clientX + pad) + 'px';
  tip.style.top = (evt.clientY + pad) + 'px';
}}
function vizHide() {{
  document.getElementById('viz-tooltip').style.display = 'none';
}}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
