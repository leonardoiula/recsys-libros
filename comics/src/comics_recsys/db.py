"""
Acceso a comics/data/raw/comics.db.

Mismo enfoque que src/recsys/data.py del proyecto de libros (sqlite3 puro, sin ORM),
pero acá además necesitamos ESCRIBIR de a un issue por vez -- nunca acumular todas
las reviews scrapeadas en memoria antes de guardarlas. Esa fue justo la lección que
ya pagó caro el proyecto de libros (ArrayMemoryError armando todos los candidatos
de una), así que acá la aplicamos desde el vamos: cada llamada a
`guardar_pagina_issue` abre su propia transacción chica y hace commit.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from comics_recsys.scraping.parse import ComicMetadata, ReviewCritico, ReviewUsuario

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "comics.db"

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS comics (
    id_comic TEXT PRIMARY KEY,
    titulo TEXT,
    serie TEXT,
    numero TEXT,
    editorial TEXT,
    anio_edicion INTEGER,
    escritor TEXT,
    dibujante TEXT,
    precio_tapa TEXT,
    img_src TEXT,
    url TEXT
);

CREATE TABLE IF NOT EXISTS usuarios (
    id_usuario TEXT PRIMARY KEY,
    nombre TEXT
);

CREATE TABLE IF NOT EXISTS interacciones (
    id_usuario TEXT NOT NULL,
    id_comic TEXT NOT NULL,
    fecha TEXT,
    rating REAL,
    texto TEXT,
    PRIMARY KEY (id_usuario, id_comic),
    FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario),
    FOREIGN KEY (id_comic) REFERENCES comics(id_comic)
);

CREATE TABLE IF NOT EXISTS critic_reviews (
    id_comic TEXT NOT NULL,
    outlet TEXT NOT NULL,
    reviewer TEXT,
    fecha TEXT,
    rating REAL,
    texto TEXT,
    url_externa TEXT,
    PRIMARY KEY (id_comic, outlet, reviewer),
    FOREIGN KEY (id_comic) REFERENCES comics(id_comic)
);
"""


def conectar(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def crear_esquema(conn: sqlite3.Connection) -> None:
    conn.executescript(_ESQUEMA)
    conn.commit()


def existe_comic(conn: sqlite3.Connection, id_comic: str) -> bool:
    cur = conn.execute("SELECT 1 FROM comics WHERE id_comic = ?", (id_comic,))
    return cur.fetchone() is not None


def _normalizar_fecha(fecha_texto: str) -> str | None:
    """'Sep 02, 2025' -> '2025-09-02'. Si no matchea el formato esperado, guarda None
    en vez de reventar todo el scraping por una fecha rara (ej. 'N/A')."""
    try:
        return datetime.strptime(fecha_texto, "%b %d, %Y").date().isoformat()
    except (ValueError, TypeError):
        return None


def guardar_pagina_issue(
    conn: sqlite3.Connection,
    comic: ComicMetadata,
    reviews_usuario: list[ReviewUsuario],
    reviews_critico: list[ReviewCritico],
) -> None:
    """Guarda todo lo scrapeado de UN issue en una sola transacción chica.
    Idempotente: correr el scraper dos veces sobre el mismo issue no duplica filas."""
    with conn:  # `with` sobre la conexión = una transacción, commit/rollback automático
        conn.execute(
            """
            INSERT INTO comics (id_comic, titulo, serie, numero, editorial,
                                 anio_edicion, escritor, dibujante, precio_tapa, img_src, url)
            VALUES (:id_comic, :titulo, :serie, :numero, :editorial,
                    :anio_edicion, :escritor, :dibujante, :precio_tapa, :img_src, :url)
            ON CONFLICT (id_comic) DO UPDATE SET
                titulo=excluded.titulo, serie=excluded.serie, numero=excluded.numero,
                editorial=excluded.editorial, anio_edicion=excluded.anio_edicion,
                escritor=excluded.escritor, dibujante=excluded.dibujante,
                precio_tapa=excluded.precio_tapa, img_src=excluded.img_src, url=excluded.url
            """,
            vars(comic),
        )

        for r in reviews_usuario:
            conn.execute(
                "INSERT OR IGNORE INTO usuarios (id_usuario, nombre) VALUES (?, ?)",
                (r.id_usuario, r.nombre_usuario),
            )
            conn.execute(
                """
                INSERT INTO interacciones (id_usuario, id_comic, fecha, rating, texto)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (id_usuario, id_comic) DO UPDATE SET
                    fecha=excluded.fecha, rating=excluded.rating, texto=excluded.texto
                """,
                (r.id_usuario, comic.id_comic, _normalizar_fecha(r.fecha_texto), r.rating, r.texto),
            )

        for c in reviews_critico:
            conn.execute(
                """
                INSERT INTO critic_reviews (id_comic, outlet, reviewer, fecha, rating, texto, url_externa)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id_comic, outlet, reviewer) DO UPDATE SET
                    fecha=excluded.fecha, rating=excluded.rating, texto=excluded.texto,
                    url_externa=excluded.url_externa
                """,
                (comic.id_comic, c.outlet, c.reviewer, _normalizar_fecha(c.fecha_texto), c.rating, c.texto, c.url_externa),
            )
