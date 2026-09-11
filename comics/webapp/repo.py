"""Capa de datos propia del webapp: auth (passwords, reclamo de identidad) y
las consultas para el dashboard. SQL directo con `sqlite3` (mismo estilo que
`comics_recsys/db.py`), sin ORM. Vive acá y no en `comics_recsys` para no
meter Flask/werkzeug como dependencia del paquete core -- ese paquete lo usan
también los scripts de scraping/evaluación offline, que no necesitan saber
nada de auth. Ver comics/docs/webapp/guia.md, sección "Capa de datos"."""

from __future__ import annotations

import secrets
import sqlite3
from datetime import date

from werkzeug.security import check_password_hash, generate_password_hash


def buscar_usuarios_sin_reclamar(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict]:
    filas = conn.execute(
        """
        SELECT u.id_usuario, u.nombre, COUNT(i.id_comic) AS n_reviews
        FROM usuarios u
        LEFT JOIN interacciones i ON i.id_usuario = u.id_usuario
        WHERE u.password_hash IS NULL AND u.nombre LIKE ?
        GROUP BY u.id_usuario
        ORDER BY n_reviews DESC
        LIMIT ?
        """,
        (f"%{query}%", limit),
    ).fetchall()
    return [dict(fila) for fila in filas]


def reclamar_identidad(conn: sqlite3.Connection, id_usuario: str, password: str) -> None:
    """Setea el password de un usuario scrapeado todavía sin reclamar.

    El WHERE password_hash IS NULL (en vez de un UPDATE sin condición) evita
    una carrera de doble-reclamo: si dos personas intentan reclamar la misma
    identidad casi al mismo tiempo, sólo la primera consigue el UPDATE (rowcount=1);
    la segunda ve rowcount=0 y se entera de que ya no está disponible."""
    cur = conn.execute(
        "UPDATE usuarios SET password_hash = ? WHERE id_usuario = ? AND password_hash IS NULL",
        (generate_password_hash(password), id_usuario),
    )
    if cur.rowcount == 0:
        raise ValueError(f"'{id_usuario}' no existe o ya fue reclamado por otra persona.")
    conn.commit()


def crear_cuenta_nueva(conn: sqlite3.Connection, nombre: str, password: str) -> str:
    """Los id_usuario scrapeados son numéricos (ej. '8994', extraídos del sitio);
    el prefijo 'local-' garantiza que un id sintético nunca choca con uno real."""
    id_usuario = "local-" + secrets.token_hex(8)
    conn.execute(
        "INSERT INTO usuarios (id_usuario, nombre, password_hash) VALUES (?, ?, ?)",
        (id_usuario, nombre, generate_password_hash(password)),
    )
    conn.commit()
    return id_usuario


def verificar_password(conn: sqlite3.Connection, id_usuario: str, password: str) -> bool:
    fila = conn.execute(
        "SELECT password_hash FROM usuarios WHERE id_usuario = ?", (id_usuario,)
    ).fetchone()
    if fila is None or fila["password_hash"] is None:
        return False
    return check_password_hash(fila["password_hash"], password)


def obtener_usuario(conn: sqlite3.Connection, id_usuario: str) -> dict | None:
    fila = conn.execute(
        "SELECT id_usuario, nombre FROM usuarios WHERE id_usuario = ?", (id_usuario,)
    ).fetchone()
    return dict(fila) if fila is not None else None


def obtener_usuario_por_nombre(conn: sqlite3.Connection, nombre: str) -> dict | None:
    """Login por nombre (además de por id_usuario) -- útil porque el id numérico
    scrapeado no es memorizable para el usuario final."""
    fila = conn.execute(
        "SELECT id_usuario, nombre FROM usuarios WHERE nombre = ? AND password_hash IS NOT NULL",
        (nombre,),
    ).fetchone()
    return dict(fila) if fila is not None else None


# Criterios de orden para "Tu comiteca": clave usada en la URL (?orden=...) ->
# (cláusula ORDER BY, etiqueta para mostrar en el selector). La cláusula NUNCA
# viene del usuario directo a la consulta -- se valida contra este diccionario
# antes de interpolarla, así no hay riesgo de inyección SQL por más que
# `orden` llegue crudo desde query params.
# Editorial NO es un criterio de orden -- es un FILTRO (selector con cada
# editorial que el usuario tiene en su comiteca, + "Todas"). Orden y filtro
# son ejes independientes: se puede filtrar por editorial y a la vez elegir
# si esas filas se ven por fecha o por rating.
ORDENES: dict[str, tuple[str, str]] = {
    "fecha": ("i.fecha IS NULL, i.fecha DESC", "Fecha"),
    "rating": ("i.rating IS NULL, i.rating DESC", "Rating"),
}
ORDEN_DEFAULT = "fecha"


def editoriales_leidas(conn: sqlite3.Connection, id_usuario: str) -> list[str]:
    """Editoriales distintas presentes en la comiteca del usuario, para
    poblar el selector de filtro -- no tiene sentido ofrecer editoriales que
    esa persona ni siquiera tiene leídas."""
    filas = conn.execute(
        """
        SELECT DISTINCT c.editorial
        FROM interacciones i
        JOIN comics c ON c.id_comic = i.id_comic
        WHERE i.id_usuario = ? AND c.editorial IS NOT NULL AND c.editorial != ''
        ORDER BY c.editorial ASC
        """,
        (id_usuario,),
    ).fetchall()
    return [fila["editorial"] for fila in filas]


def contar_comics_leidos(conn: sqlite3.Connection, id_usuario: str, editorial: str | None = None) -> int:
    if editorial:
        fila = conn.execute(
            """
            SELECT COUNT(*) AS n FROM interacciones i
            JOIN comics c ON c.id_comic = i.id_comic
            WHERE i.id_usuario = ? AND c.editorial = ?
            """,
            (id_usuario, editorial),
        ).fetchone()
    else:
        fila = conn.execute(
            "SELECT COUNT(*) AS n FROM interacciones WHERE id_usuario = ?", (id_usuario,)
        ).fetchone()
    return fila["n"]


def comics_leidos(
    conn: sqlite3.Connection,
    id_usuario: str,
    orden: str = ORDEN_DEFAULT,
    editorial: str | None = None,
    pagina: int = 1,
    por_pagina: int = 30,
) -> list[dict]:
    clausula, _etiqueta = ORDENES.get(orden, ORDENES[ORDEN_DEFAULT])
    offset = (max(pagina, 1) - 1) * por_pagina

    filtro_editorial = "AND c.editorial = ?" if editorial else ""
    params: list = [id_usuario]
    if editorial:
        params.append(editorial)
    params.extend([por_pagina, offset])

    filas = conn.execute(
        f"""
        SELECT c.id_comic, c.titulo, c.serie, c.numero, c.editorial, c.img_src, i.rating, i.fecha
        FROM interacciones i
        JOIN comics c ON c.id_comic = i.id_comic
        WHERE i.id_usuario = ? {filtro_editorial}
        ORDER BY {clausula}
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()
    return [dict(fila) for fila in filas]


def marcar_como_leido(conn: sqlite3.Connection, id_usuario: str, id_comic: str, rating: float) -> None:
    """Agrega (o actualiza, si ya existía) una interacción real generada desde
    el sitio -- un comic recomendado que el usuario ya leyó y puntúa ahí mismo.
    `texto` queda NULL: es un rating rápido, no una review escrita. Si el
    comic estaba en la pila por leer, se saca de ahí -- ya no está "por leer"."""
    conn.execute(
        """
        INSERT INTO interacciones (id_usuario, id_comic, fecha, rating, texto)
        VALUES (?, ?, ?, ?, NULL)
        ON CONFLICT (id_usuario, id_comic) DO UPDATE SET
            fecha = excluded.fecha, rating = excluded.rating
        """,
        (id_usuario, id_comic, date.today().isoformat(), rating),
    )
    conn.execute(
        "DELETE FROM pila_por_leer WHERE id_usuario = ? AND id_comic = ?", (id_usuario, id_comic)
    )
    conn.commit()


def agregar_a_pila(conn: sqlite3.Connection, id_usuario: str, id_comic: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO pila_por_leer (id_usuario, id_comic, agregado_en) VALUES (?, ?, ?)",
        (id_usuario, id_comic, date.today().isoformat()),
    )
    conn.commit()


def quitar_de_pila(conn: sqlite3.Connection, id_usuario: str, id_comic: str) -> None:
    conn.execute(
        "DELETE FROM pila_por_leer WHERE id_usuario = ? AND id_comic = ?", (id_usuario, id_comic)
    )
    conn.commit()


def pila_por_leer(conn: sqlite3.Connection, id_usuario: str) -> list[dict]:
    filas = conn.execute(
        """
        SELECT c.id_comic, c.titulo, c.serie, c.numero, c.editorial, c.img_src, p.agregado_en
        FROM pila_por_leer p
        JOIN comics c ON c.id_comic = p.id_comic
        WHERE p.id_usuario = ?
        ORDER BY p.agregado_en DESC
        """,
        (id_usuario,),
    ).fetchall()
    return [dict(fila) for fila in filas]


def comics_por_id(conn: sqlite3.Connection, ids_comic: list[str]) -> list[dict]:
    """Preserva el orden de `ids_comic` (no el de SQL) -- es el orden de score
    del recomendador, perderlo acá arruinaría el ranking mostrado."""
    if not ids_comic:
        return []
    placeholders = ",".join("?" * len(ids_comic))
    filas = conn.execute(
        f"SELECT id_comic, titulo, serie, numero, img_src FROM comics WHERE id_comic IN ({placeholders})",
        ids_comic,
    ).fetchall()
    por_id = {fila["id_comic"]: dict(fila) for fila in filas}
    return [por_id[id_comic] for id_comic in ids_comic if id_comic in por_id]
