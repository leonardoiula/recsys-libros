"""Datos del onboarding (la ficción interactiva para usuarios nuevos, guion en
comics/1. El Despertar y la Bienvenida.txt): qué comics mostrar en cada fase,
dónde se guardan las respuestas y cómo se traducen en un perfil para el
recomendador. El blueprint (webapp/onboarding.py) solo orquesta pantallas.

Qué va a dónde (decisión explícita del usuario, 2026-09-26):
- "Lo leí" + rating -> fila REAL en `interacciones` (igual que el botón
  "Leído" del dashboard, reusa `repo.marcar_como_leido`). Es un dato
  verdadero de lectura, sirve también para entrenar.
- Preferencias por editorial y "me da curiosidad" -> tablas propias de
  onboarding. No son lecturas, así que NO se escriben en `interacciones`
  (ensuciarían el entrenamiento offline); entran al recomendador solo como
  `perfil_coldstart` / orden del relleno.
Ver comics/docs/webapp/guia.md, sección "Onboarding".
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from datetime import date

import pandas as pd

# (valor en comics.editorial, nombre para mostrar, logo, sitio oficial).
# Las 12 con más reviews de usuarios en el dataset (Marvel+DC+Image concentran
# el 91%). Logos: mismo hotlink provisorio que las tapas (ver guia.md,
# sección 6); None = el sitio no tiene logo para esa editorial, se muestra el
# nombre. Sitios verificados a mano el 2026-09-26.
_LOGOS = "https://comicbookroundup.com/images/publishers/logos/"
EDITORIALES = [
    ("Marvel", "Marvel", _LOGOS + "marvel_logo.jpg", "https://www.marvel.com"),
    ("DC", "DC", _LOGOS + "dc_comics_logo.png", "https://www.dc.com"),
    ("Image", "Image", _LOGOS + "image_comics_logo.jpg", "https://imagecomics.com"),
    ("Boom!", "BOOM! Studios", _LOGOS + "boom_studios_logo.jpg", "https://www.boom-studios.com"),
    ("IDW", "IDW", _LOGOS + "idw_pub_logo.jpg", "https://idwpublishing.com"),
    ("Dark Horse", "Dark Horse", _LOGOS + "dark_horse_logo.jpg", "https://www.darkhorse.com"),
    ("vertigo", "Vertigo", _LOGOS + "vertigo_logo.jpg", "https://www.dc.com"),
    ("Dynamite", "Dynamite", _LOGOS + "dynamite_logo.jpg", "https://dynamite.com"),
    ("aftershock-comics", "AfterShock", None, "https://aftershockcomics.com"),
    ("vault-comics", "Vault", _LOGOS + "vault_logo.png", "https://vaultcomics.com"),
    ("titan-books", "Titan", _LOGOS + "titan_logo.jpg", "https://titan-comics.com"),
    ("mad-cave-studios", "Mad Cave", _LOGOS + "mad_cave_logo.jpg", "https://www.madcavestudios.com"),
]
EDITORIALES_VALIDAS = {e[0] for e in EDITORIALES}

# Preferencia por editorial (Fase 1). Sin elegir nada = "no la conozco".
PREFERENCIAS = {
    "encanta": "Me encanta",
    "gusta": "Me gusta",
    "curiosidad": "No la conozco, pero me da curiosidad",
    "no_atrae": "No me atrae",
}
PREFERIDAS = {"encanta", "gusta", "curiosidad"}
MIN_EDITORIALES_EVALUADAS = 3
# Fases 2-5: respuestas mínimas (nota o curiosidad) por carrusel.
MIN_INTERACCIONES_POR_FASE = 3

# Peso de un "me da curiosidad" en el perfil, en la misma escala 0-10 que los
# ratings (que el recomendador usa directo como peso). Sin datos para
# validarlo -- no hay forma de medir el NDCG del onboarding offline --, así
# que es un valor de criterio: algo menos que un comic leído y bien puntuado.
PESO_CURIOSIDAD = 7.0

N_POR_FASE = 12
FASES_COMICS = {2: "reciente", 3: "cinco_anios", 4: "fundacionales", 5: "anomalias"}

_DDL = """
CREATE TABLE IF NOT EXISTS onboarding_progreso (
    id_usuario TEXT PRIMARY KEY REFERENCES usuarios(id_usuario),
    fase INTEGER NOT NULL DEFAULT 0,
    completado_en TEXT
);
CREATE TABLE IF NOT EXISTS onboarding_editoriales (
    id_usuario TEXT NOT NULL REFERENCES usuarios(id_usuario),
    editorial TEXT NOT NULL,
    preferencia TEXT NOT NULL,
    PRIMARY KEY (id_usuario, editorial)
);
CREATE TABLE IF NOT EXISTS onboarding_curiosidad (
    id_usuario TEXT NOT NULL REFERENCES usuarios(id_usuario),
    id_comic TEXT NOT NULL REFERENCES comics(id_comic),
    fase INTEGER NOT NULL,
    PRIMARY KEY (id_usuario, id_comic)
);
"""


def crear_tablas(conn: sqlite3.Connection) -> None:
    """Idempotente, se corre al arrancar la app: así un deploy nuevo sobre una
    BD ya poblada (la de PythonAnywhere) no necesita un script de migración."""
    conn.executescript(_DDL)
    conn.commit()


# --- progreso -----------------------------------------------------------------


def progreso(conn: sqlite3.Connection, id_usuario: str) -> dict | None:
    fila = conn.execute(
        "SELECT fase, completado_en FROM onboarding_progreso WHERE id_usuario = ?", (id_usuario,)
    ).fetchone()
    return None if fila is None else {"fase": fila[0], "completado_en": fila[1]}


def avanzar(conn: sqlite3.Connection, id_usuario: str, fase: int, completado: bool = False) -> None:
    """Nunca retrocede: volver a una pantalla anterior con el botón "atrás"
    del navegador no pierde el avance ya hecho."""
    conn.execute(
        """
        INSERT INTO onboarding_progreso (id_usuario, fase, completado_en) VALUES (?, ?, ?)
        ON CONFLICT (id_usuario) DO UPDATE SET
            fase = MAX(fase, excluded.fase),
            completado_en = COALESCE(completado_en, excluded.completado_en)
        """,
        (id_usuario, fase, date.today().isoformat() if completado else None),
    )
    conn.commit()


# --- respuestas ---------------------------------------------------------------


def guardar_editoriales(conn: sqlite3.Connection, id_usuario: str, preferencias: dict[str, str]) -> None:
    """Reemplaza las respuestas de la Fase 1 (si la repite, vale la última)."""
    conn.execute("DELETE FROM onboarding_editoriales WHERE id_usuario = ?", (id_usuario,))
    conn.executemany(
        "INSERT INTO onboarding_editoriales (id_usuario, editorial, preferencia) VALUES (?, ?, ?)",
        [(id_usuario, e, p) for e, p in preferencias.items()],
    )
    conn.commit()


def preferencias_editoriales(conn: sqlite3.Connection, id_usuario: str) -> dict[str, str]:
    return dict(
        conn.execute(
            "SELECT editorial, preferencia FROM onboarding_editoriales WHERE id_usuario = ?",
            (id_usuario,),
        )
    )


def guardar_curiosidad(conn: sqlite3.Connection, id_usuario: str, ids_comic: list[str], fase: int) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO onboarding_curiosidad (id_usuario, id_comic, fase) VALUES (?, ?, ?)",
        [(id_usuario, c, fase) for c in ids_comic],
    )
    conn.commit()


def perfil_para_recomendar(conn: sqlite3.Connection, id_usuario: str) -> dict | None:
    """Argumentos extra para `Recomendador.recomendar` si el usuario pasó por
    el onboarding, o None si no (y entonces el recomendador usa su camino de
    siempre, sin cambios para usuarios scrapeados/reclamados).

    El perfil junta los ratings reales (incluye los "lo leí" del onboarding,
    que ya son `interacciones`) + las curiosidades con `PESO_CURIOSIDAD`. Un
    rating real pisa a una curiosidad del mismo comic."""
    if progreso(conn, id_usuario) is None:
        return None
    perfil = {
        id_comic: PESO_CURIOSIDAD
        for (id_comic,) in conn.execute(
            "SELECT id_comic FROM onboarding_curiosidad WHERE id_usuario = ?", (id_usuario,)
        )
    }
    perfil.update(
        conn.execute("SELECT id_comic, rating FROM interacciones WHERE id_usuario = ?", (id_usuario,))
    )
    prefs = preferencias_editoriales(conn, id_usuario)
    return {
        # dict vacío (y no None) => recomendar() rellena con popularidad
        # ordenada por editorial, en vez de caer al historial real (vacío igual)
        "perfil_coldstart": perfil,
        "editoriales_preferidas": {e for e, p in prefs.items() if p in PREFERIDAS},
        "editoriales_evitadas": {e for e, p in prefs.items() if p == "no_atrae"},
    }


# --- qué comics mostrar en cada fase -----------------------------------------


@dataclass(frozen=True)
class Ventanas:
    """Límites temporales de las fases, relativos al "hoy" del dataset (la
    review más reciente), no a la fecha del sistema: si el scraping queda
    viejo, "el último año" sigue significando el último año CON datos."""

    hoy: pd.Timestamp

    @property
    def hace_1_anio(self) -> pd.Timestamp:
        return self.hoy - pd.DateOffset(years=1)

    @property
    def hace_5_anios(self) -> pd.Timestamp:
        return self.hoy - pd.DateOffset(years=5)


class CatalogoOnboarding:
    """Estadísticas por comic, calculadas UNA vez al arrancar la app (mismo
    criterio que el Recomendador: nada pesado por request).

    La fecha de un comic es la de su PRIMERA review de usuario: `anio_edicion`
    falta en ~36% de los comics, y el scraping recorre semanas de lanzamiento,
    así que la primera review cae, en la práctica, en la semana en que salió.
    Solo entran comics con al menos una review: uno sin reviews no tiene
    vecinos en la similitud item-item, marcarlo no mejoraría ninguna
    recomendación."""

    def __init__(self, comics_db_path):
        with sqlite3.connect(comics_db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT c.id_comic, c.titulo, c.serie, c.numero, c.editorial, c.img_src,
                       COUNT(*) AS n, AVG(i.rating) AS media, MIN(i.fecha) AS primera
                FROM interacciones i JOIN comics c ON c.id_comic = i.id_comic
                GROUP BY c.id_comic
                """,
                conn,
            )
        df["primera"] = pd.to_datetime(df["primera"], errors="coerce")
        # Sin tapa, afuera: acá la pregunta es "¿reconocés esta historia?" y
        # lo que se reconoce es la tapa (ej. Deadpool / Batman #1, top del
        # último año, no tiene tapa ni en el sitio original). Es ~0.6% de
        # los comics con reviews.
        df = df.dropna(subset=["primera", "img_src"])
        # Misma fórmula bayesiana que models/popularity.py (con C = n medio):
        # un comic con 3 reviews de 10 no es "más fundacional" que uno con 150 de 9,5.
        m, c = df["media"].mean(), df["n"].mean()
        df["bayes"] = (df["n"] / (df["n"] + c)) * df["media"] + (c / (c + df["n"])) * m
        # serie = id_comic sin el número: "marvel-comics/house-of-x/2" -> "marvel-comics/house-of-x"
        df["clave_serie"] = df["id_comic"].str.rsplit("/", n=1).str[0]
        self._df = df
        self.ventanas = Ventanas(hoy=df["primera"].max()) if len(df) else Ventanas(pd.Timestamp.today())
        self.ids = set(df["id_comic"])

    def comics_de_fase(
        self,
        fase: int,
        id_usuario: str,
        editoriales_elegidas: set[str],
        excluir: set[str] = frozenset(),
        n: int = N_POR_FASE,
    ) -> list[dict]:
        df = self._df[~self._df["id_comic"].isin(excluir)]
        v = self.ventanas
        # Si no eligió ninguna editorial "preferida" (todo "no me atrae"), las
        # fases 2-3 muestran de todas -- mejor eso que un carrusel vacío.
        elegidas = df[df["editorial"].isin(editoriales_elegidas)] if editoriales_elegidas else df

        if fase == 2:  # memoria reciente: último año, los de más impacto
            candidatos = elegidas[elegidas["primera"] > v.hace_1_anio].sort_values("n", ascending=False)
        elif fase == 3:  # ecos de hace 1 a 5 años
            en_ventana = (elegidas["primera"] <= v.hace_1_anio) & (elegidas["primera"] > v.hace_5_anios)
            candidatos = elegidas[en_ventana].sort_values("n", ascending=False)
        elif fase == 4:  # "fundacionales": lo mejor puntuado de lo más viejo del dataset (arranca en 2018)
            viejos = df[(df["primera"] <= v.hace_5_anios) & (df["n"] >= 20)]
            candidatos = viejos.sort_values("bayes", ascending=False)
        elif fase == 5:  # anomalías: al azar, de editoriales NO elegidas
            ajenas = df[~df["editorial"].isin(editoriales_elegidas) & (df["n"] >= 10)]
            ajenas = _uno_por_serie(ajenas.sort_values("n", ascending=False))
            # semilla fija por usuario: recargar la página no cambia la selección
            # (si no, la lista del POST no coincidiría con la que vio)
            rng = random.Random(f"anomalias:{id_usuario}")
            filas = ajenas.to_dict("records")
            return rng.sample(filas, min(n, len(filas)))
        else:
            raise ValueError(f"La fase {fase} no muestra comics")

        return _intercalar_editoriales(_uno_por_serie(candidatos), n)


def _uno_por_serie(df: pd.DataFrame) -> pd.DataFrame:
    """Se queda con el primer issue de cada serie según el orden que ya trae
    `df`. Sin esto el carrusel del último año eran 4 números de Absolute
    Batman seguidos, y fundacionales 3 de House of X: la pregunta es "¿conocés
    esta historia?", repetir la serie no aporta."""
    return df.drop_duplicates(subset="clave_serie", keep="first")


def _intercalar_editoriales(df: pd.DataFrame, n: int) -> list[dict]:
    """Round-robin entre editoriales, conservando el orden dentro de cada una
    (si no, con Marvel+DC elegidas, "el último año" salía 100% DC)."""
    colas = [grupo.to_dict("records") for _, grupo in df.groupby("editorial", sort=False)]
    resultado: list[dict] = []
    while len(resultado) < n and any(colas):
        for cola in colas:
            if cola and len(resultado) < n:
                resultado.append(cola.pop(0))
    return resultado
