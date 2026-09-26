"""Recomendador en runtime: carga UNA vez (al boot del proceso Flask) la
similitud item-item precomputada por comics/scripts/precomputar_item_item.py
y el ranking de popularidad, y combina ambos por request. Nada se recalcula
por request -- importante por la cuota de CPU-segundos de PythonAnywhere
free. Ver comics/docs/webapp/guia.md, sección "Motor de recomendación"."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from comics_recsys.data import comics_calificados_por_usuario, load_interacciones
from comics_recsys.models.popularity import fit_popularity


# Tope de issues de una misma serie entre los recomendados. Sin tope, la
# similitud item-item llena la lista con la serie del comic que el usuario
# puntuó (puntuar Ultimate Spider-Man #21 daba 10 de 10 recomendaciones de
# Ultimate Spider-Man): los issues de una serie son casi siempre los vecinos
# más parecidos entre sí, porque los leen las mismas personas. 2 y no 1: el
# número siguiente de algo que te gustó sigue siendo una buena recomendación.
MAX_POR_SERIE = 2


def _cargar_similitud(cache_path: Path) -> tuple[sp.csr_matrix, np.ndarray]:
    datos = np.load(cache_path)
    similitud = sp.csr_matrix(
        (datos["data"], datos["indices"], datos["indptr"]), shape=tuple(datos["shape"])
    )
    return similitud, datos["comic_ids"]


class Recomendador:
    def __init__(self, cache_path: Path, comics_db_path: Path):
        self._similitud, self._comic_ids = _cargar_similitud(cache_path)
        self._idx_de_comic = {comic_id: i for i, comic_id in enumerate(self._comic_ids)}

        interacciones = load_interacciones(comics_db_path)
        self._perfiles = comics_calificados_por_usuario(interacciones)
        # dict, no pandas Series: `registrar_interaccion` necesita poder
        # sumarle entradas nuevas en memoria (un rating puesto desde el
        # sitio) en O(1), sin reabrir la BD ni tocar una estructura inmutable.
        self._ratings: dict[tuple[str, str], float] = dict(
            zip(zip(interacciones["id_usuario"], interacciones["id_comic"]), interacciones["rating"])
        )
        self._ranking_popularidad = fit_popularity(interacciones)["id_comic"].tolist()
        # Solo para ordenar el relleno por popularidad según las editoriales
        # que el usuario eligió en el onboarding (ver `recomendar`).
        with sqlite3.connect(comics_db_path) as conn:
            self._editorial_de: dict[str, str | None] = dict(
                conn.execute("SELECT id_comic, editorial FROM comics")
            )

    def registrar_interaccion(self, id_usuario: str, id_comic: str, rating: float) -> None:
        """Refleja en memoria un rating puesto desde el sitio (ver
        `webapp.repo.marcar_como_leido`) para que la próxima llamada a
        `recomendar()` de este usuario, en el mismo proceso, ya lo tenga en
        cuenta -- sin esperar a un restart. Deliberadamente NO recalcula la
        similitud item-item (sigue siendo el precómputo offline, ver
        comics/docs/webapp/guia.md) ni el ranking de popularidad completo: una
        sola fila nueva no justifica ese costo de CPU."""
        self._perfiles.setdefault(id_usuario, set()).add(id_comic)
        self._ratings[(id_usuario, id_comic)] = rating

    def recomendar(
        self,
        id_usuario: str,
        k: int = 10,
        perfil_coldstart: dict[str, float] | None = None,
        editoriales_preferidas: set[str] | None = None,
        editoriales_evitadas: set[str] | None = None,
        max_por_serie: int | None = MAX_POR_SERIE,
    ) -> list[str]:
        """`perfil_coldstart` ({id_comic: peso}) permite alimentar la
        recomendación sin leer el historial real de `interacciones` del
        usuario -- es la interfaz que usa el onboarding (webapp/onboarding.py).

        `editoriales_preferidas`/`editoriales_evitadas` (también del
        onboarding) solo reordenan el RELLENO por popularidad: primero lo
        popular de las preferidas, después el resto, al final las evitadas.
        No tocan el score item-item -- una editorial es una señal mucho más
        gruesa que un comic concreto, y no debería pisar a los vecinos.

        `max_por_serie`: tope de issues de una misma serie en la lista final
        (None = sin tope). Ver `MAX_POR_SERIE`."""
        if perfil_coldstart is not None:
            perfil = perfil_coldstart
        else:
            leidos = self._perfiles.get(id_usuario, set())
            perfil = {
                id_comic: float(self._ratings.get((id_usuario, id_comic), 1.0)) for id_comic in leidos
            }

        ya_visto = set(perfil)
        candidatos = self._agregar_vecinos(perfil) if perfil else {}
        por_score = [
            id_comic
            for id_comic, _score in sorted(candidatos.items(), key=lambda par: par[1], reverse=True)
            if id_comic not in ya_visto
        ]
        relleno = self._relleno_por_popularidad(
            ya_visto | set(por_score), editoriales_preferidas or set(), editoriales_evitadas or set()
        )
        return _limitar_por_serie(por_score + relleno, k, max_por_serie)

    def _agregar_vecinos(self, perfil: dict[str, float]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for id_comic, peso in perfil.items():
            idx = self._idx_de_comic.get(id_comic)
            if idx is None:
                continue
            inicio, fin = self._similitud.indptr[idx], self._similitud.indptr[idx + 1]
            for vecino_idx, sim in zip(
                self._similitud.indices[inicio:fin], self._similitud.data[inicio:fin]
            ):
                vecino_id = self._comic_ids[vecino_idx]
                scores[vecino_id] = scores.get(vecino_id, 0.0) + peso * sim
        return scores

    def _relleno_por_popularidad(self, excluidos: set[str], preferidas: set[str], evitadas: set[str]) -> list[str]:
        relleno = [c for c in self._ranking_popularidad if c not in excluidos]
        if preferidas or evitadas:
            # sort estable: dentro de cada grupo se conserva el orden de popularidad
            def grupo(id_comic: str) -> int:
                editorial = self._editorial_de.get(id_comic)
                return 0 if editorial in preferidas else 2 if editorial in evitadas else 1

            relleno.sort(key=grupo)
        return relleno


def _serie(id_comic: str) -> str:
    # "marvel-comics/house-of-x/2" -> "marvel-comics/house-of-x"
    return id_comic.rsplit("/", 1)[0]


def _limitar_por_serie(ordenados: list[str], k: int, max_por_serie: int | None) -> list[str]:
    """Primeros `k` de `ordenados` salteando los que excedan el tope por
    serie. Recorre en orden y corta apenas junta `k`: no reordena nada, solo
    saltea."""
    if max_por_serie is None:
        return ordenados[:k]
    vistos: dict[str, int] = {}
    resultado: list[str] = []
    for id_comic in ordenados:
        serie = _serie(id_comic)
        if vistos.get(serie, 0) >= max_por_serie:
            continue
        vistos[serie] = vistos.get(serie, 0) + 1
        resultado.append(id_comic)
        if len(resultado) == k:
            break
    return resultado
