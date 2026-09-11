"""Recomendador en runtime: carga UNA vez (al boot del proceso Flask) la
similitud item-item precomputada por comics/scripts/precomputar_item_item.py
y el ranking de popularidad, y combina ambos por request. Nada se recalcula
por request -- importante por la cuota de CPU-segundos de PythonAnywhere
free. Ver comics/docs/webapp/guia.md, sección "Motor de recomendación"."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from comics_recsys.data import comics_calificados_por_usuario, load_interacciones
from comics_recsys.models.popularity import fit_popularity


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
    ) -> list[str]:
        """`perfil_coldstart` ({id_comic: peso}) permite alimentar la
        recomendación sin leer el historial real de `interacciones` del
        usuario -- es la interfaz que la fase 2 (perfil armado por el juego
        de onboarding) va a poder usar sin que este método cambie de firma."""
        if perfil_coldstart is not None:
            perfil = perfil_coldstart
        else:
            leidos = self._perfiles.get(id_usuario, set())
            perfil = {
                id_comic: float(self._ratings.get((id_usuario, id_comic), 1.0)) for id_comic in leidos
            }

        if not perfil:
            return self._completar_con_popularidad([], set(), k)

        candidatos = self._agregar_vecinos(perfil)
        ya_visto = set(perfil)
        recomendados = [
            id_comic
            for id_comic, _score in sorted(candidatos.items(), key=lambda par: par[1], reverse=True)
            if id_comic not in ya_visto
        ][:k]

        return self._completar_con_popularidad(recomendados, ya_visto, k)

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

    def _completar_con_popularidad(
        self, recomendados: list[str], excluidos: set[str], k: int
    ) -> list[str]:
        if len(recomendados) >= k:
            return recomendados[:k]
        ya_elegidos = excluidos | set(recomendados)
        relleno = [c for c in self._ranking_popularidad if c not in ya_elegidos]
        return (recomendados + relleno)[:k]
