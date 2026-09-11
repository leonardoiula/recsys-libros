"""Convención de nombre de archivo para las tapas descargadas en comics/static/covers/.

Se separa de `data.py` (que es sobre leer tablas) porque tanto
`scripts/descargar_tapas.py` (que las escribe) como el webapp (que arma las
URLs `/static/covers/<archivo>` para mostrarlas) necesitan la MISMA función --
si se duplica en los dos lugares, la convención se desincroniza con el tiempo.
"""

from __future__ import annotations


def nombre_archivo(id_comic: str, img_src: str) -> str:
    extension = img_src.rsplit(".", 1)[-1] if "." in img_src.rsplit("/", 1)[-1] else "webp"
    return id_comic.replace("/", "_") + "." + extension
