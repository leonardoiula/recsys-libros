"""PLANTILLA del WSGI config de PythonAnywhere -- este archivo NO se ejecuta
desde el repo. Su contenido se copia (pestaña "Web" > link "WSGI configuration
file") en /var/www/leonardoiula_pythonanywhere_com_wsgi.py, reemplazando todo
lo que PythonAnywhere trae por default.

Por qué el secreto va acá y no en el repo: ese archivo de /var/www vive fuera
del clon de git, así que la SECRET_KEY real nunca termina versionada (el repo
es público). Reemplazar CAMBIAR-POR-UN-VALOR-RANDOM por la salida de:
    python3.11 -c "import secrets; print(secrets.token_hex(32))"

COMICS_DB_PATH y COMICS_RECS_CACHE_PATH no hace falta setearlos si la BD y el
.npz se suben a las rutas default de config.py (comics/data/raw/comics.db y
comics/data/cache/item_item_top50.npz, dentro del clon).
"""

import os
import sys

os.environ["COMICS_SECRET_KEY"] = "CAMBIAR-POR-UN-VALOR-RANDOM"

sys.path.insert(0, "/home/leonardoiula/recsys-libros/comics/webapp")

from wsgi import application  # noqa: E402,F401
