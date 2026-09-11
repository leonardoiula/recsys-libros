"""Entrypoint para correr el server local o para el WSGI config de
PythonAnywhere (que espera encontrar acá un objeto `app`/`application` ya
construido a nivel módulo).

Uso local: uv run python comics/webapp/wsgi.py
PythonAnywhere: apuntar el WSGI config a `application = ...` importado de este
archivo (ver comics/docs/webapp/guia.md, sección "Deployment")."""

from __future__ import annotations

import sys
from pathlib import Path

COMICS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMICS_DIR))
sys.path.insert(0, str(COMICS_DIR / "src"))

from webapp import create_app

app = create_app()
application = app  # nombre que espera el WSGI config de PythonAnywhere

if __name__ == "__main__":
    app.run(debug=True)
