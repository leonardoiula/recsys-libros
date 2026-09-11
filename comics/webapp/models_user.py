"""`User` liviano para Flask-Login: solo envuelve el id_usuario, no carga todo
el perfil en la sesión (eso se resuelve por request contra la BD, en
`repo.py`). Ver comics/docs/webapp/guia.md, sección "Autenticación"."""

from __future__ import annotations

from flask_login import UserMixin

from . import repo
from .db import get_db
from .extensions import login_manager


class User(UserMixin):
    def __init__(self, id_usuario: str, nombre: str):
        self.id = id_usuario
        self.nombre = nombre


@login_manager.user_loader
def cargar_usuario(id_usuario: str) -> User | None:
    fila = repo.obtener_usuario(get_db(), id_usuario)
    if fila is None:
        return None
    return User(fila["id_usuario"], fila["nombre"])
