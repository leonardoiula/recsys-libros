"""Instancias de extensiones creadas a nivel módulo (sin `app` todavía) para
evitar imports circulares entre `__init__.py` y los blueprints -- patrón
estándar de Flask ("Application Factories" en la docs oficial): la extensión
se crea una vez acá, y se le hace `.init_app(app)` recién dentro de
`create_app()`."""

from __future__ import annotations

from flask_login import LoginManager

login_manager = LoginManager()
login_manager.login_view = "auth.login"
