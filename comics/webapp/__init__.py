"""App factory: `create_app()` construye y devuelve la app Flask ya
configurada, en vez de un `app = Flask(...)` a nivel módulo. Esto permite
testear con una BD y un cache de recomendaciones temporales
(`create_app(DB_PATH=..., RECS_CACHE_PATH=...)`) sin depender de variables de
entorno, y es la forma que espera un WSGI server como el de PythonAnywhere.
Los blueprints se importan DENTRO de la función (no al tope del módulo) para
evitar el import circular típico de Flask: `auth.py`/`dashboard.py` hacen
`from . import repo`, que necesita que este paquete ya exista en
`sys.modules` -- si se importara al tope de este archivo, ese import pasaría
a mitad de la inicialización del propio paquete. Ver
comics/docs/webapp/guia.md, sección "Arquitectura"."""

from __future__ import annotations

from comics_recsys import db as core_db
from flask import Flask

from .config import COMICS_DIR, Config
from .db import init_app as init_db
from .extensions import login_manager
from .recsys_runtime import Recomendador


def create_app(**config_overrides) -> Flask:
    app = Flask(__name__, static_folder=str(COMICS_DIR / "static"), static_url_path="/static")
    app.config.from_object(Config)
    app.config.update(config_overrides)

    login_manager.init_app(app)
    init_db(app)

    from . import models_user  # noqa: F401 -- registra el user_loader al importarse
    from . import onboarding_datos
    from .auth import bp as auth_bp
    from .dashboard import bp as dashboard_bp
    from .onboarding import bp as onboarding_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(onboarding_bp)

    # Tablas propias del onboarding: se crean acá (idempotente) para que un
    # `git pull` + Reload alcance en PythonAnywhere, sin migración a mano.
    conn = core_db.conectar(app.config["DB_PATH"])
    onboarding_datos.crear_tablas(conn)
    conn.close()

    app.extensions["recomendador"] = Recomendador(
        app.config["RECS_CACHE_PATH"], app.config["DB_PATH"]
    )
    app.extensions["catalogo_onboarding"] = onboarding_datos.CatalogoOnboarding(app.config["DB_PATH"])

    return app
