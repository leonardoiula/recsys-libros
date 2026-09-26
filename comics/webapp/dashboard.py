"""Blueprint del dashboard: recomendados, pila por leer, y "tu comiteca"
(paginada, ordenable) -- más las acciones que un usuario puede hacer sobre un
comic recomendado (marcarlo leído y puntuarlo, o guardarlo para después). Ver
comics/docs/webapp/guia.md, sección "Arquitectura"."""

from __future__ import annotations

import math

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required

from . import onboarding_datos, repo
from .db import get_db

bp = Blueprint("dashboard", __name__)

POR_PAGINA = 30


@bp.route("/")
@login_required
def index():
    conn = get_db()
    recomendador = current_app.extensions["recomendador"]

    orden = request.args.get("orden", repo.ORDEN_DEFAULT)
    if orden not in repo.ORDENES:
        orden = repo.ORDEN_DEFAULT

    editoriales = repo.editoriales_leidas(conn, current_user.id)
    editorial = request.args.get("editorial") or None
    if editorial not in editoriales:
        editorial = None  # ignora un valor que no es una editorial real de este usuario

    total_leidos = repo.contar_comics_leidos(conn, current_user.id, editorial=editorial)
    total_paginas = max(math.ceil(total_leidos / POR_PAGINA), 1)
    pagina = request.args.get("pagina", 1, type=int) or 1
    pagina = min(max(pagina, 1), total_paginas)

    leidos = repo.comics_leidos(
        conn, current_user.id, orden=orden, editorial=editorial, pagina=pagina, por_pagina=POR_PAGINA
    )

    pila = repo.pila_por_leer(conn, current_user.id)
    ids_pila = {fila["id_comic"] for fila in pila}

    # se piden más de los que hacen falta (k=20) porque algunos candidatos
    # pueden quedar filtrados por ya estar en la pila -- así casi siempre
    # llegan los 10 igual, sin tener que recalcular nada.
    # Si pasó por el onboarding, el perfil suma sus curiosidades y el orden
    # por editorial; si no (ej. identidad reclamada), None = camino de siempre.
    extra = onboarding_datos.perfil_para_recomendar(conn, current_user.id) or {}
    candidatos = recomendador.recomendar(current_user.id, k=20, **extra)
    ids_recomendados = [c for c in candidatos if c not in ids_pila][:10]
    recomendados = repo.comics_por_id(conn, ids_recomendados)

    estado_onboarding = onboarding_datos.progreso(conn, current_user.id)
    ofrecer_onboarding = total_leidos == 0 and editorial is None and not (
        estado_onboarding and estado_onboarding["completado_en"]
    )

    return render_template(
        "dashboard.html",
        ofrecer_onboarding=ofrecer_onboarding,
        leidos=leidos,
        recomendados=recomendados,
        pila=pila,
        orden=orden,
        ordenes=repo.ORDENES,
        editorial=editorial,
        editoriales=editoriales,
        pagina=pagina,
        total_paginas=total_paginas,
    )


@bp.route("/comics/<path:id_comic>/leido", methods=["POST"])
@login_required
def marcar_leido(id_comic):
    rating = request.form.get("rating", type=float)
    if rating is None or not (0 <= rating <= 10):
        flash("Elegí cuántas estrellas le das para marcarlo como leído.")
        return redirect(request.referrer or url_for("dashboard.index"))

    conn = get_db()
    repo.marcar_como_leido(conn, current_user.id, id_comic, rating)
    current_app.extensions["recomendador"].registrar_interaccion(current_user.id, id_comic, rating)

    flash("¡Agregado a tu comiteca!")
    return redirect(request.referrer or url_for("dashboard.index"))


@bp.route("/comics/<path:id_comic>/pila", methods=["POST"])
@login_required
def agregar_pila(id_comic):
    repo.agregar_a_pila(get_db(), current_user.id, id_comic)
    flash("Agregado a tu pila por leer.")
    return redirect(request.referrer or url_for("dashboard.index"))


@bp.route("/comics/<path:id_comic>/pila/quitar", methods=["POST"])
@login_required
def quitar_pila(id_comic):
    repo.quitar_de_pila(get_db(), current_user.id, id_comic)
    return redirect(request.referrer or url_for("dashboard.index"))
