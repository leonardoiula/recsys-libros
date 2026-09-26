"""Blueprint del onboarding: la ficción interactiva que arma el perfil de un
usuario nuevo antes de su primer dashboard. Guion en
comics/1. El Despertar y la Bienvenida.txt; la lógica de datos vive en
onboarding_datos.py. Ver comics/docs/webapp/guia.md, sección "Onboarding".

Fases: 0 despertar, 1 universos (editoriales), 2 memoria reciente (último
año), 3 ecos (1 a 5 años), 4 fundacionales, 5 anomalías, 6 sincronización.
Server-rendered y sin JavaScript obligatorio, igual que el resto del sitio:
cada fase es un form que al enviarse guarda y redirige a la siguiente.
"""

from __future__ import annotations

from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from . import onboarding_datos as datos
from . import repo
from .db import get_db

bp = Blueprint("onboarding", __name__, url_prefix="/onboarding")

ULTIMA_FASE = 6


@bp.route("/bienvenida")
@login_required
def bienvenida():
    """Primera pantalla después de crear la cuenta: explica que el onboarding
    es un juego OPCIONAL y para qué sirve, antes de meter al usuario en la
    ficción. Misma estética de viñeta que login/registro (continuidad)."""
    return render_template("bienvenida.html")


@bp.route("/")
@login_required
def inicio():
    estado = datos.progreso(get_db(), current_user.id)
    fase = 0 if estado is None else min(estado["fase"], ULTIMA_FASE)
    return redirect(url_for("onboarding.fase", numero=fase))


@bp.route("/<int:numero>", methods=["GET", "POST"])
@login_required
def fase(numero: int):
    if not 0 <= numero <= ULTIMA_FASE:
        abort(404)
    conn = get_db()

    # No se puede saltar hacia adelante escribiendo la URL (/onboarding/6):
    # solo se entra a fases ya alcanzadas. Volver a una anterior sí se puede.
    estado = datos.progreso(conn, current_user.id)
    if estado is None:
        datos.avanzar(conn, current_user.id, 0)  # registra que empezó
        alcanzada = 0
    else:
        alcanzada = estado["fase"]
    if numero > alcanzada:
        return redirect(url_for("onboarding.fase", numero=alcanzada))

    previos: dict = {}
    if request.method == "POST":
        if numero == 1:
            valido = _guardar_editoriales(conn)
        elif numero in datos.FASES_COMICS:
            # "[ NO LOGRO RECORDAR NADA ]": avanza sin el mínimo (ver docstring)
            no_recuerda = request.form.get("accion") == "no_recuerdo"
            valido = _guardar_respuestas_comics(conn, numero, exigir_minimo=not no_recuerda)
        else:
            valido = True
        if valido:
            if numero == ULTIMA_FASE:
                datos.avanzar(conn, current_user.id, ULTIMA_FASE, completado=True)
                return redirect(url_for("dashboard.index"))
            datos.avanzar(conn, current_user.id, numero + 1)
            return redirect(url_for("onboarding.fase", numero=numero + 1))
        # Faltan respuestas: se vuelve a mostrar la MISMA pantalla (no un
        # redirect) con lo que ya había cargado, para no hacerle perder las
        # notas que escribió. No se guardó nada, así que el carrusel sale igual.
        previos = request.form

    contexto = {"numero": numero, "ultima_fase": ULTIMA_FASE, "previos": previos}
    if numero == 1:
        contexto["editoriales"] = datos.EDITORIALES
        contexto["preferencias"] = datos.PREFERENCIAS
        contexto["elegidas"] = (
            {e: previos.get(f"pref-{e}") for e in datos.EDITORIALES_VALIDAS}
            if previos
            else datos.preferencias_editoriales(conn, current_user.id)
        )
        contexto["minimo"] = datos.MIN_EDITORIALES_EVALUADAS
    elif numero in datos.FASES_COMICS:
        contexto["comics"] = _comics_de_fase(conn, numero)
        contexto["minimo"] = datos.MIN_INTERACCIONES_POR_FASE
    return render_template("onboarding.html", **contexto)


def _comics_de_fase(conn, numero: int) -> list[dict]:
    catalogo = current_app.extensions["catalogo_onboarding"]
    prefs = datos.preferencias_editoriales(conn, current_user.id)
    elegidas = {e for e, p in prefs.items() if p in datos.PREFERIDAS}
    # No volver a preguntar por lo que ya respondió en otra fase
    ya_respondidos = {
        fila[0]
        for fila in conn.execute(
            "SELECT id_comic FROM interacciones WHERE id_usuario = ? "
            "UNION SELECT id_comic FROM onboarding_curiosidad WHERE id_usuario = ?",
            (current_user.id, current_user.id),
        )
    }
    return catalogo.comics_de_fase(numero, current_user.id, elegidas, excluir=ya_respondidos)


def _guardar_editoriales(conn) -> bool:
    preferencias = {
        editorial: request.form.get(f"pref-{editorial}")
        for editorial in datos.EDITORIALES_VALIDAS
        if request.form.get(f"pref-{editorial}") in datos.PREFERENCIAS
    }
    if len(preferencias) < datos.MIN_EDITORIALES_EVALUADAS:
        flash(
            f"Señal insuficiente: evaluá al menos {datos.MIN_EDITORIALES_EVALUADAS} universos "
            "para estabilizar la conexión."
        )
        return False
    datos.guardar_editoriales(conn, current_user.id, preferencias)
    return True


def _guardar_respuestas_comics(conn, numero: int, exigir_minimo: bool = True) -> bool:
    """Cada comic del carrusel viaja en el form como hidden `comic` + sus
    campos `rating-<id>` y `curiosidad-<id>`. Se valida contra el catálogo:
    un id inventado no llega nunca a la BD.

    Exige al menos `MIN_INTERACCIONES_POR_FASE` respuestas (nota o
    curiosidad) antes de guardar nada -- si no, se podía atravesar todo el
    onboarding con "SINCRONIZAR" sin dar ninguna señal. Si el carrusel
    mostró menos comics que el mínimo, alcanza con responder todos.

    `exigir_minimo=False` es la salida "no logro recordar nada": alguien que
    de verdad no reconoce ninguno no debería tener que inventar respuestas
    (ensuciaría su perfil). Se guarda lo poco que haya marcado y se avanza."""
    catalogo = current_app.extensions["catalogo_onboarding"]
    ids = [i for i in request.form.getlist("comic") if i in catalogo.ids]
    lecturas: dict[str, float] = {}
    curiosidades: list[str] = []
    for id_comic in ids:
        rating = request.form.get(f"rating-{id_comic}", type=float)
        if rating is not None and 0 <= rating <= 10:
            lecturas[id_comic] = rating
        elif request.form.get(f"curiosidad-{id_comic}"):
            curiosidades.append(id_comic)

    minimo = min(datos.MIN_INTERACCIONES_POR_FASE, len(ids))
    if exigir_minimo and len(lecturas) + len(curiosidades) < minimo:
        flash(
            f"Señal insuficiente: interactuá con al menos {minimo} archivos "
            "(puntuá los que leíste o marcá los que te dan curiosidad)."
        )
        return False

    recomendador = current_app.extensions["recomendador"]
    for id_comic, rating in lecturas.items():
        repo.marcar_como_leido(conn, current_user.id, id_comic, rating)
        recomendador.registrar_interaccion(current_user.id, id_comic, rating)
    datos.guardar_curiosidad(conn, current_user.id, curiosidades, numero)
    return True
