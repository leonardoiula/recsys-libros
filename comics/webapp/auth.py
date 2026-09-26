"""Blueprint de autenticación: login/logout y las dos vías de registro
(cuenta nueva / reclamo de identidad scrapeada). Ver comics/docs/webapp/guia.md,
sección "Autenticación"."""

from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required, login_user, logout_user

from . import repo
from .db import get_db
from .models_user import User

bp = Blueprint("auth", __name__)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identificador = request.form["identificador"].strip()
        password = request.form["password"]
        conn = get_db()

        fila = repo.obtener_usuario(conn, identificador) or repo.obtener_usuario_por_nombre(
            conn, identificador
        )
        if fila is not None and repo.verificar_password(conn, fila["id_usuario"], password):
            login_user(User(fila["id_usuario"], fila["nombre"]))
            return redirect(url_for("dashboard.index"))

        flash("Usuario o contraseña incorrectos.")
    return render_template("login.html")


@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))


@bp.route("/registro", methods=["GET", "POST"])
def registro():
    if request.method == "POST":
        nombre = request.form["nombre"].strip()
        password = request.form["password"]
        if not nombre or not password:
            flash("Nombre y contraseña son obligatorios.")
            return render_template("registro.html")

        id_usuario = repo.crear_cuenta_nueva(get_db(), nombre, password)
        login_user(User(id_usuario, nombre))
        # Cuenta nueva = sin historial: se le ofrece el onboarding (opcional,
        # ver onboarding.bienvenida). Una identidad reclamada
        # (registro_reclamar) ya trae reviews y va directo al dashboard.
        return redirect(url_for("onboarding.bienvenida"))

    return render_template("registro.html")


@bp.route("/registro/buscar")
def registro_buscar():
    query = request.args.get("q", "").strip()
    resultados = repo.buscar_usuarios_sin_reclamar(get_db(), query) if query else []
    return render_template("registro_reclamar.html", query=query, resultados=resultados)


@bp.route("/registro/reclamar", methods=["POST"])
def registro_reclamar():
    id_usuario = request.form["id_usuario"]
    password = request.form["password"]
    conn = get_db()

    try:
        repo.reclamar_identidad(conn, id_usuario, password)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("auth.registro"))

    fila = repo.obtener_usuario(conn, id_usuario)
    login_user(User(fila["id_usuario"], fila["nombre"]))
    return redirect(url_for("dashboard.index"))
