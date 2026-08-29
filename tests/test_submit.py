"""Tests para el nombrado de archivos de submission (nunca pisa nombres)."""

from recsys.submit import _nombre_submission


def test_nombre_incluye_timestamp():
    nombre = _nombre_submission("als", tag=None)
    assert nombre.startswith("als_")
    assert nombre.endswith(".csv")


def test_nombre_con_tag_sanitiza_caracteres_raros():
    nombre = _nombre_submission("als", tag="alpha tuneado! (v2)")
    assert "alpha-tuneado-v2" in nombre or "alpha-tuneado--v2" in nombre
    assert " " not in nombre
    assert "!" not in nombre


def test_nombre_evita_colision_si_ya_existe(tmp_path, monkeypatch):
    monkeypatch.setattr("recsys.submit.SUBMISSIONS_DIR", tmp_path)
    nombre1 = _nombre_submission("als", tag="prueba")
    (tmp_path / nombre1).write_text("x")

    nombre2 = _nombre_submission("als", tag="prueba")

    assert nombre1 != nombre2
    assert not (tmp_path / nombre2).exists()
