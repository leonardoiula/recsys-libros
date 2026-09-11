import io
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from descargar_tapas import redimensionar_y_guardar


def _png_de_prueba(ancho=800, alto=1200, modo="RGBA") -> bytes:
    img = Image.new(modo, (ancho, alto), color=(10, 20, 30, 255) if modo == "RGBA" else (10, 20, 30))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def test_redimensiona_manteniendo_proporcion(tmp_path):
    contenido = _png_de_prueba(ancho=800, alto=1200)
    ruta = tmp_path / "tapa.png"

    redimensionar_y_guardar(contenido, ruta, ancho=200)

    with Image.open(ruta) as resultado:
        assert resultado.width == 200
        assert resultado.height == 300  # 1200 * 200/800, misma proporción


def test_no_agranda_imagenes_mas_chicas_que_el_ancho_pedido(tmp_path):
    contenido = _png_de_prueba(ancho=100, alto=150)
    ruta = tmp_path / "tapa.png"

    redimensionar_y_guardar(contenido, ruta, ancho=200)

    with Image.open(ruta) as resultado:
        assert resultado.width == 100
        assert resultado.height == 150


def test_convierte_rgba_a_rgb_para_guardar_como_jpg(tmp_path):
    contenido = _png_de_prueba(ancho=400, alto=400, modo="RGBA")
    ruta = tmp_path / "tapa.jpg"

    redimensionar_y_guardar(contenido, ruta, ancho=200)

    with Image.open(ruta) as resultado:
        assert resultado.mode == "RGB"  # JPEG no soporta canal alfa
