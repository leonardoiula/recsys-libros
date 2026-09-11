import sys
from pathlib import Path

COMICS_DIR = Path(__file__).resolve().parents[1]

# comics/ todavía no es un paquete instalado (no tiene su propio pyproject.toml);
# insertamos comics/src en el path para poder hacer `import comics_recsys...` en los tests,
# igual que si estuviera instalado en modo editable.
sys.path.insert(0, str(COMICS_DIR / "src"))

# mismo criterio para `import webapp...` (comics/webapp/), usado por los tests del sitio Flask.
sys.path.insert(0, str(COMICS_DIR))
