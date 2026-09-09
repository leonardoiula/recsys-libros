import sys
from pathlib import Path

# comics/ todavía no es un paquete instalado (no tiene su propio pyproject.toml);
# insertamos comics/src en el path para poder hacer `import comics_recsys...` en los tests,
# igual que si estuviera instalado en modo editable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
