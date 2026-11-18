from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
MODELOS = BASE / "models"
CORRIDAS = MODELOS / "corridas"
ACTIVO_PATH = MODELOS / "activo.txt"

# ImageFolder ordena alfabéticamente: NORMAL=0, PNEUMONIA=1
CLASES = ["NORMAL", "PNEUMONIA"]
