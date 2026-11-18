import shutil
from pathlib import Path

import kagglehub

DESTINO = Path(__file__).resolve().parent.parent / "data"
DATASET = "paultimothymooney/chest-xray-pneumonia"

print("Descargando dataset (más de 1 GB), puede tardar...")
try:
    ruta = kagglehub.dataset_download(DATASET)
except Exception as e:
    print("No se pudo descargar:", e)
    print("Si pide credenciales: kaggle.com > Settings > Create New Token,")
    print("y guarda el kaggle.json en C:\\Users\\User\\.kaggle\\")
    raise SystemExit(1)

print("Copiando a", DESTINO)
shutil.copytree(ruta, DESTINO, dirs_exist_ok=True)
print("Listo.")
