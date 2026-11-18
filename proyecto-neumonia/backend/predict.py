import io
import torch
from PIL import Image

from backend import corridas
from backend.config import CLASES
from backend.data import tf_eval
from backend.model import crear_modelo, dispositivo

_cache = {"clave": None, "modelo": None}


def cargar():
    # recarga solo si cambió la corrida activa o su archivo
    id = corridas.activa()
    if id is None:
        raise FileNotFoundError("Todavía no hay un modelo entrenado")
    ruta = corridas.carpeta(id) / "modelo.pt"
    clave = (id, ruta.stat().st_mtime)
    if _cache["clave"] != clave:
        modelo = crear_modelo(congelar=False, preentrenado=False)
        modelo.load_state_dict(torch.load(ruta, map_location=dispositivo()))
        _cache["modelo"] = modelo.to(dispositivo()).eval()
        _cache["clave"] = clave
    return _cache["modelo"], id


def predecir(contenido: bytes, umbral=0.5):
    modelo, id = cargar()
    img = Image.open(io.BytesIO(contenido)).convert("RGB")
    x = tf_eval(img).unsqueeze(0).to(dispositivo())

    with torch.no_grad():
        p = torch.softmax(modelo(x), dim=1)[0].cpu().tolist()

    i = int(p[1] >= umbral)
    return {"diagnostico": CLASES[i], "probabilidad": p[1],
            "umbral": umbral, "corrida": id}
