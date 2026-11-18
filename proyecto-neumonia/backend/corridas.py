import json
import shutil
import time

from backend.config import CORRIDAS, ACTIVO_PATH, MODELOS


def nueva_id():
    return time.strftime("%Y%m%d-%H%M%S")


def carpeta(id):
    return CORRIDAS / id


def ids():
    if not CORRIDAS.exists():
        return []
    validas = [d.name for d in CORRIDAS.iterdir()
               if (d / "modelo.pt").exists() and (d / "metricas.json").exists()]
    return sorted(validas, reverse=True)


def activa():
    if ACTIVO_PATH.exists():
        id = ACTIVO_PATH.read_text().strip()
        if id in ids():
            return id
    todas = ids()
    return todas[0] if todas else None


def activar(id):
    if id not in ids():
        raise FileNotFoundError(id)
    ACTIVO_PATH.write_text(id)


def leer(id):
    r = json.loads((carpeta(id) / "metricas.json").read_text())
    r["id"] = id
    r.setdefault("validacion", "aleatoria")
    return r


def listar():
    filas = []
    actual = activa()
    for id in ids():
        r = leer(id)
        filas.append({
            "id": id,
            "parametros": r["parametros"],
            "test": r["test"],
            "duracion_seg": r["duracion_seg"],
            "validacion": r["validacion"],
            "activa": id == actual,
        })
    return filas


def migrar_viejos():
    # rescata modelos de la versión anterior (models/modelo*.pt + metricas*.json)
    CORRIDAS.mkdir(parents=True, exist_ok=True)
    principal = None
    for pt in sorted(MODELOS.glob("modelo*.pt")):
        sufijo = pt.stem[len("modelo"):]
        js = MODELOS / f"metricas{sufijo}.json"
        if not js.exists():
            continue
        id = time.strftime("%Y%m%d-%H%M%S", time.localtime(pt.stat().st_mtime))
        while carpeta(id).exists():
            id += "b"
        carpeta(id).mkdir()
        shutil.move(str(pt), carpeta(id) / "modelo.pt")
        shutil.move(str(js), carpeta(id) / "metricas.json")
        if sufijo == "":
            principal = id
    if principal and not ACTIVO_PATH.exists():
        activar(principal)
