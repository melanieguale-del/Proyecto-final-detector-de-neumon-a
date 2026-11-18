import threading

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from backend import corridas, evaluacion, predict, train
from backend.data import buscar_raiz

corridas.migrar_viejos()
app = FastAPI(title="API detección de neumonía")


class Parametros(BaseModel):
    epocas: int = Field(5, ge=1, le=30)
    lr: float = Field(0.0003, gt=0, le=0.1)
    batch_size: int = Field(32, ge=8, le=128)
    fraccion: float = Field(1.0, ge=0.05, le=1.0)
    congelar: bool = True


@app.get("/estado")
def estado():
    gpu = torch.cuda.is_available()
    return {
        "dispositivo": torch.cuda.get_device_name(0) if gpu else "CPU",
        "gpu": gpu,
        "dataset": buscar_raiz() is not None,
        "modelo": corridas.activa() is not None,
        "entrenamiento": train.estado["estado"],
    }


@app.post("/entrenar")
def entrenar(p: Parametros):
    if train.estado["estado"] == "entrenando":
        raise HTTPException(409, "Ya hay un entrenamiento en curso")
    if buscar_raiz() is None:
        raise HTTPException(400, "Falta el dataset. Ejecuta scripts/descargar_dataset.py")

    hilo = threading.Thread(
        target=train.entrenar,
        args=(p.epocas, p.lr, p.batch_size, p.fraccion, p.congelar),
        daemon=True,
    )
    hilo.start()
    return {"mensaje": "Entrenamiento iniciado"}


@app.get("/progreso")
def progreso():
    return train.estado


@app.get("/corridas")
def lista_corridas():
    return corridas.listar()


@app.post("/corridas/{corrida}/activar")
def activar(corrida: str):
    try:
        corridas.activar(corrida)
    except FileNotFoundError:
        raise HTTPException(404, "Esa corrida no existe")
    return {"activa": corrida}


def _corrida_o_404(corrida):
    corrida = corrida or corridas.activa()
    if corrida is None:
        raise HTTPException(404, "Todavía no hay corridas guardadas")
    return corrida


@app.get("/metricas")
def metricas(corrida: str | None = None):
    r = corridas.leer(_corrida_o_404(corrida))
    r["tiene_roc"] = "prob_test" in r
    r.pop("prob_test", None)
    r.pop("real_test", None)
    return r


@app.get("/evaluar")
def evaluar(umbral: float = Query(0.5, ge=0.01, le=0.99), corrida: str | None = None):
    r = corridas.leer(_corrida_o_404(corrida))
    if "prob_test" not in r:
        raise HTTPException(404, "Esta corrida no guardó probabilidades")
    res = evaluacion.en_umbral(r["real_test"], r["prob_test"], umbral)
    res["roc"] = evaluacion.curva_roc(r["real_test"], r["prob_test"])
    res["auc"] = r["test"]["auc"]
    return res


@app.post("/predecir")
async def predecir(file: UploadFile = File(...),
                   umbral: float = Query(0.5, ge=0.01, le=0.99)):
    if corridas.activa() is None:
        raise HTTPException(400, "Primero hay que entrenar el modelo")
    try:
        return predict.predecir(await file.read(), umbral)
    except Exception:
        raise HTTPException(400, "No se pudo leer la imagen")
