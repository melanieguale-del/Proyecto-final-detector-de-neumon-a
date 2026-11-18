import copy
import json
import time

import torch
from torch import nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)

from backend import corridas
from backend.data import cargar_datos
from backend.model import crear_modelo, dispositivo

# el frontend consulta este diccionario mientras se entrena
estado = {"estado": "inactivo", "epoca": 0, "epocas": 0,
          "avance": 0.0, "historial": [], "mensaje": ""}


def correr(modelo, loader, dev, criterio, opt=None, progreso=None):
    entrenando = opt is not None
    modelo.train(entrenando)
    perdida, total = 0.0, 0
    preds, probs, reales = [], [], []

    with torch.set_grad_enabled(entrenando):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(dev), y.to(dev)
            salida = modelo(x)
            loss = criterio(salida, y)

            if entrenando:
                opt.zero_grad()
                loss.backward()
                opt.step()

            perdida += loss.item() * len(y)
            total += len(y)
            p = torch.softmax(salida, dim=1)
            preds += p.argmax(1).cpu().tolist()
            probs += p[:, 1].detach().cpu().tolist()
            reales += y.cpu().tolist()

            if progreso:
                progreso((i + 1) / len(loader))

    acc = accuracy_score(reales, preds)
    return perdida / total, acc, reales, preds, probs


def entrenar(epocas, lr, batch_size, fraccion, congelar):
    try:
        estado.update(estado="entrenando", epoca=0, epocas=epocas, avance=0.0,
                      historial=[], mensaje="Cargando imágenes")
        inicio = time.time()
        dev = dispositivo()
        datos = cargar_datos(batch_size, fraccion)

        modelo = crear_modelo(congelar).to(dev)
        criterio = nn.CrossEntropyLoss(weight=datos["pesos"].to(dev))
        params = [p for p in modelo.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)

        mejor_acc, mejores_pesos = -1, None

        for ep in range(1, epocas + 1):
            estado.update(epoca=ep, mensaje=f"Entrenando época {ep} de {epocas}")

            def avance(frac):
                estado["avance"] = (ep - 1 + frac) / epocas

            l_tr, a_tr, *_ = correr(modelo, datos["train"], dev, criterio, opt, avance)
            l_val, a_val, *_ = correr(modelo, datos["val"], dev, criterio)

            estado["historial"].append({
                "epoca": ep,
                "perdida_train": round(l_tr, 4), "acc_train": round(a_tr, 4),
                "perdida_val": round(l_val, 4), "acc_val": round(a_val, 4),
            })

            if a_val > mejor_acc:
                mejor_acc = a_val
                mejores_pesos = copy.deepcopy(modelo.state_dict())

        # evaluación final con el set de test, que el modelo nunca vio
        estado["mensaje"] = "Evaluando con imágenes de prueba"
        modelo.load_state_dict(mejores_pesos)
        _, _, reales, preds, probs = correr(modelo, datos["test"], dev, criterio)

        id = corridas.nueva_id()
        resultado = {
            "parametros": {"epocas": epocas, "lr": lr, "batch_size": batch_size,
                           "fraccion": fraccion, "congelar": congelar},
            "validacion": "por paciente",
            "datos": datos["info"],
            "dispositivo": str(dev),
            "duracion_seg": round(time.time() - inicio, 1),
            "historial": estado["historial"],
            "test": {
                "exactitud": accuracy_score(reales, preds),
                "precision": precision_score(reales, preds, zero_division=0),
                "sensibilidad": recall_score(reales, preds, zero_division=0),
                "f1": f1_score(reales, preds, zero_division=0),
                "auc": roc_auc_score(reales, probs),
                "imagenes": len(reales),
            },
            "matriz": confusion_matrix(reales, preds).tolist(),
            # se guardan para poder mover el umbral y dibujar la curva ROC
            "prob_test": [round(float(p), 4) for p in probs],
            "real_test": reales,
        }

        carpeta = corridas.carpeta(id)
        carpeta.mkdir(parents=True)
        torch.save(modelo.state_dict(), carpeta / "modelo.pt")
        (carpeta / "metricas.json").write_text(json.dumps(resultado))
        corridas.activar(id)
        estado.update(estado="listo", avance=1.0, mensaje="Entrenamiento terminado")

    except Exception as e:
        estado.update(estado="error", mensaje=str(e))
