import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve)


def en_umbral(reales, probs, umbral):
    reales = np.array(reales)
    preds = (np.array(probs) >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(reales, preds, labels=[0, 1]).ravel()
    return {
        "umbral": float(umbral),
        "exactitud": float(accuracy_score(reales, preds)),
        "precision": float(precision_score(reales, preds, zero_division=0)),
        "sensibilidad": float(recall_score(reales, preds, zero_division=0)),
        "especificidad": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(reales, preds, zero_division=0)),
        "matriz": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def curva_roc(reales, probs):
    fpr, tpr, _ = roc_curve(reales, probs)
    return {"fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr]}
