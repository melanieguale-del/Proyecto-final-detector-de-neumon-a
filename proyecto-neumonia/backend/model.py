import torch
from torch import nn
from torchvision import models


def dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def crear_modelo(congelar=True, preentrenado=True):
    pesos = models.ResNet18_Weights.DEFAULT if preentrenado else None
    modelo = models.resnet18(weights=pesos)

    if congelar:
        for p in modelo.parameters():
            p.requires_grad = False

    # capa final nueva para 2 clases (se entrena siempre)
    modelo.fc = nn.Linear(modelo.fc.in_features, 2)
    return modelo
