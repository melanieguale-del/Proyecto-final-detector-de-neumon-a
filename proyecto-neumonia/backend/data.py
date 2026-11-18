import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from backend.config import DATA_DIR
from backend.particion import separar

MEDIA = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

tf_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, STD),
])

tf_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, STD),
])

_raiz = None


def buscar_raiz():
    # el zip de Kaggle a veces trae carpetas anidadas
    global _raiz
    if _raiz is not None:
        return _raiz
    for p in DATA_DIR.rglob("train"):
        if (p / "NORMAL").is_dir() and (p.parent / "test").is_dir():
            _raiz = p.parent
            return _raiz
    return None


def cargar_datos(batch_size, fraccion):
    raiz = buscar_raiz()
    if raiz is None:
        raise FileNotFoundError("No se encontró el dataset en la carpeta data/")

    train_full = datasets.ImageFolder(raiz / "train", tf_train)
    val_full = datasets.ImageFolder(raiz / "train", tf_eval)
    test_ds = datasets.ImageFolder(raiz / "test", tf_eval)

    # la carpeta val del dataset trae solo 16 imágenes, así que validamos
    # con 15% de los pacientes del train
    rutas = [r for r, _ in train_full.samples]
    idx_train, idx_val, pac_train, pac_val = separar(rutas, fraccion)

    # pesos para compensar que hay más casos de neumonía que normales
    conteo = torch.zeros(2)
    for i in idx_train:
        conteo[train_full.targets[i]] += 1
    pesos = conteo.sum() / (2 * conteo)

    def loader(ds, mezclar):
        return DataLoader(ds, batch_size=batch_size, shuffle=mezclar, num_workers=0)

    return {
        "train": loader(Subset(train_full, idx_train), True),
        "val": loader(Subset(val_full, idx_val), False),
        "test": loader(test_ds, False),
        "pesos": pesos,
        "info": {"imagenes_train": len(idx_train), "imagenes_val": len(idx_val),
                 "pacientes_train": pac_train, "pacientes_val": pac_val},
    }
