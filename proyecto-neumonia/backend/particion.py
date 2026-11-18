import random
import re
from pathlib import Path


def clave_paciente(ruta):
    # el nombre del archivo trae el paciente: person12_virus_30 o IM-0085-0001
    nombre = Path(ruta).stem
    m = re.match(r"person\d+_(?:bacteria|virus)", nombre)
    if m:
        return m.group(0)
    m = re.search(r"IM-(\d+)", nombre)
    if m:
        return "normal_" + m.group(1)
    return nombre


def separar(rutas, fraccion, semilla=42):
    # reparte por paciente para que nadie quede en entrenamiento y validación a la vez
    grupos = {}
    for i, ruta in enumerate(rutas):
        grupos.setdefault(clave_paciente(ruta), []).append(i)

    claves = sorted(grupos)
    random.Random(semilla).shuffle(claves)

    meta = max(int(len(rutas) * fraccion), 60)
    elegidos, total = [], 0
    for c in claves:
        if total >= meta:
            break
        elegidos.append(c)
        total += len(grupos[c])

    corte = max(int(len(elegidos) * 0.85), 1)
    pac_train, pac_val = elegidos[:corte], elegidos[corte:]
    idx_train = [i for c in pac_train for i in grupos[c]]
    idx_val = [i for c in pac_val for i in grupos[c]]
    return idx_train, idx_val, len(pac_train), len(pac_val)
