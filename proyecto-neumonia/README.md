# Detector de neumonía en radiografías de tórax

Proyecto final de Machine Learning. Clasifica radiografías de tórax como
normales o con neumonía usando transfer learning (ResNet18, PyTorch).

- **Backend:** FastAPI (entrenar, métricas, predecir)
- **Interfaz:** Streamlit
- **Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia),
  Kermany et al., Kaggle. Radiografías pediátricas reales, licencia CC BY 4.0.

## Instalación

```
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu134
pip install -r requirements.txt
python scripts/descargar_dataset.py
```

El comando de torch depende de tu tarjeta. Revisa https://pytorch.org/get-started/locally/

## Ejecución

Con `iniciar.bat`, o a mano en dos terminales:

```
uvicorn backend.main:app --port 8000
streamlit run frontend/app.py
```

La interfaz queda en http://localhost:8501 y la documentación de la API en
http://127.0.0.1:8000/docs

## Cómo funciona

- Cada entrenamiento se guarda como una corrida en `models/corridas/`, y desde la
  pestaña Corridas se elige cuál usa la app.
- La validación se separa por paciente (el nombre del archivo trae el paciente), para que
  imágenes de una misma persona no queden en entrenamiento y validación a la vez.
- El umbral para decir "neumonía" se puede mover en la pestaña Resultados, y se ve su
  efecto en la matriz de confusión y en la curva ROC.

## Estructura

```
backend/    API, modelo, entrenamiento, partición por paciente y corridas
frontend/   interfaz en Streamlit
scripts/    descarga del dataset
data/       imágenes (no se sube a git)
models/     corridas guardadas, cada una con su modelo y métricas (no se sube a git)
docs/       informe y presentación
```

## Integrantes

- (nombre)
