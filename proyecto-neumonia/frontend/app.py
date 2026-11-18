from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API = "http://127.0.0.1:8000"

TINTA = "#12263A"
TEAL = "#1B7F86"
AZUL = "#6B8CAE"
ALERTA = "#C2453B"
OK = "#2E7D5B"

st.set_page_config(page_title="Detector de neumonía", page_icon="🫁", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Newsreader:wght@500;600&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Manrope', sans-serif; }
h1, h2, h3 { font-family: 'Newsreader', serif !important; font-weight: 600 !important;
             color: #12263A; letter-spacing: -0.01em }
.block-container { padding-top: 2rem; max-width: 1180px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeaderActionElements"] { display: none; }

.hero { background: #12263A; border-radius: 14px; padding: 30px 34px; margin-bottom: 26px; }
.hero h1 { color: #FFFFFF; font-size: 2.3rem; margin: 0 0 8px 0; padding: 0; }
.hero p { color: #B9C7D3; margin: 0 0 18px 0; max-width: 620px; font-size: 1rem; line-height: 1.55; }
.chip { display: inline-block; background: #1E3850; color: #E4ECF2; border-radius: 99px;
        padding: 5px 14px; font-size: 0.85rem; margin: 0 8px 0 0; }
.chip i { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
          margin-right: 8px; background: #6E8296; }
.chip i.si { background: #4CC38A; }
.chip i.no { background: #E0715F; }

[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #D5DEE5;
                           border-radius: 10px; padding: 14px 18px; }
[data-testid="stMetricLabel"] p { color: #55697B; font-size: 0.9rem; }

.panel { background: #FFFFFF; border: 1px solid #D5DEE5; border-radius: 10px; padding: 20px 22px; }
.resultado { border-radius: 10px; padding: 22px 24px; background: #FFFFFF;
             border: 1px solid #D5DEE5; border-left: 6px solid var(--c); }
.resultado h2 { margin: 0 0 4px 0; color: var(--c); font-size: 1.9rem; }
.resultado p { margin: 0; color: #55697B; }
.nota { color: #55697B; font-size: 0.88rem; line-height: 1.5; }

button[data-baseweb="tab"] p { font-size: 1rem; font-weight: 600; }
.stButton > button { border-radius: 8px; font-weight: 600; padding: 0.55rem 1.4rem; }
</style>
""", unsafe_allow_html=True)


def api(metodo, ruta, silencioso=False, **kw):
    # devuelve None si falla, y avisa en pantalla (salvo que sea un 404 esperado)
    try:
        r = requests.request(metodo, API + ruta, timeout=kw.pop("timeout", 15), **kw)
        if not r.ok:
            if not (silencioso and r.status_code == 404):
                detalle = r.json().get("detail", "Error en la API")
                st.error(detalle if isinstance(detalle, str) else "Error en la API")
            return None
        return r.json()
    except requests.ConnectionError:
        return None


def layout_base(fig, alto=300):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=40, b=70),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Manrope", color=TINTA),
                      legend=dict(orientation="h", yanchor="top", y=-0.32, x=0))
    fig.update_xaxes(gridcolor="#E1E8ED")
    fig.update_yaxes(gridcolor="#E1E8ED")
    return fig


def curvas(hist):
    ep = [h["epoca"] for h in hist]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pérdida", "Exactitud"))
    for col, (k_tr, k_val) in enumerate([("perdida_train", "perdida_val"),
                                         ("acc_train", "acc_val")], start=1):
        fig.add_trace(go.Scatter(x=ep, y=[h[k_tr] for h in hist], name="Entrenamiento",
                                 line=dict(color=TEAL, width=3), showlegend=col == 1), 1, col)
        fig.add_trace(go.Scatter(x=ep, y=[h[k_val] for h in hist], name="Validación",
                                 line=dict(color=AZUL, width=3, dash="dot"),
                                 showlegend=col == 1), 1, col)
    fig.update_xaxes(title_text="Época", dtick=1)
    return layout_base(fig)


def grafica_roc(ev):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ev["roc"]["fpr"], y=ev["roc"]["tpr"], mode="lines",
                             name="Modelo", line=dict(color=TEAL, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Al azar",
                             line=dict(color="#9FB0BF", width=2, dash="dot")))
    # el punto rojo es donde está el umbral que elegiste
    fig.add_trace(go.Scatter(x=[1 - ev["especificidad"]], y=[ev["sensibilidad"]],
                             mode="markers", name="Umbral elegido",
                             marker=dict(size=13, color=ALERTA)))
    fig.update_xaxes(title_text="Normales marcadas como neumonía", range=[0, 1])
    fig.update_yaxes(title_text="Neumonías detectadas", range=[0, 1.02])
    return layout_base(fig, 340)


def fecha_corrida(id):
    try:
        return datetime.strptime(id[:15], "%Y%m%d-%H%M%S").strftime("%d/%m %H:%M")
    except ValueError:
        return id


def nombre_gpu(nombre, hay_gpu):
    if not hay_gpu:
        return "Sin GPU, entrena con CPU"
    corto = nombre.replace("NVIDIA GeForce ", "").replace(" Laptop GPU", "")
    return "GPU " + corto


def chip(texto, ok):
    return f'<span class="chip"><i class="{"si" if ok else "no"}"></i>{texto}</span>'


# ---- estado general ----
est = api("GET", "/estado")
if est is None:
    st.error("No se puede conectar con la API. Inicia el backend con: "
             "uvicorn backend.main:app --port 8000")
    st.stop()

st.markdown(f"""
<div class="hero">
  <h1>Detector de neumonía en radiografías</h1>
  <p>Entrena el modelo con radiografías de tórax reales y pruébalo con imágenes nuevas.</p>
  {chip(nombre_gpu(est["dispositivo"], est["gpu"]), est["gpu"])}
  {chip("Imágenes cargadas" if est["dataset"] else "Faltan las imágenes", est["dataset"])}
  {chip("Modelo entrenado" if est["modelo"] else "Sin modelo todavía", est["modelo"])}
</div>
""", unsafe_allow_html=True)

tab_entrenar, tab_resultados, tab_diag, tab_corridas = st.tabs(
    ["Entrenamiento", "Resultados", "Diagnóstico", "Corridas"])

# ---- pestaña 1: entrenamiento ----
with tab_entrenar:
    izq, der = st.columns([1, 1.5], gap="large")

    with izq:
        st.subheader("Parámetros")
        epocas = st.slider("Épocas", 1, 15, 5)
        lr = st.select_slider("Tasa de aprendizaje", [0.0001, 0.0003, 0.001, 0.003], value=0.0003)
        batch = st.select_slider("Tamaño de lote", [16, 32, 64], value=32)
        fraccion = st.slider("Porcentaje de imágenes a usar", 10, 100, 100, step=10)
        congelar = st.toggle("Entrenar solo la última capa", value=True,
                             help="Es más rápido. Si lo apagas, se ajusta toda la red.")

        en_curso = est["entrenamiento"] == "entrenando"
        if st.button("Entrenar modelo", type="primary", disabled=en_curso, width="stretch"):
            r = api("POST", "/entrenar", json={"epocas": epocas, "lr": lr, "batch_size": batch,
                                              "fraccion": fraccion / 100, "congelar": congelar})
            if r:
                st.session_state["vigilando"] = True
                st.rerun()
        st.caption("Cada entrenamiento se guarda como una corrida nueva, sin borrar las anteriores.")

    with der:
        st.subheader("Progreso")

        def panel_progreso():
            p = api("GET", "/progreso")
            if not p or p["estado"] == "inactivo":
                st.markdown('<div class="panel nota">Cuando inicies el entrenamiento, '
                            'aquí verás el avance y las curvas de aprendizaje.</div>',
                            unsafe_allow_html=True)
                return

            if p["estado"] == "error":
                st.error(f"El entrenamiento falló: {p['mensaje']}")
            else:
                st.progress(min(p["avance"], 1.0), text=p["mensaje"])
            if p["historial"]:
                st.plotly_chart(curvas(p["historial"]), width="stretch",
                                key="curvas_progreso")

            # al terminar, recarga toda la página para actualizar el resto
            if p["estado"] != "entrenando" and st.session_state.get("vigilando"):
                st.session_state["vigilando"] = False
                st.rerun()

        st.fragment(run_every=2 if en_curso else None)(panel_progreso)()

# ---- pestaña 2: resultados ----
with tab_resultados:
    m = api("GET", "/metricas", silencioso=True)
    if not m:
        st.info("Aún no hay resultados. Entrena un modelo en la pestaña Entrenamiento.")
    else:
        t = m["test"]
        con_umbral = m["tiene_roc"]
        st.caption(f"Corrida del {fecha_corrida(m['id'])}. Evaluada con {t['imagenes']} "
                   "radiografías que el modelo no vio al entrenar.")

        umbral = st.slider("Umbral para decir neumonía", 0.10, 0.90, 0.50, 0.05,
                           key="umbral", disabled=not con_umbral,
                           help="Qué tan segura debe estar la red para marcar una radiografía "
                                "como neumonía. Con 50% decide por mayoría.")
        u = umbral if con_umbral else 0.5
        st.session_state["umbral_en_uso"] = u  # el diagnóstico usa este mismo valor

        ev = api("GET", "/evaluar", params={"umbral": u}, silencioso=True) if con_umbral else None
        if not con_umbral:
            st.caption("Esta corrida viene de una versión anterior y no guardó las "
                       "probabilidades, por eso el umbral queda fijo en 50%.")

        if ev:
            z, v = ev["matriz"], ev
        else:
            z, v = m["matriz"], t
        espec = z[0][0] / max(z[0][0] + z[0][1], 1)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Exactitud", f"{v['exactitud']:.1%}")
        c2.metric("Precisión", f"{v['precision']:.1%}")
        c3.metric("Sensibilidad", f"{v['sensibilidad']:.1%}")
        c4.metric("Especificidad", f"{espec:.1%}")
        c5.metric("F1", f"{v['f1']:.1%}")

        a, b = st.columns([1, 1], gap="large")
        with a:
            st.subheader("Matriz de confusión")
            fig = go.Figure(go.Heatmap(
                z=z, x=["Normal", "Neumonía"], y=["Normal", "Neumonía"],
                colorscale=[[0, "#FFFFFF"], [1, TEAL]], showscale=False,
                text=z, texttemplate="%{text}", textfont=dict(size=22)))
            fig.update_xaxes(title_text="Lo que predijo el modelo")
            fig.update_yaxes(title_text="Diagnóstico real", autorange="reversed")
            st.plotly_chart(layout_base(fig, 340), width="stretch", key="matriz")
        with b:
            st.subheader(f"Curva ROC (AUC {t['auc']:.3f})")
            if ev:
                st.plotly_chart(grafica_roc(ev), width="stretch", key="roc")
            else:
                st.info("Entrena una corrida nueva para ver la curva ROC.")

        p = m["parametros"]
        d = m.get("datos")
        st.caption(f"{p['epocas']} épocas, tasa {p['lr']}, lote {p['batch_size']}, "
                   f"{int(p['fraccion'] * 100)}% de las imágenes, "
                   f"{'solo la última capa' if p['congelar'] else 'toda la red'}, "
                   f"{m['duracion_seg']} s en {m['dispositivo']}.")
        if d:
            st.caption(f"Validación por paciente: {d['imagenes_train']} imágenes de "
                       f"{d['pacientes_train']} pacientes para entrenar y {d['imagenes_val']} de "
                       f"{d['pacientes_val']} pacientes para validar, sin repetir pacientes.")
        else:
            st.caption("La validación de esta corrida se tomó al azar entre las imágenes de "
                       "entrenamiento, así que pudo mezclar imágenes de un mismo paciente.")

        # cerrado por defecto para que la pantalla quede limpia al presentar
        with st.expander("¿Cómo leer estas métricas?"):
            st.markdown("""
**Sensibilidad**: de todos los casos con neumonía, cuántos detectó el modelo.
Es la más importante aquí, porque un caso que se escapa es el error más costoso.

**Especificidad**: de las radiografías normales, cuántas reconoció como normales.

**Precisión**: cuando el modelo dice neumonía, cuántas veces acierta.

**AUC**: qué tan bien separa las dos clases sin importar el umbral.
Mientras más cerca de 1, mejor.

**Umbral**: si lo subes, marca menos normales como neumonía pero se le pueden escapar
más casos reales. Si lo bajas, pasa lo contrario. El punto rojo de la curva ROC
muestra dónde estás.
""")

        st.subheader("Curvas de aprendizaje")
        st.plotly_chart(curvas(m["historial"]), width="stretch",
                        key="curvas_resultados")

# ---- pestaña 3: diagnóstico ----
with tab_diag:
    if not est["modelo"]:
        st.info("Primero entrena un modelo para poder analizar radiografías.")
    else:
        u_diag = st.session_state.get("umbral_en_uso", 0.5)
        subida = st.file_uploader("Sube una radiografía de tórax", type=["png", "jpg", "jpeg"])
        if subida:
            col_img, col_res = st.columns([1, 1], gap="large")
            col_img.image(subida, width="stretch")

            with col_res:
                if st.button("Analizar radiografía", type="primary"):
                    r = api("POST", "/predecir", timeout=60, params={"umbral": u_diag},
                            files={"file": (subida.name, subida.getvalue(), subida.type)})
                    if r:
                        hay = r["diagnostico"] == "PNEUMONIA"
                        color = ALERTA if hay else OK
                        titulo = "Posible neumonía" if hay else "Sin signos de neumonía"
                        st.markdown(f"""
<div class="resultado" style="--c:{color}">
  <h2>{titulo}</h2>
  <p>Probabilidad de neumonía: {r['probabilidad']:.1%}</p>
</div>""", unsafe_allow_html=True)
                        st.write("")
                        st.progress(r["probabilidad"],
                                    text=f"Umbral de decisión: {r['umbral']:.0%}")
                        st.markdown('<p class="nota">Proyecto académico. Este resultado no '
                                    'reemplaza el diagnóstico de un profesional de la salud.</p>',
                                    unsafe_allow_html=True)

# ---- pestaña 4: corridas ----
with tab_corridas:
    lista = api("GET", "/corridas")
    if not lista:
        st.info("Todavía no hay corridas guardadas.")
    else:
        filas = []
        for c in lista:
            p, t = c["parametros"], c["test"]
            filas.append({
                "Activa": "✓" if c["activa"] else "",
                "Fecha": fecha_corrida(c["id"]),
                "Entrenó": "Solo última capa" if p["congelar"] else "Toda la red",
                "Épocas": p["epocas"],
                "Tasa": p["lr"],
                "Lote": p["batch_size"],
                "Imágenes": f"{int(p['fraccion'] * 100)}%",
                "Validación": "Por paciente" if c["validacion"] == "por paciente" else "Al azar",
                "Exactitud": round(t["exactitud"] * 100, 1),
                "Precisión": round(t["precision"] * 100, 1),
                "Sensibilidad": round(t["sensibilidad"] * 100, 1),
                "F1": round(t["f1"] * 100, 1),
                "AUC": round(t["auc"], 3),
                "Tiempo": f"{c['duracion_seg']:.0f} s",
            })
        df = pd.DataFrame(filas)
        st.subheader("Comparación de corridas")
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption("Exactitud, precisión, sensibilidad y F1 están en porcentaje. "
                   "Las métricas son sobre las radiografías de prueba.")
        st.download_button("Descargar tabla (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           "corridas.csv", "text/csv")

        st.subheader("Modelo que usa la app")
        etiquetas = {
            f"{fecha_corrida(c['id'])} - {c['parametros']['epocas']} épocas, "
            f"{'última capa' if c['parametros']['congelar'] else 'toda la red'}, "
            f"exactitud {c['test']['exactitud']:.1%}": c["id"] for c in lista}
        ids_lista = list(etiquetas.values())
        actual = next(c["id"] for c in lista if c["activa"])
        eleccion = st.selectbox("Elige una corrida", list(etiquetas),
                                index=ids_lista.index(actual))
        if etiquetas[eleccion] != actual:
            if st.button("Usar este modelo", type="primary"):
                if api("POST", f"/corridas/{etiquetas[eleccion]}/activar"):
                    st.rerun()
        else:
            st.caption("Este es el modelo que se está usando en Resultados y Diagnóstico.")
