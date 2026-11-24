import streamlit as st
import joblib
import pandas as pd
from groq import Groq
from datetime import datetime

# Funcion para cargar el modelo con st
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_mantenimiento.pkl")

@st.cache_resource
def obtener_api_groq():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

try:
    client = obtener_api_groq()
except Exception as e:
    st.error(f"No se pudo obtener la API {e}")

# Función para detectar dinámicamente un modelo de chat activo en Groq
@st.cache_data(ttl=3600)
def detectar_modelo_chat():
    modelos_disponibles = client.models.list().data
    
    # Palabras clave de modelos que NO son de chat general
    excluidos = ("whisper", "guard", "orpheus", "tts", "vision", "embed")
    # Familias de modelos de texto/chat soportadas en Groq
    familias_chat = ("llama", "qwen", "gemma", "deepseek", "mistral", "gpt-oss")
    
    modelos_chat = [
        m.id for m in modelos_disponibles
        if getattr(m, "active", True)
        and not any(ex in m.id.lower() for ex in excluidos)
        and any(fam in m.id.lower() for fam in familias_chat)
    ]
    
    # Priorizar modelos de chat general potentes si están disponibles
    for preferido in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
        if preferido in modelos_chat:
            return preferido
            
    return modelos_chat[0]

# Configuración básica de página (opcional, mejora la visualización en 4 columnas)
st.set_page_config(page_title="Mantenimiento Predictivo", layout="wide")
# Cargar modelo
modelo = cargar_modelo()

st.title("Prediccion de Mantenimiento")

# --- 4 controles interactivos en st.sidebar ---
st.sidebar.header("Lectura de Sensores")

temperatura_c = st.sidebar.slider(
    "Temperatura (°C)",
    min_value=0.0,
    max_value=150.0,
    value=65.0,
    step=0.5
)

vibracion_mms = st.sidebar.slider(
    "Vibración (mm/s)",
    min_value=0.0,
    max_value=25.0,
    value=3.5,
    step=0.1
)

horas_operacion = st.sidebar.number_input(
    "Horas de operación",
    min_value=0,
    max_value=50000,
    value=1200,
    step=50
)

presion_psi = st.sidebar.slider(
    "Presión (PSI)",
    min_value=0.0,
    max_value=200.0,
    value=90.0,
    step=1.0
)

# 3. Mostrar las lecturas actuales mediante tarjetas visuales st.metric en 4 columnas
st.subheader("Lecturas actuales de los sensores")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Temperatura", value=f"{temperatura_c:.1f} °C")

with col2:
    st.metric(label="Vibración", value=f"{vibracion_mms:.1f} mm/s")

with col3:
    st.metric(label="Horas de operación", value=f"{horas_operacion:,} h")

with col4:
    st.metric(label="Presión", value=f"{presion_psi:.1f} PSI")

st.divider()

# Preparar el DataFrame con el orden y nombres exactos de los sensores
datos_sensores = pd.DataFrame([{
    "temperatura_c": temperatura_c,
    "vibracion_mms": vibracion_mms,
    "horas_operacion": horas_operacion,
    "presion_psi": presion_psi
}])

# 4. Mostrar alertas diferenciadas (0, 1 o 2) junto con predict_proba()
st.subheader("Diagnóstico del equipo")

prediccion = int(modelo.predict(datos_sensores)[0])
probabilidades = modelo.predict_proba(datos_sensores)[0]
prob_predicha = probabilidades[prediccion] * 100

if prediccion == 0:
    st.success(
        f"**Estado 0 — Operación Normal** | Probabilidad estimada: **{prob_predicha:.1f}%**"
    )
elif prediccion == 1:
    st.warning(
        f"**Estado 1 — Advertencia / Mantenimiento Preventivo** | Probabilidad estimada: **{prob_predicha:.1f}%**"
    )
elif prediccion == 2:
    st.error(
        f"**Estado 2 — Alerta Crítica / Riesgo de Falla** | Probabilidad estimada: **{prob_predicha:.1f}%**"
    )

st.divider()

# 5. Sección C: Orden de Trabajo / Dictamen con Groq
st.subheader("Asistente de Mantenimiento Industrial (Groq)")

if st.button("Generar Dictamen / Orden de Trabajo", type="primary"):
    with st.spinner("Consultando al Jefe de Mantenimiento Industrial..."):
        modelo_groq = detectar_modelo_chat()
        folio = f"OT-{datetime.now().strftime('%Y%m%d-%H%M')}"

        system_prompt = (
            "Eres un Jefe de Mantenimiento Industrial con amplia experiencia en confiabilidad "
            "de maquinaria rotativa y sistemas hidráulicos/térmicos. Responde en español, "
            "de forma técnica, clara y directa."
        )

        contexto_sensores = (
            f"Lecturas actuales del equipo:\n"
            f"- Temperatura: {temperatura_c} °C\n"
            f"- Vibración: {vibracion_mms} mm/s\n"
            f"- Horas de operación: {horas_operacion} h\n"
            f"- Presión: {presion_psi} PSI\n"
            f"- Estado predicho por el modelo ML: {prediccion} (Confianza: {prob_predicha:.1f}%)\n"
        )

        if prediccion == 0:
            user_prompt = (
                f"{contexto_sensores}\n"
                "El estado es 0 (Operación Normal). Emite un dictamen breve de monitoreo preventivo "
                "(máximo 1 párrafo corto) confirmando estabilidad operativa y sugerencias ligeras de seguimiento."
            )
        else:
            severidad = "Advertencia (Preventivo)" if prediccion == 1 else "Crítico (Correctivo Urgente)"
            user_prompt = (
                f"{contexto_sensores}\n"
                f"El estado es {prediccion} ({severidad}). Genera una Orden de Trabajo formal en formato Markdown "
                f"que incluya exactamente estas 4 secciones:\n"
                f"1. **Folio:** {folio}\n"
                f"2. **Diagnóstico:** análisis técnico basado en las variables que estén elevadas o fuera de rango.\n"
                f"3. **3 Acciones Técnicas:** lista numerada de 3 intervenciones concretas a ejecutar.\n"
                f"4. **Equipo de Seguridad (EPP):** equipo de protección personal obligatorio para realizar la intervención."
            )

        respuesta = client.chat.completions.create(
            model=modelo_groq,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        resultado_ia = respuesta.choices[0].message.content
        st.caption(f"Modelo activo detectado: `{modelo_groq}`")
        st.markdown(resultado_ia)