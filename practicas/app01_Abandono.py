import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Configuración inicial de la interfaz
st.set_page_config(page_title="Predicción de Abandono",
                   page_icon="🔮", layout="centered")

st.title("🔮 Predicción de Abandono de Clientes")
st.write("Completa los datos del cliente y consulta al modelo.")

# Cargar el modelo guardado de forma segura
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_abandono.joblib"


@st.cache_resource
def cargar_modelo():
    if not MODEL_PATH.exists():
        st.error(f"❌ No se encontró el archivo del modelo en: {MODEL_PATH}")
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None


modelo = cargar_modelo()

if modelo is not None:
    # --- FORMULARIO: SECCIÓN 1 - DATOS DEL CLIENTE ---
    st.subheader("👤 Datos del cliente")
    col1, col2 = st.columns(2)

    with col1:
        genero = st.selectbox("Género", ["Femenino", "Masculino"])
        pareja = st.selectbox("¿Tiene pareja?", ["No", "Sí"])
        antiguedad = st.number_input(
            "Antigüedad (meses)", min_value=0, max_value=120, value=12)

    with col2:
        jubilado = st.selectbox("¿Es jubilado?", [0, 1])
        dependientes = st.selectbox("¿Tiene dependientes?", ["No", "Sí"])

    # --- FORMULARIO: SECCIÓN 2 - SERVICIOS CONTRATADOS ---
    st.subheader("📺 Servicios contratados")
    col3, col4 = st.columns(2)

    with col3:
        servicio_telefonico = st.selectbox("Servicio telefónico", ["No", "Sí"])
        servicio_internet = st.selectbox("Servicio de internet", [
                                         "DSL", "Fibra Óptica", "No"])
        respaldo_online = st.selectbox(
            "Respaldo online", ["No", "Sí", "Sin servicio de internet"])
        soporte_tecnico = st.selectbox(
            "Soporte técnico", ["No", "Sí", "Sin servicio de internet"])
        streaming_peliculas = st.selectbox(
            "Streaming películas", ["No", "Sí", "Sin servicio de internet"])

    with col4:
        lineas_multiples = st.selectbox(
            "Líneas múltiples", ["No", "Sí", "Sin servicio telefónico"])
        seguridad_online = st.selectbox(
            "Seguridad online", ["No", "Sí", "Sin servicio de internet"])
        proteccion_dispositivos = st.selectbox("Protección de dispositivos", [
                                               "No", "Sí", "Sin servicio de internet"])
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Sí", "Sin servicio de internet"])

    st.markdown("---")

    # --- BOTÓN Y PREDICCIÓN ---
    if st.button("Consultar Modelo", type="primary"):
        try:
            # Crear DataFrame con los datos seleccionados
            datos_cliente = pd.DataFrame([{
                'genero': genero,
                'jubilado': jubilado,
                'pareja': pareja,
                'dependientes': dependientes,
                'antiguedad': antiguedad,
                'servicio_telefonico': servicio_telefonico,
                'lineas_multiples': lineas_multiples,
                'servicio_internet': servicio_internet,
                'seguridad_online': seguridad_online,
                'respaldo_online': respaldo_online,
                'proteccion_dispositivos': proteccion_dispositivos,
                'soporte_tecnico': soporte_tecnico,
                'streaming_tv': streaming_tv,
                'streaming_peliculas': streaming_peliculas
            }])

            # Evaluar con el modelo
            prediccion = modelo.predict(datos_cliente)[0]
            probabilidad = modelo.predict_proba(datos_cliente)[0][1]

            # Mostrar resultado gráfico
            if prediccion == 1 or str(prediccion).lower() in ['yes', 'si', '1']:
                st.error(
                    f"⚠️ **Riesgo Elevado**: El cliente tiene un **{probabilidad*100:.1f}%** de probabilidad de abandonar el servicio.")
            else:
                st.success(
                    f"✅ **Cliente Estable**: Probabilidad de retención del **{(1-probabilidad)*100:.1f}%**.")
        except Exception as err:
            st.error(f"⚠️ Error al procesar la predicción: {err}")
