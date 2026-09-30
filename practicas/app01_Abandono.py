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
    st.subheader("👤 Datos Personales y Contrato")
    col1, col2 = st.columns(2)

    with col1:
        genero = st.selectbox("Género", ["Femenino", "Masculino"])
        socio = st.selectbox("¿Tiene pareja / socio?", ["No", "Sí"])
        dependientes = st.selectbox("¿Tiene dependientes?", ["No", "Sí"])
        permanencia = st.number_input(
            "Permanencia (meses)", min_value=0, max_value=120, value=12)
        contrato = st.selectbox("Tipo de contrato", [
                                "Mes a mes", "Un año", "Dos años"])

    with col2:
        jubilado = st.selectbox("¿Es jubilado?", [0, 1])
        facturacion = st.selectbox("Facturación electrónica", ["No", "Sí"])
        metodo_pago = st.selectbox("Método de pago", [
            "Cheque electrónico",
            "Cheque enviado",
            "Transferencia bancaria",
            "Tarjeta de crédito"
        ])
        cargo_mensual = st.number_input(
            "Cargo Mensual ($)", min_value=0.0, max_value=500.0, value=70.0)
        total_cargos = st.number_input(
            "Total Cargos ($)", min_value=0.0, max_value=10000.0, value=840.0)

    # --- FORMULARIO: SECCIÓN 2 - SERVICIOS CONTRATADOS ---
    st.subheader("📺 Servicios Contratados")
    col3, col4 = st.columns(2)

    with col3:
        servicio_telefonico = st.selectbox("Servicio telefónico", ["No", "Sí"])
        lineas_multiples = st.selectbox(
            "Líneas múltiples", ["No", "Sí", "Sin servicio telefónico"])
        internet_service = st.selectbox("Servicio de internet", [
                                        "DSL", "Fibra Óptica", "No"])
        seguridad_online = st.selectbox(
            "Seguridad online", ["No", "Sí", "Sin servicio de internet"])
        respaldo_online = st.selectbox(
            "Respaldo online", ["No", "Sí", "Sin servicio de internet"])

    with col4:
        proteccion_dispositivos = st.selectbox("Protección de dispositivos", [
                                               "No", "Sí", "Sin servicio de internet"])
        tech_support = st.selectbox("Soporte técnico (TechSupport)", [
                                    "No", "Sí", "Sin servicio de internet"])
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Sí", "Sin servicio de internet"])
        streaming_movies = st.selectbox(
            "Streaming películas", ["No", "Sí", "Sin servicio de internet"])

    st.markdown("---")

    # --- BOTÓN Y PREDICCIÓN ---
    if st.button("Consultar Modelo", type="primary"):
        try:
            # Estructurar el DataFrame EXACTAMENTE con los nombres de columna que espera el modelo
            datos_cliente = pd.DataFrame([{
                'Genero': genero,
                'Jubilado': jubilado,
                'Socio': socio,
                'Dependientes': dependientes,
                'Permanencia': permanencia,
                'ServicioTelefonico': servicio_telefonico,
                'LineasMultiples': lineas_multiples,
                'InternetService': internet_service,
                'SeguridadOnline': seguridad_online,
                'RespaldoOnline': respaldo_online,
                'ProteccionDispositivos': proteccion_dispositivos,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contrato': contrato,
                'Facturacion': facturacion,
                'MetodoPago': metodo_pago,
                'CargoMensual': cargo_mensual,
                'TotalCargos': total_cargos
            }])

            # Predecir con el pipeline del modelo
            prediccion = modelo.predict(datos_cliente)[0]
            probabilidad = modelo.predict_proba(datos_cliente)[0][1]

            if prediccion == 1 or str(prediccion).lower() in ['yes', 'si', '1']:
                st.error(
                    f"⚠️ **Riesgo Elevado**: El cliente tiene un **{probabilidad*100:.1f}%** de probabilidad de abandonar el servicio.")
            else:
                st.success(
                    f"✅ **Cliente Estable**: Probabilidad de retención del **{(1-probabilidad)*100:.1f}%**.")
        except Exception as err:
            st.error(f"⚠️ Error al procesar la predicción: {err}")
