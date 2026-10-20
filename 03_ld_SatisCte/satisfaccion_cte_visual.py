import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# -------------------------------------------------------------
# Configuración de la interfaz visual
# -------------------------------------------------------------
st.set_page_config(page_title="Simulador de Satisfacción Difusa",
                   page_icon="⭐", layout="centered")

st.title("⭐ Sistema Difuso Mamdani")
st.subheader("Evaluación de Satisfacción del Cliente")
st.markdown(
    "Ajusta los controles interactivos para observar cómo responde el motor de lógica difusa.")

# -------------------------------------------------------------
# Controles interactivos (Sliders / Deslizadores)
# -------------------------------------------------------------
st.sidebar.header("🔧 Parámetros de Entrada")

val_calidad = st.sidebar.slider(
    "Calidad del Servicio (0 a 10):",
    min_value=0.0,
    max_value=10.0,
    value=7.5,
    step=0.5
)

val_tiempo = st.sidebar.slider(
    "Tiempo de Espera (minutos):",
    min_value=0,
    max_value=60,
    value=12,
    step=1
)

# -------------------------------------------------------------
# Motor de Lógica Difusa (Mamdani)
# -------------------------------------------------------------
# Universos de discurso
calidad = ctrl.Antecedent(np.arange(0, 11, 1), 'calidad')
tiempo_espera = ctrl.Antecedent(np.arange(0, 61, 1), 'tiempo_espera')
satisfaccion = ctrl.Consequent(np.arange(0, 101, 1), 'satisfaccion')

# Funciones de pertenencia
calidad['mala'] = fuzz.trapmf(calidad.universe, [0, 0, 2, 4])
calidad['regular'] = fuzz.trimf(calidad.universe, [3, 5, 7])
calidad['excelente'] = fuzz.trapmf(calidad.universe, [6, 8, 10, 10])

tiempo_espera['corto'] = fuzz.trapmf(tiempo_espera.universe, [0, 0, 10, 20])
tiempo_espera['aceptable'] = fuzz.trimf(tiempo_espera.universe, [15, 25, 35])
tiempo_espera['prolongado'] = fuzz.trapmf(
    tiempo_espera.universe, [30, 45, 60, 60])

satisfaccion['baja'] = fuzz.trapmf(satisfaccion.universe, [0, 0, 20, 40])
satisfaccion['media'] = fuzz.trimf(satisfaccion.universe, [30, 50, 70])
satisfaccion['alta'] = fuzz.trapmf(satisfaccion.universe, [60, 80, 100, 100])

# Reglas
regla1 = ctrl.Rule(
    calidad['mala'] | tiempo_espera['prolongado'], satisfaccion['baja'])
regla2 = ctrl.Rule(calidad['regular'] &
                   tiempo_espera['aceptable'], satisfaccion['media'])
regla3 = ctrl.Rule(calidad['excelente'] &
                   tiempo_espera['corto'], satisfaccion['alta'])
regla4 = ctrl.Rule(calidad['excelente'] &
                   tiempo_espera['aceptable'], satisfaccion['media'])
regla5 = ctrl.Rule(calidad['regular'] &
                   tiempo_espera['corto'], satisfaccion['media'])

sistema = ctrl.ControlSystem([regla1, regla2, regla3, regla4, regla5])
simulacion = ctrl.ControlSystemSimulation(sistema)

# Inyectar entradas
simulacion.input['calidad'] = val_calidad
simulacion.input['tiempo_espera'] = val_tiempo

# -------------------------------------------------------------
# Cálculo y Presentación de Resultados
# -------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Calidad Ingresada", value=f"{val_calidad} / 10")
with col2:
    st.metric(label="Tiempo de Espera", value=f"{val_tiempo} min")

st.markdown("---")

try:
    simulacion.compute()
    resultado = simulacion.output['satisfaccion']

    st.subheader(f"Índice de Satisfacción Calculado: **{resultado:.2f}%**")

    # Barra de progreso visual
    st.progress(int(resultado))

    # Diagnóstico interpretativo tipo semáforo
    if resultado >= 70:
        st.success(
            "🟢 **Nivel de Satisfacción: ALTO** — El cliente percibe una experiencia de servicio positiva.")
    elif resultado >= 40:
        st.warning(
            "🟡 **Nivel de Satisfacción: MEDIO** — Percepción regular o compromiso entre calidad y demora.")
    else:
        st.error(
            "🔴 **Nivel de Satisfacción: BAJO** — Servicio deficiente o demora inaceptable.")

except Exception:
    st.info("Ajusta los valores para activar las reglas de inferencia.")
