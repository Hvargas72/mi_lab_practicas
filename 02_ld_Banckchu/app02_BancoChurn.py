#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplicación Web Interactiva para Predicción de Abandono Bancario.
Consume el artefacto entrenado: modelo_banco.joblib
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Configuración básica de la página
st.set_page_config(
    page_title="Sistema de Retención Bancaria",
    page_icon="💳",
    layout="wide"
)

# 1. Cargar el modelo en memoria (con caché para que sea instantáneo)


@st.cache_resource
def cargar_modelo():
    BASE_DIR = Path(__file__).resolve().parent
    ruta = BASE_DIR / "modelo_banco.joblib"
    return joblib.load(ruta)


artefacto = cargar_modelo()
pipeline = artefacto["pipeline"]

st.title("💳 Diagnóstico de Retención de Tarjetahabientes")
st.write("Modelo predictivo activo: **Random Forest** (AUC-ROC: 98.19%)")
st.markdown("---")

# 2. Formulario dividido en 3 columnas
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Perfil Demográfico")
    edad = st.slider("Edad del cliente", 18, 85, 45)
    genero = st.selectbox("Género", ["M", "F"])
    dependientes = st.number_input("Número de dependientes", 0, 10, 2)
    educacion = st.selectbox(
        "Nivel educativo",
        ["High School", "Graduate", "Uneducated", "College",
            "Post-Graduate", "Doctorate", "Unknown"]
    )
    estado_civil = st.selectbox(
        "Estado civil", ["Married", "Single", "Divorced", "Unknown"])
    ingresos = st.selectbox(
        "Categoría de ingresos",
        ["Less than $40K", "$40K - $60K", "$60K - $80K",
            "$80K - $120K", "$120K +", "Unknown"]
    )

with col2:
    st.subheader("📑 Datos de Cuenta")
    tarjeta = st.selectbox("Categoría de tarjeta", [
                           "Blue", "Silver", "Gold", "Platinum"])
    antiguedad = st.slider("Meses de antigüedad (Months on book)", 12, 60, 36)
    productos = st.slider("Total de productos contratados", 1, 6, 3)
    inactividad = st.slider("Meses inactivo en el último año", 0, 6, 1)
    contactos = st.slider(
        "Contactos/quejas con el banco (último año)", 0, 6, 2)

with col3:
    st.subheader("📊 Comportamiento Financiero")
    limite = st.number_input("Límite de crédito ($)", 1000.0, 50000.0, 8500.0)
    saldo_rotativo = st.number_input(
        "Saldo rotativo mantenido ($)", 0.0, 30000.0, 1200.0)
    disponible = limite - saldo_rotativo
    cambio_monto = st.number_input(
        "Cambio monto compras (Q4 vs Q1)", 0.0, 5.0, 0.75)
    total_trans_monto = st.number_input(
        "Monto total transacciones ($)", 100.0, 25000.0, 4200.0)
    total_trans_cant = st.number_input(
        "Número total de transacciones", 1, 200, 65)
    cambio_cant = st.number_input(
        "Cambio cantidad transacciones (Q4 vs Q1)", 0.0, 5.0, 0.70)
    utilizacion = st.slider("Tasa de utilización del crédito", 0.0, 1.0, 0.25)

st.markdown("---")

# 3. Botón de predicción y dictamen
if st.button("🔍 Evaluar Riesgo del Cliente", use_container_width=True):
    # Construir la fila con los mismos nombres de columna que usó el modelo
    datos_cliente = pd.DataFrame([{
        "Customer_Age": edad,
        "Gender": genero,
        "Dependent_count": dependientes,
        "Education_Level": educacion,
        "Marital_Status": estado_civil,
        "Income_Category": ingresos,
        "Card_Category": tarjeta,
        "Months_on_book": antiguedad,
        "Total_Relationship_Count": productos,
        "Months_Inactive_12_mon": inactividad,
        "Contacts_Count_12_mon": contactos,
        "Credit_Limit": limite,
        "Total_Revolving_Bal": saldo_rotativo,
        "Avg_Open_To_Buy": disponible,
        "Total_Amt_Chng_Q4_Q1": cambio_monto,
        "Total_Trans_Amt": total_trans_monto,
        "Total_Trans_Ct": total_trans_cant,
        "Total_Ct_Chng_Q4_Q1": cambio_cant,
        "Avg_Utilization_Ratio": utilizacion
    }])

    # Predecir probabilidad de la clase 1 (Abandono)
    prob_abandono = pipeline.predict_proba(datos_cliente)[0][1]
    porcentaje = prob_abandono * 100

    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        st.metric("Riesgo de Cancelación", f"{porcentaje:.1f}%")

    with col_res2:
        if porcentaje >= 65.0:
            st.error("🚨 **ALERTA CRÍTICA: Alto Riesgo de Abandono**")
            st.write(
                "Sugerencia de retención: Ofrecer exoneración de anualidad o incremento de línea.")
        elif porcentaje >= 35.0:
            st.warning("⚠️ **RIESGO MODERADO: Cliente en Observación**")
            st.write(
                "Sugerencia de retención: Incentivar uso con programa de recompensas en comercios.")
        else:
            st.success("✅ **CLIENTE SEGURO / BAJO RIESGO**")
            st.write(
                "El comportamiento transaccional muestra alta estabilidad y fidelidad.")
