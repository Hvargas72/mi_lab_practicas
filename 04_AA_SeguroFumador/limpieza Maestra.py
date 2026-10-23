# -*- coding: utf-8 -*-
"""
PLANTILLA MAESTRA: Depuración, Preprocesamiento y Feature Engineering
Ubicación recomendada: scripts/01_limpieza_avanzada.py

Este script consolida TODAS las técnicas de limpieza, desde las básicas
hasta las avanzadas (Imputación, Codificación y Escalado), listas para
ser integradas antes del entrenamiento de Machine Learning.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def limpieza_maestra(ruta_crudo, ruta_salida):
    print("="*60)
    print("INICIANDO LIMPIEZA MAESTRA Y PREPROCESAMIENTO")
    print("="*60)

    # 1. Cargar datos
    df = pd.read_csv(ruta_crudo)
    print(f"[1] Datos cargados: {df.shape[0]} filas.")

    # 2. Eliminación de Duplicados
    duplicados = df.duplicated().sum()
    if duplicados > 0:
        df = df.drop_duplicates()
        print(f"[2] Eliminados {duplicados} registros duplicados.")

    # 3. Limpieza de Texto y Conversión Numérica (Regex)
    # Ejemplo: Quitar '$' y ',' de columnas monetarias
    for col in ['IngresosMensuales', 'CostoAnual']:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(
                r'[$,\s]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(
                f"[3] Columna monetaria '{col}' limpiada y convertida a decimal.")

    # 4. Eliminación de Anomalías Imposibles (Filtro Lógico)
    if 'Edad' in df.columns:
        df = df[(df['Edad'] > 0) & (df['Edad'] <= 105)]
        print("[4] Edades biológicamente imposibles descartadas.")

    # 5. EL ARTE DE RELLENAR HUECOS (IMPUTACIÓN AVANZADA)
    # En lugar de borrar las filas con nulos (dropna), rellenamos inteligentemente.
    # Numéricas -> Rellenar con la mediana (para no verse afectado por valores extremos)
    cols_num = df.select_dtypes(include=['int64', 'float64']).columns
    imputer_num = SimpleImputer(strategy='median')
    df[cols_num] = imputer_num.fit_transform(df[cols_num])

    # Categóricas -> Rellenar con la moda (el valor más frecuente)
    cols_cat = df.select_dtypes(include=['object', 'category']).columns
    if len(cols_cat) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[cols_cat] = imputer_cat.fit_transform(df[cols_cat])
    print("[5] Valores nulos imputados (rellenados) matemáticamente sin perder filas.")

    # 6. EL TRADUCTOR PARA ROBOTS (ENCODING)
    # Convertir texto ('Femenino', 'Masculino') a binario (0, 1) mediante One-Hot Encoding
    # Pandas tiene una función mágica para esto: get_dummies
    if len(cols_cat) > 0:
        df = pd.get_dummies(df, columns=cols_cat, drop_first=True)
        print("[6] Variables categóricas codificadas a ceros y unos (One-Hot Encoding).")

    # 7. EL ECUALIZADOR (ESCALADO DE VARIABLES)
    # Poner todas las variables numéricas en la misma escala (media 0, varianza 1)
    # Cuidado: no escalamos la variable objetivo (CostoAnual)
    target = 'CostoAnual'
    features_to_scale = [c for c in cols_num if c != target]
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    print("[7] Variables numéricas estandarizadas (StandardScaler).")

    # 8. Guardar Dataset Final Preprocesado
    df.to_csv(ruta_salida, index=False)
    print("="*60)
    print(f"PROCESO TERMINADO. Dataset listo guardado en: {ruta_salida}")
    print(
        f"Dimensiones finales para entrenamiento: {df.shape[0]} filas, {df.shape[1]} columnas.")
    print("="*60)

# Para usarlo:
# limpieza_maestra('datos/datapractica2.csv', 'datos/ds_entrenamiento_final.csv')
