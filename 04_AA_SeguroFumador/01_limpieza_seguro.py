# -*- coding: utf-8 -*-
"""
Fase 1: Auditoría, Inspección Categórica y Limpieza de Datos
Práctica 2 - Estimación de Costo Anual de Seguro Médico
"""

import os
import pandas as pd
import numpy as np

# 1. Rutas relativas del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.join(BASE_DIR, 'datos', 'datapractica2.csv')
RUTA_SALIDA = os.path.join(BASE_DIR, 'datos', 'ds_processedP2.csv')

print("=" * 80)
print("FASE 1: AUDITORÍA INICIAL, ETIQUETAS Y LIMPIEZA — PRÁCTICA 2")
print("=" * 80)

if not os.path.exists(RUTA_ENTRADA):
    raise FileNotFoundError(
        f"No se encontró el archivo fuente en: {RUTA_ENTRADA}")

# =========================================================================
# BLOQUE NUEVO: INSPECCIÓN DE ETIQUETAS Y CATEGORÍAS EN DATOS CRUDOS
# =========================================================================

# 2. Carga del archivo crudo (cadena de custodia)
df_original = pd.read_csv(RUTA_ENTRADA)
df = df_original.copy()

print(f"\n[1] INSPECCIÓN GENERAL DEL DATASET CRUDO")
print(f"Dimensiones iniciales: {df.shape[0]} filas, {df.shape[1]} columnas.")
print("Nombres de columnas detectadas:")
for idx, col in enumerate(df.columns, 1):
    print(f"   {idx}. {col} (Tipo: {df[col].dtype})")

# 3. Lectura de etiquetas y exploración categórica ANTES de la limpieza
cols_categoricas_crudas = df.select_dtypes(
    include=['object', 'category', 'string']).columns.tolist()
print(
    f"\n[2] DETECCIÓN DE VARIABLES CATEGÓRICAS ({len(cols_categoricas_crudas)} encontradas):")
for col in cols_categoricas_crudas:
    valores_unicos = df[col].dropna().unique().tolist()
    total_unicos = len(valores_unicos)
    print(f"\n   • Columna: '{col}' (Total de categorías: {total_unicos})")
    print(f"     Valores presentes: {valores_unicos[:8]}")
    conteo = df[col].value_counts(dropna=False).to_dict()
    conteo_resumen = dict(list(conteo.items())[:5])
    print(f"     Muestra de distribución: {conteo_resumen}")

# 4. Descarte del identificador 'AseguradoID' si existe
cols_id = [c for c in df.columns if c.lower() in ['aseguradoid',
                                                  'id', 'id_cliente']]
if cols_id:
    df.drop(columns=cols_id, inplace=True)
    print(f"\n[3] Identificador descartado: {cols_id}")

# 5. Eliminación de registros duplicados
duplicados = df.duplicated().sum()
if duplicados > 0:
    df.drop_duplicates(inplace=True)
    print(f"[4] Duplicados eliminados: {duplicados} filas.")
else:
    print("[4] No se detectaron filas duplicadas.")

# 6. Conversión robusta de columnas monetarias (IngresosMensuales y CostoAnual)


def limpiar_columna_monetaria(serie):
    return pd.to_numeric(
        serie.astype(str).str.replace(r'[$,\s]', '', regex=True),
        errors='coerce'
    )


for col_moneda in ['IngresosMensuales', 'CostoAnual', 'charges', 'income', 'ingresos']:
    if col_moneda in df.columns:
        df[col_moneda] = limpiar_columna_monetaria(df[col_moneda])
        print(
            f"[5] '{col_moneda}' depurada y convertida exitosamente a numérico decimal.")

# 7. Homologación de nombres si vienen en inglés
mapeo = {
    'age': 'Edad', 'sex': 'Genero', 'bmi': 'IMC', 'children': 'Hijos',
    'smoker': 'Fumador', 'region': 'Region', 'charges': 'CostoAnual',
    'income': 'IngresosMensuales', 'ingresos': 'IngresosMensuales'
}
df.rename(columns=mapeo, inplace=True)

# 8. Depuración de edades imposibles
if 'Edad' in df.columns:
    df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce')
    filas_antes = len(df)
    condicion_edad = (df['Edad'] > 0) & (df['Edad'] <= 105)
    df = df[condicion_edad].copy()
    df['Edad'] = df['Edad'].astype(int)
    print(f"[6] Edades imposibles retiradas: {filas_antes - len(df)} filas.")

# 9. Depuración de nulos
nulos = df.isnull().sum().sum()
if nulos > 0:
    df.dropna(inplace=True)
    print(f"[7] Filas con valores nulos eliminadas: {nulos}.")
else:
    print("[7] No existen valores nulos en el conjunto.")

# 10. Selección y orden estricto de columnas requeridas
COLUMNAS_OBJETIVO = [
    'Edad', 'Genero', 'IMC', 'Hijos', 'Fumador', 'Region', 'IngresosMensuales', 'CostoAnual'
]
cols_finales = [c for c in COLUMNAS_OBJETIVO if c in df.columns]
df_limpio = df[cols_finales].copy()

# Forzar tipos estrictos finales
df_limpio['Edad'] = df_limpio['Edad'].astype(int)
df_limpio['Hijos'] = df_limpio['Hijos'].astype(int)
df_limpio['IMC'] = df_limpio['IMC'].astype(float)
df_limpio['IngresosMensuales'] = df_limpio['IngresosMensuales'].astype(float)
df_limpio['CostoAnual'] = df_limpio['CostoAnual'].astype(float)

# 11. Guardar dataset procesado
df_limpio.to_csv(RUTA_SALIDA, index=False)

# 12. Auditoría Final de Salida
print("\n" + "=" * 80)
print("AUDITORÍA FINAL: DATASET PROCESADO (ds_processedP2.csv)")
print("=" * 80)
print(
    f"Dimensiones finales: {df_limpio.shape[0]} filas, {df_limpio.shape[1]} columnas.\n")

print("Etiquetas y resumen de variables categóricas listas para el modelo:")
for col in df_limpio.select_dtypes(include=['object', 'category', 'string']).columns:
    unicos = df_limpio[col].unique().tolist()
    print(
        f"   • {col}: {unicos} (Distribución: {dict(df_limpio[col].value_counts())})")

print("\nResumen de variables numéricas:")
for col in df_limpio.select_dtypes(include=['int64', 'float64']).columns:
    print(
        f"   • {col}: Rango [{df_limpio[col].min()} - {df_limpio[col].max()}] | Media: {df_limpio[col].mean():.2f}")

print(f"\n[ÉXITO] Archivo listo para la Fase 2 en: {RUTA_SALIDA}")
print("=" * 80)
