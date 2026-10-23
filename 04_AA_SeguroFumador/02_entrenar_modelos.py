# -*- coding: utf-8 -*-
"""
Fase 2: Preprocesamiento (Pipeline) y Entrenamiento de Modelos
Práctica 02 - Seguro Médico
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Creamos las carpetas donde se guardarán los resultados y los modelos
os.makedirs('models/practica02', exist_ok=True)
os.makedirs('results/practica02', exist_ok=True)

# 1. CARGAR DATOS LIMPIOS
# Leemos el archivo que limpiaste en el paso anterior
df = pd.read_csv('datos/ds_processedP2.csv')

# 2. DIVISIÓN DE VARIABLES (X, y)
TARGET = "CostoAnual"
FEATURES = [col for col in df.columns if col != TARGET]

X = df[FEATURES]
y = df[TARGET]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 3. DIVISIÓN ENTRENAMIENTO Y PRUEBA (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(
    f"Entrenando con {len(X_train)} pacientes. Evaluando a ciegas con {len(X_test)} pacientes.\n")

# 4. PREPROCESADOR (ColumnTransformer)
# Aquí configuramos los traductores para los números y las letras
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),

    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

# 5. DEFINIR LOS MODELOS
modelos = {
    'Regresion_Lineal': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'Random_Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'KNN_Regressor': KNeighborsRegressor(n_neighbors=7)
}

# 6. PIPELINE FINAL Y ENTRENAMIENTO
resultados = []

for nombre, algoritmo in modelos.items():
    # Enganchamos el preprocesador con el algoritmo en un solo tubo (Pipeline)
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', algoritmo)
    ])

    # ENTRENAR (Aprender del pasado)
    pipe.fit(X_train, y_train)

    # EVALUAR (Predecir el futuro)
    y_pred_test = pipe.predict(X_test)

    # Calcular cómo le fue (R2 y Error)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    resultados.append({'Modelo': nombre, 'R2 (Acierto)': round(
        r2, 4), 'Error Promedio ($)': round(mae, 2)})
    print(f"Modelo {nombre} entrenado exitosamente.")

# 7. MOSTRAR RESULTADOS
print("\n" + "="*50)
print("COMPETENCIA DE MODELOS (EVALUACIÓN EN PRUEBA)")
print("="*50)
df_eval = pd.DataFrame(resultados)
print(df_eval.to_string(index=False))
