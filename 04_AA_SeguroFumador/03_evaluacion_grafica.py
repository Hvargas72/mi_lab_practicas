# -*- coding: utf-8 -*-
"""
Fase 2: Preprocesamiento y Entrenamiento (Tabla Completa de Métricas)
Práctica 02 - Seguro Médico
"""
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. CARGAR DATOS LIMPIOS
df = pd.read_csv('datos/ds_processedP2.csv')

# 2. DIVISIÓN DE VARIABLES (X, y)
TARGET = "CostoAnual"
FEATURES = [col for col in df.columns if col != TARGET]

X = df[FEATURES]
y = df[TARGET]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

# 3. DIVISIÓN ENTRENAMIENTO Y PRUEBA (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. PREPROCESADOR
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

# 6. PIPELINE FINAL, ENTRENAMIENTO Y EVALUACIÓN COMPLETA
resultados = []

for nombre, algoritmo in modelos.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', algoritmo)
    ])
    
    # Entrenar
    pipe.fit(X_train, y_train)
    
    # Hacer predicciones sobre Entrenamiento (para ver si hace trampa/sobreajuste)
    y_pred_train = pipe.predict(X_train)
    # Hacer predicciones sobre Prueba (el examen real)
    y_pred_test = pipe.predict(X_test)
    
    # Calcular métricas
    resultados.append({
        'Modelo': nombre,
        'MAE Train': round(mean_absolute_error(y_train, y_pred_train), 2),
        'MAE Test': round(mean_absolute_error(y_test, y_pred_test), 2),
        'RMSE Train': round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 2),
        'RMSE Test': round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2),
        'R2 Train': round(r2_score(y_train, y_pred_train), 4),
        'R2 Test': round(r2_score(y_test, y_pred_test), 4)
    })

# 7. MOSTRAR LA TABLA COMPLETA
print("\n" + "="*85)
print("TABLA COMPARATIVA COMPLETA (ENTRENAMIENTO VS PRUEBA)")
print("="*85)
df_eval = pd.DataFrame(resultados)
print(df_eval.to_string(index=False))
print("="*85 + "\n")
