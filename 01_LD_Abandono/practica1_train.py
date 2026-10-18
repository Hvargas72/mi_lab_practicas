import os
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # <-- Importados
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# 1. Recuperar la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent

# 2. CARPETA contenedora donde guardaremos todo
CARPETA_DATOS = ROOT / "datos"
# Crea la carpeta si no existe
CARPETA_DATOS.mkdir(parents=True, exist_ok=True)

# 3. RUTA COMPLETA al archivo de entrada (Aquí se define RUTA_ENTRADA)
PRACTICA = "practica01"
RUTA_ENTRADA = CARPETA_DATOS / "ds_processedP1.csv"

# 4. Lectura del dataset
if RUTA_ENTRADA.exists():
    ds = pd.read_csv(RUTA_ENTRADA)
    print("¡Archivo 'ds_processedP1.csv' cargado con éxito!")
else:
    raise FileNotFoundError(
        f"No se encontró el archivo en la ruta: {RUTA_ENTRADA}")

# definir las  variables predictoras y objetivo
TARGET = 'Abandono'  # variable objetivo
FEATURES = ds.columns.drop(TARGET)  # columna predictora

x = ds[FEATURES]  # variable predictora
y = ds[TARGET]  # variable objetivo

num_cols = x.select_dtypes(include=["int64", "float64"]).columns
cat_cols = x.select_dtypes(include=['object']).columns

# 3. Preprocesamiento de datos (Tuplas sin comillas para que no se lean como cadenas)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

prepocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# Dividir el dataset: 80% entrenamiento, 20% pruebas
# random es un valor aleatorio llamado pibote donde inicia conteo
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)


# modelo de regresion logistica
logistic_model = Pipeline(steps=[
    ("prepocessor", prepocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
logistic_model.fit(x_train, y_train)

# Modelo ramdom forest
rf_model = Pipeline(steps=[
    ("prepocessor", prepocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])
rf_model.fit(x_train, y_train)

# evaluacion de los modelos
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# modelo de regresion logistica
for ax, (nombre_modelo, modelo) in zip(axes, [("Regresion Logistica", logistic_model), ("Random Forest", rf_model)]):
    modelo.fit(x_train, y_train)
    y_pred = modelo.predict(x_test)

    print(f"\n---- Evaluacion del modelo: {nombre_modelo} ---")
    print(f"Accuracy: {modelo.score(x_test, y_test):.4f}")
    print(classification_report(y_test, y_pred))


# 2. GUARDAR EL MODELO ENTRENADO
# =====================================================================
# Definimos la ruta dentro de la carpeta 'practicas'
RUTA_MODELO = BASE_DIR / "modelo_abandono.joblib"

# Guardamos el modelo Random Forest (incluye preprocesamiento + clasificador)
joblib.dump(rf_model, RUTA_MODELO)

print("\n" + "="*50)
print(f"¡ÉXITO! El modelo fue guardado en:\n -> {RUTA_MODELO}")
print("="*50)
