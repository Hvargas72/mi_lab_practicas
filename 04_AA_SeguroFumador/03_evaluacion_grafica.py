import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# 1. Crear carpetas de salida si no existen
os.makedirs('results/practica02', exist_ok=True)
os.makedirs('models/practica02', exist_ok=True)

# 2. Cargar datos y dividir 80/20
df = pd.read_csv('datos/ds_processedP2.csv')
X = df.drop(columns=['CostoAnual'])
y = df['CostoAnual']

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. Preprocesador
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(
        strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
     ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# 4. Modelos a evaluar
modelos = {
    'Regresion_Lineal': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'Random_Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'KNN_Regressor': KNeighborsRegressor(n_neighbors=7)
}

# 5. Entrenar, Graficar y Guardar
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (nombre, algoritmo) in enumerate(modelos.items()):
    pipe = Pipeline(
        steps=[('preprocessor', preprocessor), ('regressor', algoritmo)])
    pipe.fit(X_train, y_train)

    # Predecir en prueba
    y_pred = pipe.predict(X_test)

    # Graficar Actividad 5
    sns.scatterplot(x=y_test, y=y_pred,
                    ax=axes[idx], alpha=0.6, color='#1B365D')
    axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(),
                   y_test.max()], 'r--', lw=2)  # Línea ideal
    axes[idx].set_title(f'{nombre}')
    axes[idx].set_xlabel('Costo Real')
    axes[idx].set_ylabel('Predicción')

    # Actividad 8: Guardar cada modelo
    ruta_modelo = f'models/practica02/modelo_{nombre.lower()}.joblib'
    joblib.dump(pipe, ruta_modelo)
    print(f"Modelo guardado en: {ruta_modelo}")

# Ajustar gráficos y guardar imagen
axes[5].axis('off')  # Ocultar el 6to recuadro vacío
plt.tight_layout()
plt.savefig('results/practica02/prediccion_vs_real.png', dpi=300)
print("\n[ÉXITO] Gráficos guardados en results/practica02/prediccion_vs_real.png")
