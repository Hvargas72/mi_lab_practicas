#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 2: Preprocesamiento modular y entrenamiento de modelos.
Dataset: BankChurners_limpio.csv
Objetivo: Predecir Attrition_Flag (Abandono de tarjeta)
"""

from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def entrenar_modelos():
    # 1. Definición de rutas relativas (ACTUALIZADO)
    BASE_DIR = Path(__file__).resolve().parent
    ruta_datos = BASE_DIR / "datos" / "BankChurners_limpio.csv"
    ruta_modelo_salida = BASE_DIR / "modelo_banco.joblib"

    if not ruta_datos.exists():
        print(
            f"❌ Error: No se encontró el archivo en:\n   {ruta_datos.resolve()}")
        print("Verifica haber corrido antes la limpieza de la Etapa 1.")
        sys.exit(1)

    print(f"📂 Cargando dataset: {ruta_datos.name}...")
    df = pd.read_csv(ruta_datos)
    print(f"   Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas.")

    # 2. Descarte preventivo de columnas no predictivas
    cols_a_descartar = [
        c for c in df.columns
        if "CLIENTNUM" in c.upper() or "NAIVE_BAYES" in c.upper() or "UNNAMED" in c.upper()
    ]
    if cols_a_descartar:
        df = df.drop(columns=cols_a_descartar)
        print(f"   - Columnas descartadas: {cols_a_descartar}")

    # 3. Separación de Target (y) y Características (X)
    col_target = "Attrition_Flag"
    if col_target not in df.columns:
        print(
            f"❌ Error: No se encontró la columna '{col_target}' en el archivo.")
        sys.exit(1)

    # 1 = Attrited Customer (abandono), 0 = Existing Customer (activo)
    y = df[col_target].apply(lambda x: 1 if "Attrited" in str(x) else 0)
    X = df.drop(columns=[col_target])

    # 4. Identificación automática de tipos de variables
    cols_numericas = X.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    cols_categoricas = X.select_dtypes(
        include=['object', 'string', 'category']).columns.tolist()

    print(
        f"   - Variables numéricas ({len(cols_numericas)}): {cols_numericas}")
    print(
        f"   - Variables categóricas ({len(cols_categoricas)}): {cols_categoricas}")

    # 5. Partición reproducible en memoria (80% Train / 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n📊 Distribución de partición:")
    print(f"   - Filas de entrenamiento (80%): {X_train.shape[0]}")
    print(f"   - Filas de prueba (20%): {X_test.shape[0]}")

    # 6. Definición del transformador de columnas
    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cols_numericas),
            ("cat", OneHotEncoder(drop='first',
             handle_unknown='ignore'), cols_categoricas)
        ]
    )

    # 7. Modelos a contrastar
    modelos = {
        "Regresion_Logistica": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced",
            random_state=42
        )
    }

    mejor_pipeline = None
    mejor_auc = 0.0
    ganador = ""

    print("\n🚀 Entrenando y evaluando modelos...")
    for nombre, estimador in modelos.items():
        # Pipeline completo: Preprocesamiento + Estimador
        pipe = Pipeline(steps=[
            ("preprocesamiento", preprocesador),
            ("clasificador", estimador)
        ])

        # Se ajusta SOLO con los datos de entrenamiento
        pipe.fit(X_train, y_train)

        # Evaluación con los datos de prueba
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)

        print(f"\n" + "=" * 50)
        print(f"🤖 Modelo: {nombre}")
        print(f"   AUC-ROC: {auc:.4f}")
        print("=" * 50)
        print(classification_report(y_test, y_pred,
              target_names=["Activo (0)", "Abandono (1)"]))

        if auc > mejor_auc:
            mejor_auc = auc
            mejor_pipeline = pipe
            ganador = nombre

    # 8. Empaquetar y guardar el artefacto ganador
    artefacto = {
        "pipeline": mejor_pipeline,
        "cols_numericas": cols_numericas,
        "cols_categoricas": cols_categoricas,
        "modelo_nombre": ganador,
        "auc_roc": mejor_auc
    }

    joblib.dump(artefacto, ruta_modelo_salida)
    print("\n" + "#" * 60)
    print(f"🏆 Modelo ganador seleccionado: {ganador} (AUC: {mejor_auc:.4f})")
    print(f"💾 Guardado con éxito en:\n   {ruta_modelo_salida.resolve()}")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    entrenar_modelos()