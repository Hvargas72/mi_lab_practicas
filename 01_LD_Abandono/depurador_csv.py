#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depurador Universal de Datasets para Machine Learning.
Configurado con rutas fijas por defecto para ejecución directa (botón Play).
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def depurar_dataset_universal(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica reglas de limpieza de datos universales y agnósticas al contenido."""
    print("\n🧹 Iniciando depuración universal...")
    df_clean = df.copy()

    # 1. Eliminar columnas residuales de índices o exportaciones previas
    cols_a_eliminar = [
        c for c in df_clean.columns if "unnamed" in c.lower() or c.lower() == "index"
    ]
    if cols_a_eliminar:
        df_clean = df_clean.drop(columns=cols_a_eliminar)
        print(f"   - Columnas de índice eliminadas: {cols_a_eliminar}")

    # 2. Limpieza de espacios en blanco en columnas de tipo texto (String)
    cols_texto = df_clean.select_dtypes(include=['object', 'string']).columns
    for col in cols_texto:
        # Quitar espacios al inicio/final
        df_clean[col] = df_clean[col].astype(str).str.strip()
        # Reemplazar cadenas vacías o espacios por NaN verdaderos de Numpy
        df_clean[col] = df_clean[col].replace(r'^\s*$', np.nan, regex=True)

    # 3. Intentar conversión automática de textos numéricos a Float/Int
    for col in cols_texto:
        convertidos = pd.to_numeric(df_clean[col], errors='coerce')
        # Si la columna convertida tiene números y al menos el 50% es convertible
        if convertidos.notnull().sum() > 0 and convertidos.notnull().sum() >= (len(df_clean) * 0.5):
            df_clean[col] = convertidos

    # 4. Eliminar filas duplicadas exactas
    duplicados_antes = df_clean.duplicated().sum()
    if duplicados_antes > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"   - Filas duplicadas eliminadas: {duplicados_antes}")

    # 5. Reporte de valores nulos
    nulos_totales = df_clean.isnull().sum().sum()
    print(f"   - Valores nulos detectados en total: {nulos_totales}")

    # 6. Eliminar filas con nulos
    df_clean = df_clean.dropna()
    print(
        f"   - Registros finales conservados tras limpiar nulos: {len(df_clean)}")

    return df_clean


def main():
    parser = argparse.ArgumentParser(
        description="Depurador Universal de Datasets CSV con rutas predeterminadas."
    )

    # =========================================================================
    # UBICACIÓN Y NOMBRE DE TUS ARCHIVOS (CONFIGURACIÓN FIJA)
    # =========================================================================
    # BASE_DIR apunta a la carpeta donde está este script (ej. .../practicas)
    BASE_DIR = Path(__file__).resolve().parent

    # Apunta a la carpeta 'datos' que está al lado de 'practicas'
    ruta_entrada_fija = BASE_DIR.parent / "datos" / "BankChurners.csv"
    ruta_salida_fija = BASE_DIR.parent / "datos" / "BankChurners_limpio.csv"
    # =========================================================================

    # Se definen los argumentos con valores 'default' (ya no son obligatorios)
    parser.add_argument(
        "--input",
        type=str,
        default=str(ruta_entrada_fija),
        help="Ruta al archivo CSV de entrada"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ruta_salida_fija),
        help="Ruta de salida para el CSV limpio"
    )

    args = parser.parse_args()
    ruta_in = Path(args.input)

    # Validación de existencia del archivo
    if not ruta_in.exists():
        print(
            f"❌ Error: El archivo no existe en la ruta:\n   {ruta_in.resolve()}")
        sys.exit(1)

    print(f"📂 Cargando archivo: {ruta_in.name}")
    df_raw = pd.read_csv(ruta_in)
    print(
        f"   Dimensiones iniciales: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas.")

    # Ejecutar la limpieza
    df_procesado = depurar_dataset_universal(df_raw)

    # Guardar el resultado
    ruta_out = Path(args.output)
    ruta_out.parent.mkdir(parents=True, exist_ok=True)
    df_procesado.to_csv(ruta_out, index=False)
    print(f"\n💾 Guardado con éxito en:\n   {ruta_out.resolve()}")


if __name__ == "__main__":
    main()
