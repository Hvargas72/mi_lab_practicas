import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# =============================================================================
# ETAPA 1: DEFINICIÓN DE UNIVERSOS DE DISCURSO Y VARIABLES LINGÜÍSTICAS
# =============================================================================

# Entradas (Antecedents):
# Calidad del servicio: Escala de 0 a 10 puntos
calidad = ctrl.Antecedent(np.arange(0, 11, 1), 'calidad')

# Tiempo de espera: Rango de 0 a 60 minutos
tiempo_espera = ctrl.Antecedent(np.arange(0, 61, 1), 'tiempo_espera')

# Salida (Consequent):
# Nivel de satisfacción: Porcentaje de 0 a 100 %
satisfaccion = ctrl.Consequent(np.arange(0, 101, 1), 'satisfaccion')

# =============================================================================
# ETAPA 2: CONFIGURACIÓN DE FUNCIONES DE PERTENENCIA (FUZZIFICACIÓN)
# =============================================================================

# Calidad: Mala, Regular, Excelente
calidad['mala'] = fuzz.trapmf(calidad.universe, [0, 0, 2, 4])
calidad['regular'] = fuzz.trimf(calidad.universe, [3, 5, 7])
calidad['excelente'] = fuzz.trapmf(calidad.universe, [6, 8, 10, 10])

# Tiempo de espera: Corto, Aceptable, Prolongado
tiempo_espera['corto'] = fuzz.trapmf(tiempo_espera.universe, [0, 0, 10, 20])
tiempo_espera['aceptable'] = fuzz.trimf(tiempo_espera.universe, [15, 25, 35])
tiempo_espera['prolongado'] = fuzz.trapmf(
    tiempo_espera.universe, [30, 45, 60, 60])

# Satisfacción: Baja, Media, Alta
satisfaccion['baja'] = fuzz.trapmf(satisfaccion.universe, [0, 0, 20, 40])
satisfaccion['media'] = fuzz.trimf(satisfaccion.universe, [30, 50, 70])
satisfaccion['alta'] = fuzz.trapmf(satisfaccion.universe, [60, 80, 100, 100])

# =============================================================================
# ETAPA 3: BASE DE CONOCIMIENTO (REGLAS DE INFERENCIA MAMDANI)
# =============================================================================

# R1: Pésimo servicio o espera excesiva provocan baja satisfacción
regla1 = ctrl.Rule(
    calidad['mala'] | tiempo_espera['prolongado'], satisfaccion['baja'])

# R2: Servicio regular con espera razonable produce satisfacción media
regla2 = ctrl.Rule(calidad['regular'] &
                   tiempo_espera['aceptable'], satisfaccion['media'])

# R3: Excelente servicio y poco tiempo de espera garantizan alta satisfacción
regla3 = ctrl.Rule(calidad['excelente'] &
                   tiempo_espera['corto'], satisfaccion['alta'])

# R4: Buen servicio que compensa una espera moderada
regla4 = ctrl.Rule(calidad['excelente'] &
                   tiempo_espera['aceptable'], satisfaccion['media'])

# R5: Espera breve pero con servicio regular
regla5 = ctrl.Rule(calidad['regular'] &
                   tiempo_espera['corto'], satisfaccion['media'])

# =============================================================================
# ETAPA 4: MOTOR DE CONTROL Y SIMULACIÓN
# =============================================================================

sistema_control = ctrl.ControlSystem([regla1, regla2, regla3, regla4, regla5])
simulacion = ctrl.ControlSystemSimulation(sistema_control)

# =============================================================================
# ETAPA 5: INYECCIÓN DE VALORES Y DEFUZZIFICACIÓN (CENTROIDE)
# =============================================================================

# Definimos los valores de prueba en variables directas
val_calidad = 7.5
val_tiempo = 12

# Inyectamos los valores al motor difuso
simulacion.input['calidad'] = val_calidad
simulacion.input['tiempo_espera'] = val_tiempo

# Computar motor difuso
simulacion.compute()

# Obtención del resultado defuzzificado
resultado_satisfaccion = simulacion.output['satisfaccion']

print("=" * 60)
print(" SISTEMA DIFUSO MAMDANI: EVALUACIÓN DE SATISFACCIÓN ")
print("=" * 60)
print(f"Calidad del Servicio ingresada : {val_calidad} / 10")
print(f"Tiempo de Espera registrado     : {val_tiempo} minutos")
print("-" * 60)
print(f"Índice de Satisfacción Calculado: {resultado_satisfaccion:.2f}%")
print("=" * 60)
