# Actividad1

Predicción de Fuga de Clientes con Regresión Logística
Este proyecto tiene como objetivo desarrollar un modelo de clasificación para predecir qué clientes tienen mayor probabilidad de abandonar una empresa de telecomunicaciones (churn). El análisis compara diferentes enfoques de Regresión Logística, evaluando el impacto del preprocesamiento, el uso de transformaciones polinomiales y la aplicación de técnicas de regularización.

## 📋 Contexto del Negocio
Una empresa de telecomunicaciones busca diseñar campañas de retención específicas. Para ello, se utiliza un dataset que contiene características demográficas y de uso de los clientes, junto con una etiqueta binaria que indica si el cliente se mantuvo o se fue (Churn = 1).

## 🎯 Objetivos
Construir y evaluar un modelo de regresión logística con características básicas.

Evaluar un modelo con transformaciones polinomiales sobre variables numéricas para capturar relaciones no lineales.

Aplicar modelos con penalización (L1/L2) para controlar la complejidad y el sobreajuste.

Determinar la mejor métrica de evaluación (F1, AUC-ROC, PR-AUC) dada la naturaleza desbalanceada del problema.

## 🛠️ Metodología y Tecnologías
La actividad se desarrolló utilizando Python y las siguientes librerías principales:

Scikit-Learn: Para el modelado, validación cruzada (StratifiedKFold) y búsqueda de hiperparámetros (GridSearchCV).

Pandas y NumPy: Para la manipulación y análisis de datos.

Matplotlib: Para la visualización de curvas ROC y Precision-Recall.

Pasos del Pipeline:
Preprocesamiento: Tratamiento de valores nulos (columna TotalCharges), eliminación de identificadores innecesarios (customerID) y codificación de variables categóricas.

Transformación: Escalado de variables numéricas y generación de características polinomiales.

Modelado: Entrenamiento de modelos de Regresión Logística básica y regularizada.

Validación: Uso de validación cruzada k-fold para asegurar la estabilidad de las métricas.

## 📊 Variables Principales
El dataset incluye, entre otras:

Demográficas: Género, si es adulto mayor, dependientes.

Servicios: Internet (DSL/Fibra), seguridad online, soporte técnico, streaming.

Cuenta: Tenencia (tenure), tipo de contrato, cargos mensuales y totales.

Curso: Machine Learning II - Magíster en Data Science 2025

Institución: Universidad de las Américas (UDLA)
