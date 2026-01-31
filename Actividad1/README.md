# 📊 Predicción de Fuga de Clientes (Churn) - Regresión Logística

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-II-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange)

Este repositorio contiene la **Actividad 1** del Magíster en Data Science (UDLA). El proyecto aborda el problema de la deserción de clientes en telecomunicaciones utilizando **Regresión Logística**, comparando la complejidad de modelos polinomiales frente a la estabilidad de modelos regularizados.

## 📋 Contexto del Negocio
Identificar clientes con probabilidad de abandonar la empresa (Churn) es vital para optimizar presupuestos de retención. Este proyecto utiliza un dataset con información demográfica, de servicios y de facturación para clasificar a los usuarios.

## 🎯 Objetivos
* **Construir un Pipeline Robusto:** Automatizar el preprocesamiento y escalado de datos.
* **Ingeniería de Características:** Evaluar el impacto de transformaciones polinomiales de grado 2.
* **Regularización:** Optimizar hiperparámetros $L_1$ (Lasso) y $L_2$ (Ridge) mediante `GridSearchCV`.
* **Métricas de Evaluación:** Analizar el rendimiento con foco en el desbalance de clases (Precision-Recall y ROC-AUC).

## 🛠️ Metodología y Tecnologías

### Stack Técnico
- **Procesamiento:** `Pandas`, `NumPy`.
- **Modelado:** `Scikit-Learn` (Pipelines, ColumnTransformer, StratifiedKFold).
- **Visualización:** `Matplotlib`, `Seaborn`.

### Flujo de Trabajo (Pipeline)
1.  **Limpieza:** Tratamiento de datos faltantes en `TotalCharges` y eliminación de ruido (`customerID`).
2.  **Codificación:** `OneHotEncoder` para variables categóricas.
3.  **Escalado:** `StandardScaler` para asegurar la convergencia del modelo logístico.
4.  **Expansión:** Generación de interacciones entre variables numéricas.
5.  **Validación:** K-Fold estratificado para manejar el desbalance de la clase objetivo.

## 📊 Hallazgos y Resultados

### Comparación de Modelos
A lo largo de la actividad, se observó que:
* **Modelos Polinomiales:** Logran capturar relaciones complejas, pero aumentan drásticamente la dimensionalidad, arriesgando el sobreajuste (*overfitting*).
* **Regularización ($C$):** La aplicación de penalizaciones permitió reducir la magnitud de los coeficientes, mejorando la generalización en datos no vistos.

### Análisis del Trade-off
En el análisis crítico, se determinó que:
* **Campaña Barata (Emails):** Se debe priorizar el **Recall** (detectar la mayor cantidad de fugas).
* **Campaña Costosa (Descuentos):** Se debe priorizar la **Precision** (no gastar en clientes que no se iban a ir).
