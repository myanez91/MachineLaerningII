Actividad 2 

Modelos basados en Árboles y Ensambles para Predicción de Churn

📌 Descripción general
Este proyecto aborda el problema de churn de clientes mediante el uso de modelos no lineales basados en árboles, extendiendo el análisis desarrollado previamente con modelos lineales. El trabajo se centra en Árboles de Decisión y Random Forest, con énfasis en la optimización de hiperparámetros, la evaluación de desempeño en contextos de clases desbalanceadas y el análisis de la varianza de las predicciones.

🎯 Objetivos

Ajustar y optimizar un árbol de decisión de clasificación usando validación cruzada.
Comparar Grid Search CV y Random Search CV en términos de desempeño y costo computacional.
Visualizar e interpretar el árbol de decisión seleccionado.
Analizar el efecto del número de árboles en la estabilidad y varianza de un Random Forest.
Seleccionar y evaluar el mejor Random Forest mediante búsqueda de hiperparámetros.
Comparar modelos utilizando métricas robustas para problemas de clases desbalanceadas.
Discutir resultados desde una perspectiva técnica y de negocio.


🧠 Metodología
1. Árbol de decisión

Implementación de un DecisionTreeClassifier utilizando el mismo dataset y preprocesamiento de la Actividad 1.
Definición y justificación de una grilla de hiperparámetros (max_depth, min_samples_split, min_samples_leaf, criterion).
Optimización mediante:

Grid Search Cross-Validation
Random Search Cross-Validation


Selección del modelo usando F1-score y PR-AUC de la clase churn.
Comparación de desempeño y tiempos de ejecución.

2. Interpretación del modelo

Visualización del árbol óptimo usando plot_tree.
Análisis de las variables más relevantes en las primeras divisiones.
Comparación de interpretabilidad frente a la regresión logística.

3. Random Forest y análisis de varianza

Implementación de validación cruzada k-fold estratificada.
Entrenamiento de Random Forest con distintos números de árboles (2 a 128).
Cálculo de:

Varianza de las probabilidades predichas entre folds.
Métricas de desempeño (F1, AUC-ROC, PR-AUC).


Análisis gráfico de:

Varianza vs. número de árboles.
Métricas vs. número de árboles.



4. Selección del mejor Random Forest

Definición de grilla de hiperparámetros (n_estimators, max_depth, min_samples_leaf, max_features).
Selección del mejor modelo mediante búsqueda con validación cruzada.
Comparación final entre Árbol de Decisión y Random Forest.

5. Evaluación y análisis crítico

Uso de pesos por clase para manejar el desbalance.
Reporte de métricas finales:

Accuracy
Precision
Recall
F1-score
AUC-ROC
PR-AUC


Análisis de curvas ROC y Precision–Recall.
Discusión sobre:

Reducción de varianza en modelos de ensamble.
Trade-off entre interpretabilidad y desempeño.
Métricas más relevantes desde una perspectiva de negocio para campañas de retención.




📊 Resultados principales

Los Random Forest muestran mayor estabilidad y mejor desempeño general que los árboles individuales.
El incremento del número de árboles reduce la varianza de las predicciones, a costa de un mayor costo computacional.
En un contexto de churn (clases desbalanceadas), métricas como PR-AUC y F1-score resultan más informativas que la accuracy.
Los árboles de decisión destacan por su alta interpretabilidad, mientras que los ensambles priorizan capacidad predictiva y generalización.


🗂️ Estructura del repositorio
├── notebook.ipynb        # Desarrollo completo de la actividad
├── data/                # Dataset (si aplica)
├── figures/             # Gráficos generados
├── README.md            # Descripción del proyecto


🛠️ Tecnologías utilizadas

Python
scikit-learn
pandas
numpy
matplotlib
seaborn


👨‍🏫 Contexto académico
Actividad desarrollada en el marco del curso Machine Learning II.
Profesor: Francisco Pérez Galarce.
