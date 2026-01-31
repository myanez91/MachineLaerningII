# 🌲 Actividad 2: Árboles de Decisión y Ensambles para Predicción de Churn

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-II-green)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Ensemble Methods](https://img.shields.io/badge/Ensemble-Random%20Forest-orange)

Este repositorio contiene el desarrollo de la **Actividad 2** del Magíster en Data Science (UDLA). Tras analizar modelos lineales en la entrega anterior, aquí exploramos la potencia de los **modelos no lineales** y la capacidad de los ensambles para reducir la varianza y mejorar la generalización en la predicción de fuga de clientes.



## 🎯 Objetivos del Proyecto
* **Optimización Sistemática:** Comparar `GridSearchCV` vs `RandomizedSearchCV` para encontrar hiperparámetros óptimos.
* **Reducción de Varianza:** Analizar empíricamente cómo el aumento de estimadores en un **Random Forest** estabiliza las predicciones.
* **Interpretabilidad vs. Desempeño:** Evaluar el trade-off entre un árbol único (visualizable) y un ensamble (caja negra).
* **Manejo de Desbalance:** Implementar pesos por clase (`class_weight`) para mejorar el Recall en la clase minoritaria (Churn).

## 🛠️ Stack Tecnológico
* **Core:** `Scikit-Learn`, `Pandas`, `NumPy`.
* **Visualización:** `Matplotlib`, `Seaborn` y `Graphviz` para la exportación de árboles.
* **Utilidades:** Uso de `utils.py` personalizado para evaluación detallada de métricas.

## 🧠 Metodología y Hallazgos

### 1. Árbol de Decisión Único
Se optimizaron parámetros como `max_depth` y `min_samples_leaf`. Aunque el modelo es altamente interpretable, tiende a sufrir de **alta varianza** si no se poda correctamente.



### 2. Random Forest: El Poder del Ensamble
Entrenamos modelos desde 2 hasta 128 árboles para observar el comportamiento de la varianza:
* **Resultado:** A medida que aumenta el número de árboles, la varianza de las probabilidades predichas disminuye y las métricas (F1, PR-AUC) se estabilizan, demostrando el efecto del *Bagging*.

### 3. Comparación de Búsqueda de Hiperparámetros
| Método | Ventaja | Observación en este Proyecto |
| :--- | :--- | :--- |
| **Grid Search** | Exhaustivo | Garantiza el mejor resultado dentro de la grilla pero es costoso en tiempo. |
| **Random Search** | Eficiencia | Encontró resultados competitivos en una fracción del tiempo original. |

## 📊 Resultados Finales
Basado en las curvas de desempeño:
* **Mejor Modelo:** Random Forest con pesos balanceados.
* **Métrica Clave:** Se priorizó el **PR-AUC**, logrando una mejor separación de clases en comparación con la Regresión Logística de la Actividad 1.
* **Interpretación:** Variables como `Contract_Month-to-month` y `tenure` siguen siendo los predictores más fuertes, ahora capturando interacciones no lineales.

## 🏗️ Estructura del Proyecto
* `Actividad2.ipynb`: Notebook principal con el flujo de preprocesamiento, entrenamiento y análisis.
* `utils.py`: Funciones auxiliares para el cálculo de métricas de validación cruzada y graficación.
* `data-churn.csv`: Conjunto de datos de clientes de telecomunicaciones.

## 👥 Autores
* **Felipe Santiago Goicolea Guerra**
* **Matías Elier Labraña Abarca**
* **Marcelo Andrés Yáñez Barrientos**

---
**Magíster en Data Science 2025** | **Universidad de las Américas (UDLA)**
