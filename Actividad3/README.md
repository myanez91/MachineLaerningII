# 🛡️ Actividad 3: Naïve Bayes & SVM para Predicción de Churn

![SVM](https://img.shields.io/badge/Model-SVM%20RBF-red)
![Bayes](https://img.shields.io/badge/Model-Na%C3%AFve%20Bayes-lightgrey)
![Status](https://img.shields.io/badge/Mag%C3%ADster-Data%20Science-gold)

## 👥 Integrantes
* **Felipe Santiago Goicolea Guerra**
* **Matías Elier Labraña Abarca**
* **Marcelo Andrés Yáñez Barrientos**

Este repositorio presenta la tercera fase del proyecto de predicción de fuga de clientes. Tras explorar modelos lineales y de ensamble, profundizamos en clasificadores probabilísticos (**Naïve Bayes**) y de margen máximo (**Support Vector Machines**), evaluando su capacidad para resolver fronteras de decisión complejas.



## 🎯 Objetivos de la Fase
* **Modelado Probabilístico:** Implementar Naïve Bayes considerando los supuestos de independencia de características.
* **Maximización del Margen:** Ajustar modelos SVM para encontrar el hiperplano óptimo de separación.
* **Kernel Trick:** Comparar la eficacia del Kernel Lineal frente al **RBF (Radial Basis Function)** para capturar patrones no lineales.
* **Optimización de Hiperparámetros:** Búsqueda exhaustiva y aleatoria de los parámetros $C$ (regularización) y $\gamma$ (coeficiente del kernel).

## 🧪 Metodología Técnica

### 1. Naïve Bayes
Se evaluó como modelo base probabilístico. A pesar de su "ingenuidad" respecto a la correlación de variables, ofrece una base de comparación rápida y eficiente en términos computacionales.

### 2. Support Vector Machines (SVM)
El análisis se dividió en:
* **Escalamiento Crítico:** Implementación rigurosa de `StandardScaler`, dado que SVM es extremadamente sensible a la magnitud de las variables.
* **Formulación Lineal vs. RBF:** * El modelo **lineal** permitió interpretar la importancia de las variables a través de los vectores de soporte.
    * El modelo **RBF** proyectó los datos a dimensiones superiores para resolver interacciones complejas.



## 📊 Resultados y Análisis Crítico

| Modelo | Ventaja Principal | Hallazgo Clave |
| :--- | :--- | :--- |
| **Naïve Bayes** | Velocidad / Simplicidad | Útil como baseline, pero limitado por el desbalance. |
| **SVM Lineal** | Interpretabilidad | Identifica claramente el peso de los contratos mensuales. |
| **SVM RBF** | **Alto Desempeño** | El mejor capturando la relación "Cargos vs. Tenencia". |

### Conclusiones Destacadas:
* **Impacto del Balanceo:** El uso de `class_weight='balanced'` en SVM fue el factor determinante para elevar el **Recall**, permitiendo que el modelo no ignore a los clientes que efectivamente se fugan.
* **Dimensionalidad:** Se discutió cómo el *One-Hot Encoding* aumenta la carga computacional de SVM, haciendo necesaria una selección cuidadosa de características.

## 🏗️ Estructura del Proyecto
* `Actividad3.ipynb`: Desarrollo completo (Preprocesamiento -> Tuning -> Evaluación).
* `utils.py`: Funciones de soporte para visualización de curvas **ROC** y **Precision-Recall**.
* `data-churn.csv`: Dataset original de telecomunicaciones.


**Asignatura:** Machine Learning II | **UDLA 2025**
