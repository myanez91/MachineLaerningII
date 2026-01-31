# 🥛 Evaluación de la Calidad de la Leche mediante Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-red.svg)](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)

Este repositorio contiene un estudio avanzado de clasificación para categorizar la calidad de la leche en niveles **Bajo**, **Medio** y **Alto**. El proyecto aborda la necesidad de optimizar los controles de calidad en la industria láctea mediante la automatización de análisis físico-químicos.

## 👥 Autores
* **Felipe Santiago Goicolea Guerra**
* **Matias Elier Labraña Abarca**
* **Marcelo Andres Yañez Barrientos**
* *Magister Data Science (2025) - Universidad de las Américas, Santiago, Chile.*

---

## 📋 Resumen del Proyecto
El flujo de trabajo implementa la metodología **CRISP-DM**, garantizando un enfoque orientado a resultados de negocio y rigor estadístico. El objetivo principal es predecir la calidad de la leche de forma instantánea, reduciendo los costos asociados a los análisis de laboratorio tradicionales.

### 🔬 Fases de Implementación:
1.  **Análisis Exploratorio (EDA):** Identificación de outliers y análisis de correlación entre pH, temperatura y grasa.
2.  **Ingeniería de Datos:** Limpieza de duplicados, codificación de variables categóricas y escalamiento robusto de características.
3.  **Modelado:** Evaluación comparativa de múltiples algoritmos (Random Forest, SVM, KNN, entre otros).
4.  **Evaluación:** Validación cruzada y análisis detallado de métricas (Accuracy, F1-Score y Matrices de Confusión).

---

## 📊 El Dataset
El conjunto de datos comprende **1,059 registros** con variables críticas para la industria:

| Variable | Descripción |
| :--- | :--- |
| **pH** | Nivel de acidez (fundamental para detectar degradación). |
| **Temprature** | Temperatura de la muestra. |
| **Taste/Odor** | Evaluación sensorial (Binario). |
| **Fat** | Contenido graso (Binario). |
| **Turbidity** | Nivel de turbidez (Binario). |
| **Colour** | Valor cromático de la muestra. |
| **Grade** | **Target:** Calidad (Low, Medium, High). |

---

## 🚀 Resultados y Conclusiones
*(Nota: Sugerimos completar con sus métricas finales)*
El modelo basado en **[Insertar Mejor Modelo, ej. Random Forest]** alcanzó una precisión del **XX%**, demostrando que variables como el pH y la temperatura son los predictores más influyentes en la estabilidad del producto.

---

## 🛠️ Instalación y Uso

```bash

git clone https://github.com/FelipeGoico/Proyecto-Milk-Quality-ML2

https://github.com/FelipeGoico/Proyecto-Milk-Quality-ML2
