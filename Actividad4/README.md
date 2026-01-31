# 🧠 Actividad 4: Deep Learning - MLP & CNN para Predicción de Churn

![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![Deep Learning](https://img.shields.io/badge/Architecture-MLP%20%26%20CNN-blue)
![Status](https://img.shields.io/badge/Mag%C3%ADster-Data%20Science-gold)

## 👥 Integrantes
* **Felipe Santiago Goicolea Guerra**
* **Matías Elier Labraña Abarca**
* **Marcelo Andrés Yáñez Barrientos**

Este repositorio contiene la entrega final de la asignatura **Machine Learning II**. En esta fase, superamos los modelos clásicos para implementar arquitecturas de **Redes Neuronales Artificiales**, explorando cómo el aprendizaje profundo puede extraer características complejas de datos tabulares de telecomunicaciones.



## 🎯 Objetivos de la Actividad
* **Implementación de MLP:** Diseñar un Perceptrón Multicapa (Fully Connected) para clasificación binaria.
* **Experimentación Hiperparamétrica:** Analizar el impacto crítico del *Learning Rate* y el *Batch Size* en la convergencia del modelo.
* **Innovación con CNN 1D:** Adaptar datos tabulares para ser procesados por capas convolucionales, extrayendo patrones locales entre características.
* **Benchmarking:** Realizar una comparación final entre Regresión Logística, Random Forest, SVM y Redes Neuronales.

## 🏗️ Arquitecturas Implementadas

### 1. Multilayer Perceptron (MLP)
* **Estructura:** Capas densas con funciones de activación ReLU y Dropout para mitigar el sobreajuste.
* **Optimización:** Uso de `BCELoss` (Binary Cross Entropy) y el optimizador Adam.
* **Hallazgo:** Un *Learning Rate* demasiado alto provocaba inestabilidad en la pérdida, mientras que uno muy bajo ralentizaba excesivamente la convergencia.

### 2. Red Neuronal Convolucional (CNN 1D)
* **Concepto:** Aunque los datos son tabulares, se trataron como secuencias 1D para aplicar kernels que detectan interacciones entre variables contiguas.
* **Componentes:** Capas `Conv1d`, `MaxPool1d` y capas densas finales.



## 📊 Análisis Comparativo y Conclusiones

| Característica | Modelos Clásicos (RF/SVM) | Redes Neuronales (MLP/CNN) |
| :--- | :--- | :--- |
| **Interpretabilidad** | Alta (Feature Importance) | Baja ("Caja Negra") |
| **Costo Computacional** | Bajo | Alto (Requiere entrenamiento por épocas) |
| **Desempeño** | Muy robusto en datos pequeños | Potencialmente superior con grandes volúmenes |

### Reflexiones Finales:
* **Riesgo de Sobreajuste:** Se observó que con el tamaño actual del dataset, las redes neuronales requieren una regularización agresiva (Dropout) para no memorizar el ruido de los datos de entrenamiento.
* **Escalabilidad:** Las ANN/CNN son la opción preferida cuando el volumen de datos crece masivamente o cuando se integran datos no estructurados.

## 🛠️ Stack Tecnológico
* **Deep Learning Framework:** `PyTorch`
* **Análisis de Datos:** `Pandas`, `NumPy`
* **Visualización:** `Matplotlib` (Curvas de Loss y Accuracy por época)
* **Preprocesamiento:** `Scikit-Learn`

---
**Magíster en Data Science 2025** | **Universidad de las Américas (UDLA)**
