# Actividad 4: Redes Neuronales Artificiales y Convolucionales

Este proyecto corresponde a la **Actividad 4** de la asignatura **Machine Learning II** (Magíster en Data Science, UDLA). El objetivo principal es implementar y comparar modelos de Deep Learning (MLP y CNN) frente a modelos clásicos para la predicción de fuga de clientes (*Churn*).

## 👥 Integrantes
* FELIPE SANTIAGO GOICOLEA GUERRA
* MATÍAS ELIER LABRAÑA ABARCA
* MARCELO ANDRÉS YÁÑEZ BARRIENTOS

## 📋 Descripción del Proyecto
La actividad se centra en el análisis de un dataset de telecomunicaciones para predecir el comportamiento de abandono de los clientes. Se exploran arquitecturas de redes neuronales utilizando **PyTorch** y se comparan con modelos previos como Random Forest y SVM.

### Fases Principales:
1.  **Preprocesamiento:** Limpieza de datos, codificación de variables categóricas y normalización.
2.  **Multilayer Perceptron (MLP):** Experimentación con hiperparámetros como *Learning Rate* y *Batch Size*.
3.  **Red Neuronal Convolucional (CNN 1D):** Adaptación de datos tabulares a una estructura matricial para aplicar capas convolucionales.
4.  **Análisis Comparativo:** Evaluación de métricas (Accuracy, F1-Score, AUC-ROC) y discusión sobre costo computacional e interpretabilidad.

## 🚀 Tecnologías Utilizadas
* **Lenguaje:** Python 3.x
* **Librerías Principales:**
    * `PyTorch`: Construcción y entrenamiento de redes neuronales.
    * `Scikit-Learn`: Métricas de evaluación y preprocesamiento.
    * `Pandas` & `NumPy`: Manipulación de datos.
    * `Matplotlib`: Visualización de curvas de aprendizaje y ROC.

## 📂 Estructura de Archivos
* `Actividad4_churn_MLP_CNN_adapted.ipynb`: Notebook principal con el desarrollo, experimentos y conclusiones.
* `utils.py`: Funciones auxiliares para el entrenamiento de modelos Torch y cálculo de métricas.
* `data-churn.csv`: Conjunto de datos utilizado para el entrenamiento y test.
* `Actividad4.pdf`: Enunciado y requerimientos de la actividad.

## 🛠️ Instalación y Uso
1. Asegúrate de tener instalado Python y las dependencias necesarias:
   ```bash
   pip install torch pandas scikit-learn matplotlib numpy
