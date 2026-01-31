Actividad 3 - Machine Learning II > Proyecto enfocado en la comparación de modelos probabilísticos y de margen máximo para la detección de fuga de clientes en telecomunicaciones.
Predicción de Churn: Naïve Bayes y SVM 📊

👥 Integrantes

Felipe Santiago Goicolea Guerra\
Matías Elier Labraña Abarca\
Marcelo Andrés Yáñez Barrientos

📝 Contexto y Objetivos

Este proyecto aborda el problema de Churn (fuga de clientes) en una empresa de telecomunicaciones. Tras explorar modelos lineales y basados en árboles en etapas anteriores, esta fase se centra en:\
Naïve Bayes: Implementación de modelos probabilísticos basados en el Teorema de Bayes.\
Support Vector Machines (SVM): Optimización de hiperparámetros y uso de kernels (RBF y Lineal) para maximizar el margen de separación.\
Manejo de Desbalance: Estrategias de balanceo de clases para mejorar el Recall en la clase positiva.

🛠️ Tecnologías y Herramientas

Lenguaje: Python\
Bibliotecas Principales: pandas, numpy, scikit-learn, matplotlib, seaborn.\
Entorno: Jupyter Notebook.

Scripts de apoyo: utils.py (contiene funciones personalizadas para evaluación de métricas y visualización de curvas ROC/PR).

📂 Estructura del Repositorio

Actividad3.ipynb: Notebook principal con el flujo de preprocesamiento, entrenamiento y análisis.

data-churn.csv: Conjunto de datos de telecomunicaciones.\
utils.py: Módulo con funciones auxiliares para validación cruzada y graficación.\

Actividad 3.pdf: Enunciado y requerimientos del proyecto.

🚀 Conclusiones Destacadas
Del análisis realizado, se determinó que:

Escalamiento: El uso de StandardScaler resultó crítico para el desempeño de SVM.\
Kernel RBF: Logró capturar relaciones no lineales (como interacción entre cargos mensuales y tipo de contrato) que los modelos lineales pasaron por alto.\
Balanceo: La incorporación de class_weight='balanced' mejoró significativamente el Recall, permitiendo identificar mejor a los clientes en riesgo de fuga.
