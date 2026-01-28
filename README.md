# Machine Learning Supervisado (Supervised Learning)

Este repositorio contiene implementaciones de algoritmos fundamentales de aprendizaje supervisado en Python.

## Contenido

Este proyecto implementa los siguientes algoritmos de aprendizaje supervisado:

### Algoritmos de Regresión
- **Linear Regression**: Regresión lineal con descenso de gradiente

### Algoritmos de Clasificación
- **Logistic Regression**: Regresión logística para clasificación binaria
- **Decision Tree Classifier**: Árbol de decisión para clasificación multi-clase

### Utilidades
- **train_test_split**: División de datos en conjuntos de entrenamiento y prueba
- **StandardScaler**: Normalización de características (media 0, desviación estándar 1)
- **MinMaxScaler**: Escalado de características a un rango específico
- **Métricas de evaluación**: accuracy_score, mean_squared_error, r2_score, confusion_matrix

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/myanez91/MachineLaerningII.git
cd MachineLaerningII
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Ejemplo de Regresión Lineal

```python
from supervised_learning.linear_regression import LinearRegression
from supervised_learning.utils import train_test_split, StandardScaler
import numpy as np

# Generar datos sintéticos
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.squeeze() + np.random.randn(100)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo
model = LinearRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train_scaled, y_train)

# Predecir
predictions = model.predict(X_test_scaled)
print(f"R² Score: {model.score(X_test_scaled, y_test):.4f}")
```

### Ejemplo de Regresión Logística

```python
from supervised_learning.logistic_regression import LogisticRegression
from supervised_learning.utils import train_test_split, StandardScaler
import numpy as np

# Generar datos de clasificación binaria
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Dividir y normalizar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train_scaled, y_train)

# Predecir
predictions = model.predict(X_test_scaled)
print(f"Accuracy: {model.score(X_test_scaled, y_test):.4f}")
```

### Ejemplo de Árbol de Decisión

```python
from supervised_learning.decision_tree import DecisionTreeClassifier
from supervised_learning.utils import train_test_split
import numpy as np

# Generar datos de clasificación
X = np.random.randn(150, 2)
y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar modelo
model = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
model.fit(X_train, y_train)

# Predecir
predictions = model.predict(X_test)
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

## Ejemplos Completos

El directorio `examples/` contiene scripts completos que demuestran el uso de cada algoritmo:

```bash
# Ejecutar ejemplo de regresión lineal
python examples/linear_regression_example.py

# Ejecutar ejemplo de regresión logística
python examples/logistic_regression_example.py

# Ejecutar ejemplo de árbol de decisión
python examples/decision_tree_example.py
```

## Estructura del Proyecto

```
MachineLaerningII/
│
├── supervised_learning/
│   ├── __init__.py
│   ├── linear_regression.py      # Implementación de regresión lineal
│   ├── logistic_regression.py    # Implementación de regresión logística
│   ├── decision_tree.py           # Implementación de árbol de decisión
│   └── utils.py                   # Utilidades y funciones auxiliares
│
├── examples/
│   ├── linear_regression_example.py
│   ├── logistic_regression_example.py
│   └── decision_tree_example.py
│
├── requirements.txt               # Dependencias del proyecto
├── .gitignore
└── README.md
```

## Requisitos

- Python 3.7+
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0 (opcional, para comparación)
- pandas >= 1.3.0 (opcional, para manejo de datos)
- matplotlib >= 3.4.0 (opcional, para visualización)

## Características

✅ Implementaciones desde cero (sin usar librerías de ML)
✅ Código bien documentado con docstrings
✅ Ejemplos de uso para cada algoritmo
✅ Utilidades para preprocesamiento de datos
✅ Métricas de evaluación incluidas

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para sugerencias o mejoras.

## Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## Autor

myanez91

---

**Nota**: Este es un proyecto educativo para entender los fundamentos del aprendizaje supervisado. Para aplicaciones en producción, se recomienda usar bibliotecas establecidas como scikit-learn.
