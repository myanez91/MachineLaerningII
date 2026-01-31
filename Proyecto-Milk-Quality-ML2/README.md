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


🥛 Pipeline IoT para una Lechera

1. Captura de datos (sensores)
- pH → sensor electroquímico conectado a un módulo IoT.\
- Temperatura → termistor/RTD digital en el tanque de leche.\
- Taste/Odor → sensores de gases volátiles (eNose) para detectar olores anómalos.\
- Fat → sensor NIR o ultrasonido para estimar contenido graso.\
- Turbidity → sensor óptico de turbidez en línea.\
- Colour → sensor espectral RGB para medir tonalidad.\
👉 Todos los sensores envían datos en tiempo real a un microcontrolador IoT (ESP32, Raspberry Pi, Arduino con WiFi).

2. Transmisión de datos
- Protocolos: MQTT o HTTP.\
- Los datos se envían hacia un broker IoT (ej. Mosquitto, Azure IoT Hub, AWS IoT Core).\
- Se almacenan en una base de datos (InfluxDB, PostgreSQL, o en la nube).

3. Procesamiento y análisis
- Preprocesamiento: limpieza de datos, normalización de valores.\
- Clasificación ML:\
- Entrenar un modelo (Random Forest, SVM, KNN) con las variables como features.\
- La salida es la Grade (Low, Medium, High).\
- Reglas de negocio: alertas si pH < 6.5 o temperatura > 10°C, etc.

4. Visualización y control
- Dashboard en Grafana/Power BI mostrando:\
- Tendencias de pH y temperatura.\
- Indicadores binarios (olor, grasa, turbidez).\
- Colorimetría en escala.\
- Resultado final: Grade de calidad.\
- Alertas automáticas vía SMS/WhatsApp/email si la leche baja de calidad.

5. Acciones en la planta
- Si la leche se clasifica como Low, se deriva a descarte o subproductos.\
- Si es Medium, se procesa con controles adicionales.\
- Si es High, se envía a producción premium.\

📊 En resumen: el pipeline de la lechera sería Sensores → Microcontrolador IoT → Transmisión → Base de datos → Modelo ML → Dashboard → Acción.


