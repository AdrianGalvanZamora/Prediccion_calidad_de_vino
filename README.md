# 🍷 Predicción de Calidad de Vino

**Sector:** Industria vitivinícola, Biotecnología, Control de calidad

**Herramientas:** Python (Pandas, NumPy, Seaborn, Scikit-learn, SciPy, Matplotlib)

---

## 📋 Descripción General

Este proyecto analiza el dataset *Wine Quality* del **UCI Machine Learning Repository** con el objetivo de **predecir la calidad de vinos tintos** a partir de sus propiedades fisicoquímicas.

Su propósito es doble:

1.  **Comprender los factores químicos que influyen en la calidad del vino.**
2.  **Desarrollar un modelo predictivo** basado en *Random Forest* que apoye la optimización de procesos de producción y clasificación en la industria vitivinícola.

El estudio combina análisis exploratorio de datos (EDA), pruebas estadísticas y modelado predictivo para ofrecer una visión cuantitativa de cómo variables como el alcohol, el pH o los sulfitos afectan la calidad percibida del vino.

---

## 📊 Dataset

-   **Fuente:** [UCI Machine Learning Repository – Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
-   **Tipo:** Vinos tintos
-   **Tamaño:** 1,599 instancias, 12 características
-   **Archivo:** `winequality-red.csv`

---

## 🔍 Metodología

1.  **Carga y Limpieza de Datos**
    -   Se verificaron valores atípicos (e.g., 155 en *residual sugar*).
    -   Se revisaron distribuciones y correlaciones entre variables.

2.  **Análisis Exploratorio (EDA)**
    -   La mayoría de los vinos tienen calidad 5 o 6 (~83 % del total).
    -   El contenido de alcohol aumenta conforme lo hace la calidad.
    -   Correlación destacada entre *fixed acidity* y *density* (~0.67).

3.  **Pruebas de Hipótesis**
    -   *t-test* confirmó diferencia significativa en alcohol según calidad (p-value ≈ 1.14 × 10⁻⁷⁷).

4.  **Preparación y Modelado**
    -   Escalado con *StandardScaler* y división 80/20 (train/test).
    -   Modelo: *Random Forest Regressor* (100 árboles, profundidad = 10).

5.  **Evaluación del Modelo**
    -   RMSE: **0.56**
    -   R²: **0.51**
    -   Principal variable predictora: **alcohol (importancia = 0.297)**.

6.  **Visualizaciones Clave**
    -   Histogramas, boxplots, matriz de correlación y gráfico de importancia de variables.

---

## 🌎 Principales Hallazgos

-   **El alcohol y los sulfitos** son los factores más influyentes en la calidad del vino.
-   El modelo explica alrededor del **51 % de la variabilidad** de la calidad.
-   Los vinos con mayor graduación alcohólica suelen ser mejor calificados.

---

## 🧠 Aplicación en el Mundo Real

Los resultados pueden aplicarse a:

-   Mejorar procesos de fermentación y control de calidad.
-   Clasificación automatizada de lotes de vino.
-   Identificación de parámetros químicos óptimos para vinos de alta calidad.

---

## ⚙️ Requisitos de Ejecución

-   Python 3.8+
-   Librerías: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`
-   Archivo: `winequality-red.csv`

---

## 🚀 Mejoras Futuras

-   Probar modelos de regresión avanzada como XGBoost o LightGBM.
-   Incluir vinos blancos para una comparación más amplia.
-   Implementar una interfaz web con Streamlit para predicciones en tiempo real.

---

## 👨‍💻 Autor

Adrián Galván Zamora

*Proyecto académico desarrollado con fines de aprendizaje y análisis de datos.*

<br>
<hr>
<br>

# 🍷 Wine Quality Prediction

**Sector:** Wine Industry, Biotechnology, Quality Control

**Tools:** Python (Pandas, NumPy, Seaborn, Scikit-learn, SciPy, Matplotlib)

---

## 📋 General Description

This project analyzes the *Wine Quality* dataset from the **UCI Machine Learning Repository** with the goal of **predicting the quality of red wines** based on their physicochemical properties.

Its purpose is twofold:

1.  **To understand the chemical factors that influence wine quality.**
2.  **To develop a predictive model** based on *Random Forest* to support the optimization of production and classification processes in the wine industry.

The study combines exploratory data analysis (EDA), statistical tests, and predictive modeling to offer a quantitative view of how variables such as alcohol, pH, or sulfites affect the perceived quality of wine.

---

## 📊 Dataset

-   **Source:** [UCI Machine Learning Repository – Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
-   **Type:** Red wines
-   **Size:** 1,599 instances, 12 features
-   **File:** `winequality-red.csv`

---

## 🔍 Methodology

1.  **Data Loading and Cleaning**
    -   Outliers were checked (e.g., 155 in *residual sugar*).
    -   Distributions and correlations between variables were reviewed.

2.  **Exploratory Data Analysis (EDA)**
    -   Most wines have a quality rating of 5 or 6 (~83% of the total).
    -   Alcohol content increases as quality increases.
    -   Notable correlation between *fixed acidity* and *density* (~0.67).

3.  **Hypothesis Testing**
    -   A *t-test* confirmed a significant difference in alcohol based on quality (p-value ≈ 1.14 × 10⁻⁷⁷).

4.  **Preparation and Modeling**
    -   Scaling with *StandardScaler* and an 80/20 train/test split.
    -   Model: *Random Forest Regressor* (100 trees, depth = 10).

5.  **Model Evaluation**
    -   RMSE: **0.56**
    -   R²: **0.51**
    -   Main predictive variable: **alcohol (importance = 0.297)**.

6.  **Key Visualizations**
    -   Histograms, boxplots, correlation matrix, and feature importance plot.

---

## 🌎 Main Findings

-   **Alcohol and sulfites** are the most influential factors in wine quality.
-   The model explains about **51% of the variability** in quality.
-   Wines with higher alcohol content tend to be rated better.

---

## 🧠 Real-World Application

The results can be applied to:

-   Improve fermentation and quality control processes.
-   Automate the classification of wine batches.
-   Identify optimal chemical parameters for high-quality wines.

---

## ⚙️ Execution Requirements

-   Python 3.8+
-   Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`
-   File: `winequality-red.csv`

---

## 🚀 Future Improvements

-   Test advanced regression models like XGBoost or LightGBM.
-   Include white wines for a broader comparison.
-   Implement a web interface with Streamlit for real-time predictions.

---

## 👨‍💻 Author

Adrián Galván Zamora

*Academic project developed for learning and data analysis purposes.*
