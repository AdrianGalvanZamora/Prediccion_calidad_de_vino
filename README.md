# 🍷 Predicción de Calidad de Vino  
[English version below ⬇️]  

**Sector:** Industria vitivinícola, Biotecnología, Control de calidad  
**Herramientas:** Python (Pandas, NumPy, Seaborn, Scikit-learn, SciPy, Matplotlib)  

---

## 📋 Descripción General  
Este proyecto analiza el dataset *Wine Quality* del **UCI Machine Learning Repository** con el objetivo de **predecir la calidad de vinos tintos** a partir de sus propiedades fisicoquímicas.  

Su propósito es doble:  
1. **Comprender los factores químicos que influyen en la calidad del vino.**  
2. **Desarrollar un modelo predictivo** basado en *Random Forest* que apoye la optimización de procesos de producción y clasificación en la industria vitivinícola.  

El estudio combina análisis exploratorio de datos (EDA), pruebas estadísticas y modelado predictivo para ofrecer una visión cuantitativa de cómo variables como el alcohol, el pH o los sulfitos afectan la calidad percibida del vino.

---

## 📊 Dataset  
- **Fuente:** [UCI Machine Learning Repository – Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  
- **Tipo:** Vinos tintos  
- **Tamaño:** 1,599 instancias, 12 características  
- **Archivo:** `winequality-red.csv`  

---

## 🔍 Metodología  
1. **Carga y Limpieza de Datos**  
   - Se verificaron valores atípicos (e.g., 155 en *residual sugar*).  
   - Se revisaron distribuciones y correlaciones entre variables.  

2. **Análisis Exploratorio (EDA)**  
   - La mayoría de los vinos tienen calidad 5 o 6 (~83 % del total).  
   - El contenido de alcohol aumenta conforme lo hace la calidad.  
   - Correlación destacada entre *fixed acidity* y *density* (~0.67).  

3. **Pruebas de Hipótesis**  
   - *t-test* confirmó diferencia significativa en alcohol según calidad (p-value ≈ 1.14 × 10⁻⁷⁷).  

4. **Preparación y Modelado**  
   - Escalado con *StandardScaler* y división 80/20 (train/test).  
   - Modelo: *Random Forest Regressor* (100 árboles, profundidad = 10).  

5. **Evaluación del Modelo**  
   - RMSE: **0.56**  
   - R²: **0.51**  
   - Principal variable predictora: **alcohol (importancia = 0.297)**.  

6. **Visualizaciones Clave**  
   - Histogramas, boxplots, matriz de correlación y gráfico de importancia de variables.  

---

## 🌎 Principales Hallazgos  
- **El alcohol y los sulfitos** son los factores más influyentes en la calidad del vino.  
- El modelo explica alrededor del **51 % de la variabilidad** de la calidad.  
- Los vinos con mayor graduación alcohólica suelen ser mejor calificados.  

---

## 🧠 Aplicación en el Mundo Real  
Los resultados pueden aplicarse a:  
- Mejorar procesos de fermentación y control de calidad.  
- Clasificación automatizada de lotes de vino.  
- Identificación de parámetros químicos óptimos para vinos de alta calidad.  

---

## ⚙️ Requisitos de Ejecución  
- Python 3.8+  
- Librerías: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`  
- Archivo: `winequality-red.csv`  

Instalación rápida:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# 🍷 Predicción de Calidad de Vino  
[English version below ⬇️]  

**Sector:** Industria vitivinícola, Biotecnología, Control de calidad  
**Herramientas:** Python (Pandas, NumPy, Seaborn, Scikit-learn, SciPy, Matplotlib)  

---

## 📋 Descripción General  
Este proyecto analiza el dataset *Wine Quality* del **UCI Machine Learning Repository** con el objetivo de **predecir la calidad de vinos tintos** a partir de sus propiedades fisicoquímicas.  

Su propósito es doble:  
1. **Comprender los factores químicos que influyen en la calidad del vino.**  
2. **Desarrollar un modelo predictivo** basado en *Random Forest* que apoye la optimización de procesos de producción y clasificación en la industria vitivinícola.  

El estudio combina análisis exploratorio de datos (EDA), pruebas estadísticas y modelado predictivo para ofrecer una visión cuantitativa de cómo variables como el alcohol, el pH o los sulfitos afectan la calidad percibida del vino.

---

## 📊 Dataset  
- **Fuente:** [UCI Machine Learning Repository – Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)  
- **Tipo:** Vinos tintos  
- **Tamaño:** 1,599 instancias, 12 características  
- **Archivo:** `winequality-red.csv`  

---

## 🔍 Metodología  
1. **Carga y Limpieza de Datos**  
   - Se verificaron valores atípicos (e.g., 155 en *residual sugar*).  
   - Se revisaron distribuciones y correlaciones entre variables.  

2. **Análisis Exploratorio (EDA)**  
   - La mayoría de los vinos tienen calidad 5 o 6 (~83 % del total).  
   - El contenido de alcohol aumenta conforme lo hace la calidad.  
   - Correlación destacada entre *fixed acidity* y *density* (~0.67).  

3. **Pruebas de Hipótesis**  
   - *t-test* confirmó diferencia significativa en alcohol según calidad (p-value ≈ 1.14 × 10⁻⁷⁷).  

4. **Preparación y Modelado**  
   - Escalado con *StandardScaler* y división 80/20 (train/test).  
   - Modelo: *Random Forest Regressor* (100 árboles, profundidad = 10).  

5. **Evaluación del Modelo**  
   - RMSE: **0.56**  
   - R²: **0.51**  
   - Principal variable predictora: **alcohol (importancia = 0.297)**.  

6. **Visualizaciones Clave**  
   - Histogramas, boxplots, matriz de correlación y gráfico de importancia de variables.  

---

## 🌎 Principales Hallazgos  
- **El alcohol y los sulfitos** son los factores más influyentes en la calidad del vino.  
- El modelo explica alrededor del **51 % de la variabilidad** de la calidad.  
- Los vinos con mayor graduación alcohólica suelen ser mejor calificados.  

---

## 🧠 Aplicación en el Mundo Real  
Los resultados pueden aplicarse a:  
- Mejorar procesos de fermentación y control de calidad.  
- Clasificación automatizada de lotes de vino.  
- Identificación de parámetros químicos óptimos para vinos de alta calidad.  

---

## ⚙️ Requisitos de Ejecución  
- Python 3.8+  
- Librerías: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`  
- Archivo: `winequality-red.csv`  

Instalación rápida:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
