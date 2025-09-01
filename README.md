# Predicción de Calidad de Vino

## Descripción del Proyecto
Este proyecto utiliza el dataset Wine Quality del UCI Machine Learning Repository para predecir la calidad de vinos tintos basándose en propiedades fisicoquímicas. Es relevante para la industria vitivinícola, biotecnología y control de calidad, con aplicaciones en optimización de producción y clasificación de vinos.

- **Dataset:** Wine Quality (1599 instancias, 12 features).
- **Fuente:** [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality)
- **Herramientas:** Python con Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn y SciPy.
- **Objetivos:**
  - Realizar análisis exploratorio de datos (EDA) para identificar patrones.
  - Limpiar datos (manejo de valores atípicos si aplica).
  - Pruebas de hipótesis (e.g., diferencia en alcohol por calidad).
  - Modelado de regresión con Random Forest.
  - Evaluación con RMSE (0.56) y R² (0.51).
  - Visualizaciones: histogramas, boxplots, importancia de variables.

## Requisitos
- Python 3.8+.
- Bibliotecas: Instala con `pip install pandas numpy matplotlib seaborn scikit-learn scipy`.
- Dataset: Descarga `winequality-red.csv` de [aquí](https://archive.ics.uci.edu/dataset/186/wine+quality) y coloca en la carpeta del notebook.

## Metodología
1. **Carga y Limpieza:** Dataset cargado con 1599 instancias, manejo de valores atípicos (e.g., 155 en residual sugar) opcional.
2. **EDA:**
   - Distribución del target: ~43% calidad 5, ~40% calidad 6.
   - Histogramas: Variables como alcohol muestran sesgos.
   - Boxplots: Alcohol más alto en vinos de mayor calidad.
   - Correlaciones: Moderadas (e.g., fixed acidity y density ~0.67).
3. **Pruebas de Hipótesis:** t-test confirma diferencia significativa en alcohol (p-value: 1.14e-77).
4. **Preparación:** Escalado con StandardScaler, split 80/20 (X_train: 1279; X_test: 320).
5. **Modelado:** Random Forest (100 árboles, profundidad 10). RMSE: 0.56, R²: 0.51.
6. **Evaluación:** Distribución de errores, importancia de variables (alcohol como predictor principal).
7. **Visualizaciones:** Matriz de confusión, gráfico de importancia de features.

## Resultados Clave
- **RMSE:** 0.56.
- **R²:** 0.51 (explica ~51% de la variabilidad).
- **Mejor predictor:** Alcohol (importancia: 0.297).
- **Insights:** Alcohol y sulphates son clave para calidad, útil para optimización de procesos.
- **Limitaciones:** Modelo no optimizado; calidad tiene rango estrecho (3-8).

## Cómo Ejecutar
1. Descarga `winequality-red.csv` y coloca en la carpeta.
2. Abre `Proyecto3_PrediccionCalidadVino.ipynb` en Jupyter Notebook.
3. Ejecuta las celdas en orden.
4. Nota: El entrenamiento toma ~1 minuto.

## Mejoras Futuras
- Optimizar hiperparámetros (e.g., GridSearchCV).
- Combinar vinos tintos y blancos para mayor generalización.
- Desarrollar una interfaz web para predicciones en tiempo real.

## Licencia
MIT License. Cita el dataset original si usas este proyecto.

Autor: [Tu Nombre/Usuario de LinkedIn]  
Fecha: 1 de septiembre de 2025  
Enlace al repositorio: [Inserta enlace de GitHub aquí]
