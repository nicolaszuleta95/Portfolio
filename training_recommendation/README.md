
# Proyecto de Ciencia de Datos: Recomendador de Plan de Entrenamiento

## Objetivo
Desarrollar un modelo de Machine Learning que sugiera un plan de entrenamiento personalizado en función de las características del usuario y su historial de ejercicios.

## Plan de Trabajo

### Fase 1: Exploración y Preprocesamiento de Datos
- Cargar y analizar los datos.
- Manejo de valores nulos y outliers.
- Creación de nuevas características (IMC, intensidad del ejercicio, etc.).
- Normalización y transformación de variables.

### Fase 2: Modelado
- Clustering de Usuarios (K-Means) para identificar perfiles.
- Clasificación (Random Forest, XGBoost) para predecir el mejor entrenamiento.
- Recomendador de entrenamientos (Filtrado Colaborativo) basado en hábitos previos.

### Fase 3: Evaluación y Visualización
- Análisis de métricas de desempeño.
- Visualización de grupos de usuarios y relaciones clave.

### Fase 4: Implementación y Despliegue
- Creación de una aplicación en Streamlit para probar recomendaciones.
- Publicación del código en GitHub.
