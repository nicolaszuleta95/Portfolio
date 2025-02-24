# Script para el preprocesamiento de datos

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
file_path = "training_recommendation/data/raw/workout_fitness_tracker_data.csv"
df = pd.read_csv(file_path)

def preprocess_data(df):

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Índice de Masa Corporal (IMC o BMI)
    df['BMI'] = df['Weight (kg)'] / (df['Height (cm)'] / 100) ** 2

    # Ratio de Frecuencia Cardíaca
    df['Heart Rate Ratio'] = df['Heart Rate (bpm)'] / df['Resting Heart Rate (bpm)']

    # Diferencia de Frecuencia Cardíaca
    df['HR Change'] = df['Heart Rate (bpm)'] - df['Resting Heart Rate (bpm)']

    # Calorías Quemadas por Minuto
    df['Calories Per Minute'] = df['Calories Burned'] / df['Workout Duration (mins)']

    # Pasos por Minuto
    df['Steps Per Minute'] = df['Steps Taken'] / df['Workout Duration (mins)']


    # Déficit o Exceso de Calorías
    df['Calorie Balance'] = df['Daily Calories Intake'] - df['Calories Burned']

    # Impacto del Sueño en el Rendimiento
    df['Sleep to Performance'] = df['Calories Burned'] / df['Sleep Hours']

    # Clasificación de Intensidad de Entrenamiento
    def categorize_intensity(row):
        if row['Calories Burned'] < 200 and row['Workout Duration (mins)'] < 30:
            return 'Baja Intensidad'
        elif row['Calories Burned'] < 500 and row['Workout Duration (mins)'] < 60:
            return 'Media Intensidad'
        else:
            return 'Alta Intensidad'

    df['Workout Intensity Category'] = df.apply(categorize_intensity, axis=1)

    # Impacto del Entrenamiento en el Estado de Ánimo
    df['Mood Impact'] = df['Mood After Workout'] + " - " + df['Mood Before Workout']

    # Selección de variables numéricas para normalización
    num_cols = ['Age', 'Height (cm)', 'Weight (kg)', 'Workout Duration (mins)', 
                'Calories Burned', 'Heart Rate (bpm)', 'Steps Taken', 'VO2 Max', 'Body Fat (%)', 'BMI']

    # Normalización de variables numéricas
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Guardar dataset preprocesado
    df.to_csv("training_recommendation/data/processed/workout_fitness_tracker_preprocessed.csv", index=False)

    print("Preprocesamiento completado. Dataset guardado como 'workout_fitness_tracker_preprocessed.csv'")

if __name__ == "__main__":
    preprocess_data(df)