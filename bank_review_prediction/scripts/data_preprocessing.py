import pandas as pd
import nltk
from sklearn.model_selection import train_test_split

# Descargar recursos necesarios de nltk
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_data(file_path):
    # Cargar los datos
    df = pd.read_csv(file_path)
    
    # Preprocesamiento del texto
    # ... código de preprocesamiento ...
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    return train_df, test_df

if __name__ == "__main__":
    train, test = preprocess_data('../data/bank_reviews.csv')
    train.to_csv('../data/train.csv', index=False)
    test.to_csv('../data/test.csv', index=False)
