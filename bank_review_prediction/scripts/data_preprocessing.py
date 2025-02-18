import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import re
import kagglehub

# Descargar recursos necesarios de nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Diccionario de palabras positivas y negativas
positive_words = set([
    "good", "great", "excellent", "positive", "fortunate", "correct", "superior",
    "amazing", "awesome", "fantastic", "wonderful", "pleased", "satisfied", "happy",
    "delightful", "brilliant", "outstanding", "perfect", "love", "enjoy", "nice",
    "best", "incredible", "marvelous", "exceptional", "fabulous", "splendid", "terrific",
    "joyful", "cheerful", "glad", "content", "blissful", "radiant", "thrilled", "ecstatic",
    "enthusiastic", "euphoric", "grateful", "hopeful", "optimistic", "proud", "relieved",
    "trustworthy", "reliable", "dependable", "loyal", "faithful", "honest", "kind",
    "compassionate", "considerate", "generous", "thoughtful", "supportive", "encouraging",
    "friendly", "affectionate", "warm", "gentle", "caring", "loving", "sympathetic",
    "understanding", "patient", "tolerant", "forgiving", "respectful", "polite", "courteous"
])

negative_words = set([
    "bad", "terrible", "awful", "negative", "unfortunate", "wrong", "inferior",
    "horrible", "dreadful", "poor", "disappointed", "unsatisfied", "sad", "unhappy",
    "miserable", "pathetic", "substandard", "lousy", "abysmal", "deplorable", "worst",
    "disgusting", "atrocious", "appalling", "dismal", "unpleasant", "nasty", "horrendous",
    "angry", "annoyed", "frustrated", "irritated", "upset", "bitter", "resentful",
    "jealous", "envious", "hostile", "aggressive", "violent", "rude", "disrespectful",
    "selfish"
    ])

def load_data(file_path):
    # Download latest version
    path = kagglehub.dataset_download("dhavalrupapara/banks-customer-reviews-dataset")
    print("Path to dataset files:", path)
    # Cargar el dataset en un dataframe
    df = pd.read_csv(f"{path}\\bank_reviews3.csv")
    return df

def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenización
    tokens = word_tokenize(text)
    # Eliminar stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lematización
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Unir tokens en una sola cadena
    text = ' '.join(tokens)
    return text, tokens

def count_sentiment_words(tokens):
    positive_count = sum(1 for word in tokens if word in positive_words)
    negative_count = sum(1 for word in tokens if word in negative_words)
    return positive_count, negative_count

def preprocess_data(file_path):
    # Cargar los datos
    df = load_data(file_path)
    
    # Eliminar columnas innecesarias
    df = df.drop(columns=['author', 'date', 'address', 'bank_image'])
    
    # Manejo de valores nulos
    df = df.dropna()
    
    # Preprocesamiento del texto y cálculo de métricas
    processed_data = df['review'].apply(preprocess_text)
    df['review'] = processed_data.apply(lambda x: x[0])
    df['review_length'] = processed_data.apply(lambda x: len(x[1]))
    df['positive_words'] = processed_data.apply(lambda x: count_sentiment_words(x[1])[0])
    df['negative_words'] = processed_data.apply(lambda x: count_sentiment_words(x[1])[1])
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    return train_df, test_df

import os

if __name__ == "__main__":
    train, test = preprocess_data("bank_reviews3.csv")
    train.to_csv("bank_review_prediction/data/train.csv", index=False)
    test.to_csv("bank_review_prediction/data/test.csv", index=False)
    print("Data preprocessing completed.")
