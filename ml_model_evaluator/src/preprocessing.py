# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def prepare_data(path):
    """
    Realiza el procesamiento básico del dataframe:
    - Crea nuevas variables
    - Filtra datos erróneos
    - Elimina columnas irrelevantes
    - Devuelve X, y, columnas categóricas y numéricas
    """
    df = pd.read_csv(path, sep=';')
    # Eliminar columnas innecesarias
    df = df.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue'])
    # Eliminar registros con valores nulos
    df = df.dropna()
    # Convertir fecha a datetime si existe
    if 'Dt_Customer' in df.columns:
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
    
    df = df.copy()
    print(df.head())  # Verificar que los datos se cargaron correctamente

    # Crear variables derivadas
    df['Customer_Seniority'] = (pd.to_datetime('today') - df['Dt_Customer']).dt.days
    df['TotalMntSpent'] = df[['MntWines','MntFruits','MntMeatProducts',
                              'MntFishProducts','MntSweetProducts','MntGoldProds']].sum(axis=1)
    df['Customer_Age'] = 2025 - df['Year_Birth']
    
    # Filtrar edades no válidas
    df = df[(df['Customer_Age'] >= 18) & (df['Customer_Age'] <= 100)]
    
    # Eliminar columnas no útiles
    drop_cols = ['Year_Birth', 'Dt_Customer']
    df.drop(columns=drop_cols, inplace=True)
    
    return df

def prepare_data_app(df):
    # No intentes leer desde archivo aquí
    # Supón que df ya es un DataFrame listo para procesar
    # Resto de tu lógica aquí...
    df['Customer_Age'] = 2025 - df['Year_Birth']
    df['Customer_Seniority'] = pd.to_datetime("2025-01-01") - pd.to_datetime(df['Dt_Customer'])
    df['Customer_Seniority'] = df['Customer_Seniority'].dt.days
    df['TotalMntSpent'] = (
        df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] +
        df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
    )
    return df


def create_preprocessor(numeric_cols, categorical_cols):
    """
    Crea un ColumnTransformer con escalado para numéricas
    y codificación one-hot para categóricas.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

# def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
#     """
#     Divide los datos en train y test de forma estratificada.
#     """
#     if stratify:
#         return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
#     else:
#         return train_test_split(X, y, test_size=test_size, random_state=random_state)

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    # División inicial: train vs (val + test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # División secundaria: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test