import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

def get_scoring():
    """
    Métricas para evaluación. ROC AUC como string estándar para evitar conflictos con make_scorer.
    """
    return {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': 'roc_auc'
    }

def evaluate_models(models, preprocessor, X, y, cv_folds=5):
    """
    Evalúa múltiples modelos usando validación cruzada con métricas múltiples.
    
    Parámetros:
    - models: dict con nombre y estimador scikit-learn
    - preprocessor: objeto ColumnTransformer ya construido
    - X: características
    - y: variable objetivo
    - cv_folds: número de folds en CV (StratifiedKFold)
    
    Retorna:
    - DataFrame con resultados promedio y desviaciones estándar
    """
    scoring = get_scoring()
    results = []

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in models.items():
        # Crear pipeline completo
        pipe = Pipeline([
            ('preprocessing', preprocessor),
            ('model', model)
        ])

        # Evaluación cruzada
        cv_results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=False)

        # Promedio y desviación de métricas
        result = {
            'Model': name,
            'Accuracy': np.mean(cv_results['test_accuracy']),
            'Precision': np.mean(cv_results['test_precision']),
            'Recall': np.mean(cv_results['test_recall']),
            'F1': np.mean(cv_results['test_f1']),
            'ROC AUC': np.mean(cv_results['test_roc_auc']),
            'Std Accuracy': np.std(cv_results['test_accuracy']),
            'Std F1': np.std(cv_results['test_f1'])
        }
        results.append(result)

    return pd.DataFrame(results).sort_values(by='F1', ascending=False).reset_index(drop=True)
