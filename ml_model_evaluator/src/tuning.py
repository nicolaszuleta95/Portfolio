import optuna
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from src.model_factory import ModelFactory

def get_search_space(trial, model_name):
    """
    Define el espacio de búsqueda de hiperparámetros según el modelo.
    """
    if model_name == 'LightGBM':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
    elif model_name == 'Random Forest':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
        }
    elif model_name == 'SVM':
        return {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly'])
        }
    elif model_name == 'XGBoost':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0)
        }
    elif model_name == 'Logistic Regression':
        return {
            'C': trial.suggest_float('C', 0.01, 10.0, log=True)
        }
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

def objective(trial, model_name, X, y, preprocessor, cv_folds=5):
    """
    Función objetivo generalizada para Optuna.
    """
    params = get_search_space(trial, model_name)
    factory = ModelFactory()
    model = factory.build_model(model_name, params)

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', model)
    ])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, scoring='f1', cv=cv, n_jobs=-1)

    return np.mean(scores)

def run_optuna(X, y, preprocessor, model_name='LightGBM', n_trials=50):
    """
    Ejecuta la optimización con Optuna para cualquier modelo.
    """
    print(f"\n🔧 Optimizando modelo: {model_name}")
    study = optuna.create_study(direction='maximize', study_name=f"{model_name}_F1_Opt")
    study.optimize(lambda trial: objective(trial, model_name, X, y, preprocessor), n_trials=n_trials)

    print("✅ Mejor resultado:", study.best_value)
    print("🏆 Mejores hiperparámetros:")
    for key, val in study.best_params.items():
        print(f"   {key}: {val}")

    return study.best_params, study
