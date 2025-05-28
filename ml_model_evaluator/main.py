from src.preprocessing import prepare_data, create_preprocessor, split_data
from src.model_factory import ModelFactory
from src.evaluator import evaluate_models
from src.explain import explain_model_shap
from src.tuning import run_optuna
from src.evaluation_utils import (
    evaluate_model, plot_roc_curves, plot_conf_matrix, find_best_threshold
)

import pandas as pd
import pickle
import os
import json
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, roc_auc_score
)

if __name__ == "__main__":
    # =================== 📥 Carga y preparación ===================
    df = prepare_data("ml_model_evaluator/data/marketing_campaign.csv")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(['Response']).tolist()
    
    target = 'Response'
    X = df.drop(columns=[target])
    y = df[target]
    
    preprocessor = create_preprocessor(numeric_cols, categorical_cols)

    # Split 60/20/20: train, val, test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"✅ Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # =================== 🔍 Baseline ===================
    factory = ModelFactory()
    baseline_models = factory.get_default_models()
    results_df = evaluate_models(baseline_models, preprocessor, X_train, y_train)
    print("\n📊 Resultados iniciales:\n", results_df)
    results_df.to_csv("ml_model_evaluator/results/model_evaluation_results.csv", index=False)

    # =================== 🔧 Optuna ===================
    selected_model = "LightGBM"
    best_params, _ = run_optuna(X_train, y_train, preprocessor, model_name=selected_model, n_trials=50)

    optimized_model = factory.build_model(selected_model, best_params)
    pipe = Pipeline([
        ('preprocessing', preprocessor),
        ('model', optimized_model)
    ])
    pipe.fit(X_train, y_train)

    # =================== 🎯 Ajuste de threshold con validación ===================
    y_val_probs = pipe.predict_proba(X_val)[:, 1]
    threshold_data = find_best_threshold(y_val, y_val_probs)
    best_threshold = threshold_data['best_threshold']

    print("\n🔍 Threshold óptimo:")
    for k, v in threshold_data.items():
        print(f"{k}: {v:.3f}")

    # =================== 🧪 Evaluación final sobre test ===================
    y_test_probs = pipe.predict_proba(X_test)[:, 1]
    y_test_preds_thresh = (y_test_probs >= best_threshold).astype(int)

    print("\n📊 Clasificación final sobre test:")
    print(classification_report(y_test, y_test_preds_thresh))
    report_df = pd.DataFrame(classification_report(y_test, y_test_preds_thresh, output_dict=True)).transpose()
    report_df.to_csv("ml_model_evaluator/results/classification_report.csv", index=True)

    # ROC y matriz de confusión ajustada
    metrics_final, fpr, tpr = evaluate_model("Optimizado Ajustado", pipe, X_test, y_test)
    plot_roc_curves([("Optimizado Ajustado", fpr, tpr, metrics_final['ROC AUC'])])
    plot_conf_matrix(y_test, y_test_preds_thresh, title="Test Confusion Matrix (umbral óptimo)")

    # =================== 📊 Comparación con modelo base ===================
    base_model = factory.get_default_models()[selected_model]
    base_pipe = Pipeline([
        ('preprocessing', preprocessor),
        ('model', base_model)
    ])
    base_pipe.fit(X_train, y_train)

    metrics_base, fpr_base, tpr_base = evaluate_model("Base", base_pipe, X_test, y_test)
    metrics_opt = {
        "Model": "Optimizado Ajustado",
        "F1": f1_score(y_test, y_test_preds_thresh),
        "ROC AUC": roc_auc_score(y_test, y_test_probs),
        "Precision": precision_score(y_test, y_test_preds_thresh),
        "Recall": recall_score(y_test, y_test_preds_thresh)
    }

    df_comparison = pd.DataFrame([metrics_base, metrics_opt])
    df_comparison.to_csv("ml_model_evaluator/results/comparison_results.csv", index=False)
    print("\n✅ Comparación guardada en: ml_model_evaluator/results/comparison_results.csv")

    # Replotear ROC
    plot_roc_curves([
        ("Base", fpr_base, tpr_base, metrics_base['ROC AUC']),
        ("Optimizado Ajustado", fpr, tpr, metrics_opt['ROC AUC'])
    ])

    # =================== 🔎 SHAP si aplica ===================
    if selected_model in ['LightGBM', 'XGBoost']:
        X_train_proc = preprocessor.transform(X_train)
        feature_names = preprocessor.get_feature_names_out()
        explain_model_shap(optimized_model, X_train_proc, feature_names=feature_names)

    # =================== 💾 Guardar modelo y parámetros ===================
    os.makedirs("ml_model_evaluator/models", exist_ok=True)

    model_path = "ml_model_evaluator/models/final_model_pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)

    params_path = "ml_model_evaluator/models/final_model_best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"\n💾 Modelo guardado en: {model_path}")
    print(f"📝 Hiperparámetros guardados en: {params_path}")
