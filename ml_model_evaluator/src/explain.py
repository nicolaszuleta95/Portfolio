import shap
import matplotlib.pyplot as plt

def explain_model_shap(model, X, feature_names=None, max_display=15):
    """
    Calcula e imprime interpretabilidad del modelo usando SHAP.
    
    Parámetros:
    - model: modelo ya entrenado (LightGBM, XGBoost, etc.)
    - X: datos ya preprocesados (np.array o DataFrame)
    - feature_names: nombres de las features si X es array
    - max_display: cuántas features mostrar en los gráficos
    
    Muestra:
    - Summary plot (bar y beeswarm)
    """
    print("✅ Calculando valores SHAP...")
    
    # Crear el explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Nombre de features si es array
    if feature_names is not None:
        shap_values.feature_names = feature_names

    # Gráfico tipo summary (beeswarm)
    print("📊 SHAP Summary (Beeswarm):")
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    #guardar gráficos
    plt.tight_layout()  # Ajustar layout para evitar superposición
    plt.savefig("ml_model_evaluator/results/shap_summary_plot.png", dpi=700)  # .png, .pdf también son soportados
    plt.clf()  # Limpiar figura para evitar superposición de gráficos

    # Gráfico tipo bar (media de importancia absoluta)
    print("📊 SHAP Feature Importance (Bar):")
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    #guardar gráficos
    plt.tight_layout()  # Ajustar layout para evitar superposición
    plt.savefig("ml_model_evaluator/results/shap_bar_plot.png", dpi=700)  # .png, .pdf también son soportados
    print("✅ Gráficos SHAP guardados en resultados.")

