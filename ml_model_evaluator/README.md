
# 🧠 ML Evaluator Dashboard | Tablero de Evaluación de Modelos de ML

## 🎯 Project Objective | Objetivo del proyecto

Construir una herramienta interactiva que permita comparar y evaluar múltiples modelos de Machine Learning con métricas avanzadas y visualizaciones explicativas.

Build an interactive tool to compare and evaluate multiple Machine Learning models using advanced metrics and explanatory visualizations.

---

## 🚀 Features | Funcionalidades

✅ Comparación de modelos (Random Forest, XGBoost, SVM, etc.)  
✅ Evaluación con validación cruzada y métricas como Accuracy, F1, AUC y Log-loss  
✅ Visualización de resultados en dashboard interactivo con Streamlit  
✅ Análisis de importancia de variables y explicabilidad con SHAP  

✅ Model comparison (Random Forest, XGBoost, SVM, etc.)  
✅ Evaluation using cross-validation and metrics like Accuracy, F1, AUC, and Log-loss  
✅ Results visualization via Streamlit interactive dashboard  
✅ Feature importance analysis and explainability with SHAP

---

## 🧱 Stack Tecnológico | Tech Stack

- Python 3.9+
- Streamlit
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib / Seaborn
- Pandas / NumPy

---

## 🖼 Demo

Puedes ver la demo aquí: [ML Evaluator App](https://mlmodelevaluatornz.streamlit.app)

You can try the demo here: [ML Evaluator App](https://mlmodelevaluatornz.streamlit.app)

---

## 🏁 Cómo ejecutar localmente | How to Run Locally

```bash
# Clonar el repositorio / Clone repository
git clone https://github.com/tu_usuario/portfolio.git
cd portfolio/ml_model_evaluator

# Crear entorno virtual / Create virtual environment
python -m venv .venv
source .venv/bin/activate  # o `.\.venv\Scriptsctivate` en Windows

# Instalar dependencias / Install dependencies
pip install -r requirements.txt

# Ejecutar la aplicación / Run the app
streamlit run src/app.py
```

---

## 🧪 Estructura del Proyecto | Project Structure

```
ml_model_evaluator/
│
├── data/
│   └── marketing_campaign.csv
│
├── models/
│   ├── final_model_pipeline.pkl
│   └── final_model_best_params.json
│
├── notebooks/
│   ├── eda_marketing_campaign.ipynb
│   └── model_report.ipynb
│
├── results/
│   ├── classification_report.csv
│   ├── comparison_results.csv
│   ├── model_evaluation_results.csv
│   ├── shap_bar_plot.png
│   ├── shap_summary_plot.png
│   ├── roc_curves.png
│   └── conf_matrix.png
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── evaluation_utils.py
│   ├── evaluator.py
│   ├── explain.py
│   ├── model_factory.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── tuning.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 📊 Resultados esperados | Expected Outputs

- Gráficos comparativos de métricas entre modelos base y optimizados
- Explicaciones SHAP para interpretabilidad
- Rankings de importancia de variables
- Matrices de confusión y curvas ROC/AUC

---

## 📌 Notas finales | Final Notes

Este proyecto fue creado como parte de un portafolio de ciencia de datos.

This project is part of a Data Science portfolio.
