import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, f1_score, precision_score, recall_score, roc_auc_score
)
import sys
sys.path.append("../")
# ===================== 📥 Cargar modelo y datos =====================

@st.cache_data
def load_model():
    with open("..//models//final_model_pipeline.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    from src.preprocessing import prepare_data_app

    df_raw = pd.read_csv("..//data//marketing_campaign.csv", sep=';')
    df = prepare_data_app(df_raw)
    return df

model = load_model()
df = load_data()

X = df.drop(columns=["Response"])
y = df["Response"]
y_probs = model.predict_proba(X)[:, 1]
y_preds = model.predict(X)

# ===================== 🖥️ Título =====================
st.title("📊 Dashboard - Modelo de Respuesta a Campaña")

# ===================== 📈 Métricas =====================
st.header("✅ Métricas Generales")

f1 = f1_score(y, y_preds)
prec = precision_score(y, y_preds)
rec = recall_score(y, y_preds)
auc_val = roc_auc_score(y, y_probs)

st.metric("F1 Score", f"{f1:.3f}")
st.metric("Precision", f"{prec:.3f}")
st.metric("Recall", f"{rec:.3f}")
st.metric("ROC AUC", f"{auc_val:.3f}")

# ===================== 📊 Curva ROC =====================
st.subheader("Curva ROC")
fpr, tpr, _ = roc_curve(y, y_probs)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
st.pyplot(plt.gcf())

# ===================== 🔍 Matriz de Confusión =====================
st.subheader("Matriz de Confusión")
cm = confusion_matrix(y, y_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
st.pyplot(plt.gcf())

# ===================== 🧠 SHAP Bar Chart =====================
st.subheader("Interpretabilidad (SHAP)")
explainer = shap.TreeExplainer(model.named_steps["model"])
X_transformed = model.named_steps["preprocessing"].transform(X)
feature_names = model.named_steps["preprocessing"].get_feature_names_out()
shap_values = explainer.shap_values(X_transformed)

plt.figure()
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, plot_type="bar", show=False)
st.pyplot(plt.gcf())

# ===================== 📂 Descargar modelo =====================
with open("..//models//final_model_pipeline.pkl", "rb") as f:
    btn = st.download_button("⬇️ Descargar modelo", f, file_name="final_model_pipeline.pkl")
