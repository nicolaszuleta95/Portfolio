import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score, f1_score,
    classification_report, precision_score, recall_score
)
import pandas as pd
from sklearn.metrics import precision_recall_curve


def evaluate_model(name, model, X_test, y_test):
    """
    Evalúa un modelo y retorna métricas + curva ROC.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Métricas clave
    metrics = {
        'Model': name,
        'F1': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_prob),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return metrics, fpr, tpr

def plot_roc_curves(results, filename="roc_curves.png"):
    """
    Grafica curvas ROC comparando varios modelos y guarda la figura.
    `results` es lista de tuplas: (label, fpr, tpr, auc_val)
    """
    plt.figure(figsize=(8, 6))
    for label, fpr, tpr, auc_val in results:
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../results/"+filename)
    plt.close()

def plot_conf_matrix(y_true, y_pred, title="Confusion Matrix", filename="conf_matrix.png"):
    """
    Grafica la matriz de confusión y guarda la figura.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig("../results/"+filename)
    plt.close()

def find_best_threshold(y_true, y_probs, metric='f1'):
    """
    Encuentra el mejor threshold que maximiza una métrica (F1 por defecto).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    best_idx = f1_scores.argmax()
    return {
        'best_threshold': thresholds[best_idx],
        'best_f1': f1_scores[best_idx],
        'precision': precisions[best_idx],
        'recall': recalls[best_idx]
    }