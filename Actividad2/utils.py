from sklearn.compose import ColumnTransformer

import time
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,  
                                roc_curve, precision_recall_curve, auc,
                             confusion_matrix, classification_report)


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> list:
    """Obtiene nombres de features luego del preprocesamiento (numéricas + one-hot)."""
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        # Fallback para versiones antiguas de sklearn
        feature_names = []
        # num
        feature_names.extend([f"num__{c}" for c in numeric_features])
        # cat (sin nombres de categorías)
        feature_names.extend([f"cat__{c}" for c in categorical_features])
        return feature_names


def evaluate_model_cv_detailed(model, X, y, cv, positive_label=1, plot_curves=True, verbose=True):
    """
    Evalúa un modelo con un objeto CV (p.ej., StratifiedKFold).
    Retorna:
      - df_metrics: métricas por fold
      - summary: métricas promedio y desviación estándar
      - curves: diccionario con ROC/PR agregadas (global)
    """
    rows = []
    y_true_all, y_proba_all, y_pred_all = [], [], []

    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y.iloc[tr], y.iloc[te]

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        pred_time = time.perf_counter() - t1

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)

        rows.append({
            "fold": fold,
            "fit_time_s": fit_time,
            "pred_time_s": pred_time,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc_roc": roc_auc,
            "pr_auc": pr_auc
        })

        y_true_all.extend(y_test.tolist())
        y_proba_all.extend(proba.tolist())
        y_pred_all.extend(pred.tolist())

    df_metrics = pd.DataFrame(rows)

    summary = (df_metrics.drop(columns=["fold"])
               .agg(["mean", "std"])
               .T
               .rename(columns={"mean": "mean", "std": "std"}))

    # Curvas globales (agregadas)
    fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
    roc_auc_global = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true_all, y_proba_all)
    pr_auc_global = average_precision_score(y_true_all, y_proba_all)

    curves = {
        "fpr": fpr, "tpr": tpr, "roc_auc_global": roc_auc_global,
        "precision": precision, "recall": recall, "pr_auc_global": pr_auc_global,
        "y_true_all": np.array(y_true_all), "y_proba_all": np.array(y_proba_all), "y_pred_all": np.array(y_pred_all)
    }

    if verbose:
        display(df_metrics)
        display(summary)

        print("\nMatriz de confusión (global):")
        print(confusion_matrix(curves["y_true_all"], curves["y_pred_all"]))
        print("\nClassification report (global):")
        print(classification_report(curves["y_true_all"], curves["y_pred_all"], digits=3))

    if plot_curves:
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_global:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Curva ROC (predicciones agregadas CV)")
        plt.legend()
        plt.show()

        plt.figure()
        plt.step(recall, precision, where="post", label=f"PR (AP = {pr_auc_global:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Curva Precision–Recall (predicciones agregadas CV)")
        plt.legend()
        plt.show()

    return df_metrics, summary, curves