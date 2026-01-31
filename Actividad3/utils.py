from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve
)
import numpy as np
import matplotlib.pyplot as plt
import time
scoring_main = "roc_auc"

def plot_roc_pr(y_true, y_score, title="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(f"{title} — ROC (test)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(rec, prec)
    ax.set_title(f"{title} — Precision–Recall (test)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    plt.show()
    
def timed_search(search_obj, X_train, y_train, label="search"):
    t0 = time.perf_counter()
    search_obj.fit(X_train, y_train)
    dt = time.perf_counter() - t0
    print(f"{label} best {scoring_main}: {search_obj.best_score_:.4f} | time(s)={dt:.2f}")
    return search_obj, dt


def get_scores(estimator, X):
    """
    Retorna un score continuo para ROC/PR:
    - predict_proba(:,1) si existe
    - decision_function si existe
    """
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        return s
    raise ValueError("El estimador no expone predict_proba ni decision_function.")

def report_test_metrics(model, X_test, y_test, label="model"):
    y_pred = model.predict(X_test)
    y_score = get_scores(model, X_test)

    out = {
        "model": label,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
    }
    return out, y_pred, y_score

def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    plt.show()

def plot_mean_roc_pr_from_cv(best_estimator, X, y, cv, title_prefix="Model"):
    """
    Curvas ROC/PR promedio calculadas sobre folds de CV en el set entregado.
    Útil para mostrar "curva ROC promedio" y "curva PR promedio". :contentReference[oaicite:4]{index=4}
    """
    mean_fpr = np.linspace(0, 1, 200)
    mean_rec = np.linspace(0, 1, 200)

    tprs = []
    rocs = []
    prs = []

    # Para PR curve, interpolamos precision como función de recall (convención)
    precs_interp = []

    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        est = best_estimator
        est.fit(X_tr, y_tr)

        y_score = get_scores(est, X_te)
        fpr, tpr, _ = roc_curve(y_te, y_score)
        roc_auc = roc_auc_score(y_te, y_score)

        # Interpolación ROC
        tpr_i = np.interp(mean_fpr, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)
        rocs.append(roc_auc)

        # PR
        precision, recall, _ = precision_recall_curve(y_te, y_score)
        pr_auc = average_precision_score(y_te, y_score)

        # Interpolación PR: precision(recall)
        # (recall viene decreciente en algunos casos; ordenamos)
        order = np.argsort(recall)
        recall_sorted = recall[order]
        precision_sorted = precision[order]
        prec_i = np.interp(mean_rec, recall_sorted, precision_sorted)
        precs_interp.append(prec_i)
        prs.append(pr_auc)

    # Promedios
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(rocs)

    mean_prec = np.mean(precs_interp, axis=0)
    mean_pr_auc = np.mean(prs)

    # Plot ROC
    fig, ax = plt.subplots()
    ax.plot(mean_fpr, mean_tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(f"{title_prefix} — Mean ROC (CV) | mean AUC={mean_auc:.3f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.show()

    # Plot PR
    fig, ax = plt.subplots()
    ax.plot(mean_rec, mean_prec)
    ax.set_title(f"{title_prefix} — Mean Precision–Recall (CV) | mean PR-AUC={mean_pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    plt.show()