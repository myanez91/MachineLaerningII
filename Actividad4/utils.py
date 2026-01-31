
import math
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
import matplotlib.pyplot as plt 

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_proba_torch(model, X_np, device=None, batch_size=1024):
    """
    Retorna probas (sigmoid(logits)) para modelos torch binarios.
    Asume salida logits de shape (N,1) o (N,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    probs = []
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(X_tensor, batch_size=batch_size, shuffle=False)
    for xb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.append(p)
    return np.concatenate(probs)

@torch.no_grad()
def predict_proba_torch_img(model, X_img, device=None, batch_size=512):
    """
    Retorna probas para modelos torch que reciben tensores (N,C,H,W).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    probs = []
    X_tensor = torch.tensor(X_img, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(X_tensor, batch_size=batch_size, shuffle=False)
    for xb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.append(p)
    return np.concatenate(probs)

def binary_classification_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }
    return out, y_pred

def plot_roc_pr(y_true, y_prob, title="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

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

def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    plt.show()

class TabularDataset(Dataset):
    def __init__(self, X_np, y_np):
        self.X = torch.tensor(X_np, dtype=torch.float32)
        self.y = torch.tensor(y_np.values if hasattr(y_np, "values") else y_np, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_torch_binary(model, train_loader, val_loader, lr=1e-3, epochs=20, verbose=True):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    t0 = time.perf_counter()

    for ep in range(1, epochs+1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())

        history["epoch"].append(ep)
        history["train_loss"].append(float(np.mean(tr_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))

        if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
            print(f"epoch {ep:02d} | train_loss={history['train_loss'][-1]:.4f} | val_loss={history['val_loss'][-1]:.4f}")

    dt = time.perf_counter() - t0
    return history, dt

@torch.no_grad()
def predict_proba_torch(model, X_np, batch_size=1024):
    model.eval()
    probs = []
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    loader = DataLoader(X_tensor, batch_size=batch_size, shuffle=False)
    for xb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.append(p)
    return np.concatenate(probs)

def run_mlp_experiment(lr, batch_size, X_tr, y_tr, X_val, y_val, X_test, y_test, X_train,
                       epochs=20, hidden_dims=(64,32), dropout=0.1):
    # loaders
    tr_loader = DataLoader(TabularDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=512, shuffle=False)

    model = MLP(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    hist, dt = train_torch_binary(model, tr_loader, va_loader, lr=lr, epochs=epochs, verbose=False)

    y_prob = predict_proba_torch(model, X_test)
    m, _ = binary_classification_metrics(y_test.values, y_prob)

    row = {"lr": lr, "batch_size": batch_size, "train_time_s": dt}
    row.update(m)
    return row, hist


def to_image_like(X_np):
    p = X_np.shape[1]
    side = int(math.ceil(math.sqrt(p)))
    new_p = side * side
    if new_p > p:
        pad = np.zeros((X_np.shape[0], new_p - p), dtype=X_np.dtype)
        X_pad = np.concatenate([X_np, pad], axis=1)
    else:
        X_pad = X_np
    X_img = X_pad.reshape(X_np.shape[0], 1, side, side)
    return X_img, side, p
