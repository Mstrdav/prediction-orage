"""
bnn_model.py - Bayesian Neural Network pour la prédiction du temps restant.

Utilise Monte Carlo Dropout comme approximation bayésienne :
- Pendant l'entraînement ET l'inférence, le dropout reste actif.
- On effectue N passages forward pour obtenir une distribution de prédictions.
- La moyenne donne la prédiction, l'écart-type donne l'incertitude.

Avantage : incertitude calibrée, crucial pour les décisions opérationnelles aéroportuaires.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"

FEATURE_COLS = [
    "time_since_start",
    "time_since_last_cg",
    "n_lightnings_15m",
    "n_lightnings_30m",
    "n_ic_15m",
    "n_ic_30m",
    "ic_ratio_15m",
    "mean_dist_15m",
    "mean_dist_30m",
    "dist_trend",
    "mean_abs_amp_15m",
    "max_abs_amp_15m",
    "dist",
    "amplitude",
    "maxis",
    "icloud",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

TARGET_COL = "time_to_end"


class BayesianMLP(nn.Module):
    """
    MLP avec MC Dropout pour l'approximation bayésienne.
    Le dropout reste actif pendant l'inférence.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.15):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Effectue n_samples passages forward avec dropout actif.
        Retourne (mean, std) des prédictions.
        """
        self.train()  # Garder le dropout actif !
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(x).cpu().numpy())
        preds = np.array(preds)  # (n_samples, n_points)
        return preds.mean(axis=0), preds.std(axis=0)


def load_and_split():
    """Charge les features et fait le split temporel."""
    df = pd.read_parquet(DATA_DIR / "features_survival.parquet")
    if "icloud" in df.columns:
        df["icloud"] = df["icloud"].astype(int)
    
    df["session_key"] = df["airport"].astype(str) + "_" + df["airport_alert_id"].astype(str)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["session_key"]))
    
    train = df.iloc[train_idx].copy()
    val = df.iloc[val_idx].copy()
    
    return train, val


def train_bnn(train, val, n_epochs=80, batch_size=512, lr=1e-3):
    """Entraîne le BNN et évalue sur le set de validation."""
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS].values)
    y_train = train[TARGET_COL].values
    X_val = scaler.transform(val[FEATURE_COLS].values)
    y_val = val[TARGET_COL].values
    
    # Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = BayesianMLP(input_dim=len(FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Loss : Huber pour robustesse aux outliers
    criterion = nn.HuberLoss(delta=20.0)
    
    print("=== Entrainement BNN (MC Dropout) ===")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        for X_batch, y_batch in loader:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f}")
    
    # ─── Prédictions avec incertitude ────────
    print("\nPrediction avec incertitude (100 passages MC Dropout)...")
    y_pred_mean, y_pred_std = model.predict_with_uncertainty(X_val_t, n_samples=100)
    
    # ─── Métriques ───────────────────────────
    errors = np.abs(y_val - y_pred_mean)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((y_val - y_pred_mean)**2))
    median_err = np.median(errors)
    p90_err = np.percentile(errors, 90)
    bias = np.mean(y_pred_mean - y_val)
    mean_uncertainty = np.mean(y_pred_std)
    
    print(f"\n{'='*50}")
    print(f"  Bayesian Neural Network (MC Dropout)")
    print(f"{'='*50}")
    print(f"  MAE :                {mae:.2f} min")
    print(f"  RMSE :               {rmse:.2f} min")
    print(f"  Mediane :            {median_err:.2f} min")
    print(f"  P90 :                {p90_err:.2f} min")
    print(f"  Biais :              {bias:+.2f} min")
    print(f"  Incertitude moyenne : {mean_uncertainty:.2f} min")
    print(f"{'='*50}")
    
    # ─── Calibration de l'incertitude ────────
    # Vérifier que ~68% des vrais sont dans [pred - std, pred + std]
    in_1std = np.mean(np.abs(y_val - y_pred_mean) <= y_pred_std)
    in_2std = np.mean(np.abs(y_val - y_pred_mean) <= 2 * y_pred_std)
    print(f"\n  Calibration de l'incertitude :")
    print(f"    % dans 1 sigma : {in_1std*100:.1f}% (ideal: 68.3%)")
    print(f"    % dans 2 sigma : {in_2std*100:.1f}% (ideal: 95.4%)")
    
    # ─── Sauvegarde ──────────────────────────
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    
    torch.save({
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_cols": FEATURE_COLS,
    }, str(model_dir / "bnn_mc_dropout.pt"))
    
    # Sauvegarder les prédictions pour les figures du rapport
    val_results = val.copy()
    val_results["pred_bnn"] = y_pred_mean
    val_results["uncertainty_bnn"] = y_pred_std
    val_results.to_parquet(DATA_DIR / "bnn_predictions.parquet", index=False)
    
    print(f"\nModele sauvegarde: {model_dir / 'bnn_mc_dropout.pt'}")
    print(f"Predictions sauvegardees: {DATA_DIR / 'bnn_predictions.parquet'}")
    
    return model, scaler, y_pred_mean, y_pred_std, y_val


if __name__ == "__main__":
    print("Loading and splitting data...")
    train, val = load_and_split()
    print(f"Train: {len(train)} | Val: {len(val)}")
    
    model, scaler, y_pred, y_std, y_true = train_bnn(train, val)
