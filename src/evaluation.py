"""
evaluation.py – Split temporel, baseline 30 min, XGBoost Survival, métriques comparatives.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

DATA_DIR = Path(__file__).parent.parent / "data"

# ─── Features utilisées par XGBoost ──────────────────────────────────────────
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


def load_features() -> pd.DataFrame:
    """Charge le dataset de features pré-calculé."""
    path = DATA_DIR / "features_survival.parquet"
    df = pd.read_parquet(path)
    # Convertir les booléens en int pour XGBoost
    if "icloud" in df.columns:
        df["icloud"] = df["icloud"].astype(int)
    return df


def temporal_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split temporel par session d'alerte.
    On utilise GroupShuffleSplit avec les sessions comme groupes
    pour éviter tout leakage intra-session.
    """
    # Identifier chaque session de façon unique
    df["session_key"] = df["airport"].astype(str) + "_" + df["airport_alert_id"].astype(str)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["session_key"]))
    
    train = df.iloc[train_idx].copy()
    val = df.iloc[val_idx].copy()
    
    print(f"Train: {len(train)} lignes ({train['session_key'].nunique()} sessions)")
    print(f"Val:   {len(val)} lignes ({val['session_key'].nunique()} sessions)")
    
    return train, val


def baseline_30min(val: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline : prédit que le temps restant est toujours 30 minutes.
    C'est la règle métier actuelle.
    """
    val = val.copy()
    val["pred_baseline"] = 30.0
    return val


def train_xgboost_survival(train: pd.DataFrame, val: pd.DataFrame):
    """
    Entraîne un XGBoost avec l'objectif survival:aft.
    AFT = Accelerated Failure Time, modélise directement ln(T).
    """
    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_COL].values
    X_val = val[FEATURE_COLS].values
    y_val = val[TARGET_COL].values
    
    # Borne inférieure et supérieure pour AFT
    # Comme nos données ne sont PAS censurées (on connaît le vrai temps de fin),
    # on met y_lower = y_upper = y + epsilon (epsilon pour éviter log(0))
    y_train_safe = np.maximum(y_train, 0.01)  # Éviter log(0)
    y_val_safe = np.maximum(y_val, 0.01)
    
    dtrain = xgb.DMatrix(X_train, label=y_train_safe, feature_names=FEATURE_COLS)
    dval = xgb.DMatrix(X_val, label=y_val_safe, feature_names=FEATURE_COLS)
    
    # AFT nécessite les bornes de l'intervalle de censure
    dtrain.set_float_info("label_lower_bound", y_train_safe)
    dtrain.set_float_info("label_upper_bound", y_train_safe)
    dval.set_float_info("label_lower_bound", y_val_safe)
    dval.set_float_info("label_upper_bound", y_val_safe)
    
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.2,
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    
    print("\n=== Entrainement XGBoost survival:aft ===")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=50,
        early_stopping_rounds=30,
    )
    
    # Prédictions
    val = val.copy()
    val["pred_xgb_aft"] = model.predict(dval)
    
    return model, val


def compute_metrics(val: pd.DataFrame, pred_col: str, label: str):
    """Calcule et affiche les métriques de performance."""
    y_true = val[TARGET_COL].values
    y_pred = val[pred_col].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Métriques métier : erreur médiane en minutes
    errors = np.abs(y_true - y_pred)
    median_error = np.median(errors)
    p90_error = np.percentile(errors, 90)
    
    # Biais : est-ce qu'on sous-estime ou surestime ?
    mean_bias = np.mean(y_pred - y_true)
    
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  MAE (erreur moyenne) :     {mae:.2f} min")
    print(f"  RMSE :                     {rmse:.2f} min")
    print(f"  Erreur mediane :           {median_error:.2f} min")
    print(f"  Erreur P90 :               {p90_error:.2f} min")
    print(f"  Biais moyen :              {mean_bias:+.2f} min")
    print(f"{'='*50}")
    
    return {"label": label, "mae": mae, "rmse": rmse, "median_error": median_error,
            "p90_error": p90_error, "bias": mean_bias}


if __name__ == "__main__":
    print("Loading features...")
    df = load_features()
    
    print("\n--- Split temporel ---")
    train, val = temporal_split(df)
    
    # ─── Baseline ──────────────────────────
    val = baseline_30min(val)
    baseline_metrics = compute_metrics(val, "pred_baseline", "Baseline (30 min fixe)")
    
    # ─── XGBoost AFT ──────────────────────
    xgb_model, val = train_xgboost_survival(train, val)
    xgb_metrics = compute_metrics(val, "pred_xgb_aft", "XGBoost survival:aft")
    
    # ─── Feature importance ───────────────
    print("\n--- Feature Importance (top 15) ---")
    importance = xgb_model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    for feat, score in sorted_imp:
        print(f"  {feat:30s} {score:.1f}")
    
    # ─── Sauvegarde du modèle ─────────────
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    xgb_model.save_model(str(model_dir / "xgboost_aft.json"))
    print(f"\nModele sauvegarde: {model_dir / 'xgboost_aft.json'}")
    
    # ─── Résumé comparatif ────────────────
    print("\n" + "="*60)
    print("  COMPARAISON DES MODELES")
    print("="*60)
    for m in [baseline_metrics, xgb_metrics]:
        print(f"  {m['label']:35s} | MAE={m['mae']:.2f} | RMSE={m['rmse']:.2f} | Median={m['median_error']:.2f} | Biais={m['bias']:+.2f}")
    print("="*60)
