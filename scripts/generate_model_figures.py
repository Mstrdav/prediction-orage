"""
generate_model_figures.py – Génère les figures d'illustration pour les rapports PDF de chaque modèle.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from data_loader import load_raw, load_alerts, AIRPORT_COLORS

OUT_DIR = Path("report/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name):
    path = OUT_DIR / f"{name}.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [OK] {path}")


# ----------------------------------------------------═══════════════════════════
# Chargement des données
# ----------------------------------------------------═══════════════════════════
print("Chargement des donnees...")
feat_df = pd.read_parquet("data/features_survival.parquet")
if "icloud" in feat_df.columns:
    feat_df["icloud"] = feat_df["icloud"].astype(int)

# ----------------------------------------------------═══════════════════════════
# Fig 1 : Comparaison globale des modèles (bar chart)
# ----------------------------------------------------═══════════════════════════
print("Fig: Comparaison globale...")

models = ["Baseline\n(30 min)", "XGBoost\nAFT", "Hawkes\nClassique", "Neural\nHawkes", "BNN\nMC Dropout"]
maes = [60.26, 57.28, 83.99, 22.11, 68.20]
biases = [-52.32, -26.74, -83.82, -1.61, 1.42]
colors = ["#adb5bd", "#457B9D", "#E07A5F", "#1D3557", "#E63946"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

bars = ax1.bar(models, maes, color=colors, edgecolor="white", linewidth=1.2)
ax1.set_ylabel("MAE (minutes)", fontsize=11)
ax1.set_title("Erreur Absolue Moyenne", fontsize=13, fontweight="bold")
for bar, v in zip(bars, maes):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{v:.1f}",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_ylim(0, max(maes) * 1.15)
ax1.grid(axis="y", alpha=0.3)

bars2 = ax2.bar(models, biases, color=colors, edgecolor="white", linewidth=1.2)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("Biais (minutes)", fontsize=11)
ax2.set_title("Biais des predictions", fontsize=13, fontweight="bold")
for bar, v in zip(bars2, biases):
    ypos = v + 2 if v >= 0 else v - 4
    ax2.text(bar.get_x() + bar.get_width()/2, ypos, f"{v:+.1f}",
             ha="center", va="bottom" if v >= 0 else "top", fontsize=9, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
savefig("comparison_global")

# ----------------------------------------------------═══════════════════════════
# Fig 2 : Distribution des erreurs par modèle (violin)
# ----------------------------------------------------═══════════════════════════
print("Fig: Distribution erreurs...")

# Recalculer rapidement les erreurs de la baseline et XGBoost
from sklearn.model_selection import GroupShuffleSplit
feat_df["session_key"] = feat_df["airport"].astype(str) + "_" + feat_df["airport_alert_id"].astype(str)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, val_idx = next(gss.split(feat_df, groups=feat_df["session_key"]))
val = feat_df.iloc[val_idx].copy()
y_true = val["time_to_end"].values

# Baseline
errors_baseline = np.abs(y_true - 30.0)

# XGBoost
import xgboost as xgb
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgboost_aft.json")
FEATURE_COLS = [
    "time_since_start", "time_since_last_cg", "n_lightnings_15m", "n_lightnings_30m",
    "n_ic_15m", "n_ic_30m", "ic_ratio_15m", "mean_dist_15m", "mean_dist_30m",
    "dist_trend", "mean_abs_amp_15m", "max_abs_amp_15m", "dist", "amplitude",
    "maxis", "icloud", "hour_sin", "hour_cos", "month_sin", "month_cos",
]
dval = xgb.DMatrix(val[FEATURE_COLS].values, feature_names=FEATURE_COLS)
errors_xgb = np.abs(y_true - xgb_model.predict(dval))

# BNN
bnn_preds = pd.read_parquet("data/bnn_predictions.parquet")
errors_bnn = np.abs(bnn_preds["time_to_end"].values - bnn_preds["pred_bnn"].values)

# Neural Hawkes : approximer les erreurs via les predictions du modele
# On charge le modele pour calculer les erreurs sur le meme val set
from hawkes_models import NeuralHawkes, prepare_hawkes_sessions
raw_df = load_raw()
alerts_df = load_alerts(raw_df)
all_sessions = prepare_hawkes_sessions(alerts_df)
# Reproduire le meme split
np.random.seed(42)
n_sess = len(all_sessions)
idx_all = np.random.permutation(n_sess)
split_pt = int(n_sess * 0.8)
test_sessions = [all_sessions[i] for i in idx_all[split_pt:]]

import torch
neural_hawkes = NeuralHawkes(input_dim=4, hidden_dim=32)
neural_hawkes.model.load_state_dict(torch.load("models/neural_hawkes_gru.pt", weights_only=True))

errors_nhawkes = []
for session in test_sessions:
    times = session["times"]
    features = session["features"]
    T = session["T"]
    for frac in [0.3, 0.5, 0.7, 0.9]:
        idx_f = int(len(times) * frac)
        if idx_f < 2:
            continue
        true_rem = T - times[idx_f]
        if true_rem < 0:
            continue
        pred_rem = neural_hawkes.predict_time_remaining(features[:idx_f+1], times[:idx_f+1])
        errors_nhawkes.append(abs(true_rem - pred_rem))
errors_nhawkes = np.array(errors_nhawkes)
print(f"  Neural Hawkes errors computed: {len(errors_nhawkes)} predictions")

# Violin plot avec les 4 modeles
cap = 200
fig, ax = plt.subplots(figsize=(12, 5))
data_violin = [
    np.clip(errors_baseline, 0, cap),
    np.clip(errors_xgb, 0, cap),
    np.clip(errors_nhawkes, 0, cap),
    np.clip(errors_bnn, 0, cap),
]
labels_v = ["Baseline", "XGBoost AFT", "Neural Hawkes\n(GRU)", "BNN MC Dropout"]
colors_v = ["#adb5bd", "#457B9D", "#1D3557", "#E63946"]
parts = ax.violinplot(data_violin, showmedians=True, showextrema=False)
for pc, c in zip(parts["bodies"], colors_v):
    pc.set_facecolor(c)
    pc.set_alpha(0.7)
parts["cmedians"].set_color("black")
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(labels_v, fontsize=10)
ax.set_ylabel("Erreur absolue (minutes, cap a 200)", fontsize=11)
ax.set_title("Distribution des erreurs de prediction", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
savefig("error_distributions")

# ----------------------------------------------------═══════════════════════════
# Fig 3 : XGBoost Feature Importance
# ----------------------------------------------------═══════════════════════════
print("Fig: XGBoost Feature Importance...")

importance = xgb_model.get_score(importance_type="gain")
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:12]
feats, scores = zip(*sorted_imp)

fig, ax = plt.subplots(figsize=(8, 5))
y_pos = range(len(feats))
ax.barh(y_pos, scores, color="#457B9D", edgecolor="white")
ax.set_yticks(y_pos)
ax.set_yticklabels(feats, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Gain", fontsize=11)
ax.set_title("XGBoost survival:aft - Features les plus importantes", fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
savefig("xgboost_importance")

# ----------------------------------------------------═══════════════════════════
# Fig 4 : BNN Incertitude vs erreur
# ----------------------------------------------------═══════════════════════════
print("Fig: BNN Incertitude...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Hexbin densité : incertitude vs erreur réelle (remplacement du scatter)
uncert = bnn_preds["uncertainty_bnn"].values
err_bnn_abs = np.abs(bnn_preds["time_to_end"].values - bnn_preds["pred_bnn"].values)

hb = ax1.hexbin(uncert, err_bnn_abs, gridsize=35, cmap="YlOrRd",
                mincnt=1, bins="log", extent=(0, 80, 0, 200))
ax1.plot([0, 80], [0, 80], "k--", alpha=0.6, linewidth=1.5, label="Calibration parfaite")
ax1.set_xlabel("Incertitude predite (sigma)", fontsize=10)
ax1.set_ylabel("Erreur reelle (minutes)", fontsize=10)
ax1.set_title("Incertitude vs Erreur reelle (densite)", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
fig.colorbar(hb, ax=ax1, label="Densite (log)", shrink=0.8)

# Histogramme de l'incertitude
ax2.hist(bnn_preds["uncertainty_bnn"], bins=50, color="#E63946", alpha=0.7, edgecolor="white")
ax2.set_xlabel("Incertitude predite (sigma, minutes)", fontsize=10)
ax2.set_ylabel("Frequence", fontsize=10)
ax2.set_title("Distribution de l'incertitude", fontsize=12, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
savefig("bnn_uncertainty")

# ----------------------------------------------------═══════════════════════════
# Fig 5 : Hawkes - Intensité conditionnelle exemple
# ----------------------------------------------------═══════════════════════════
print("Fig: Hawkes intensite...")

# Charger les paramètres du Hawkes classique
hawkes_params = pd.read_json("models/hawkes_classique_params.json", typ="series")
mu, alpha, beta = hawkes_params["mu"], hawkes_params["alpha"], hawkes_params["beta"]

# Simuler une session exemple
np.random.seed(42)
cg_times = np.sort(np.random.exponential(5, 15).cumsum())
t_grid = np.linspace(0, cg_times[-1] + 30, 500)

intensities = np.zeros_like(t_grid)
for i, t in enumerate(t_grid):
    lam = mu
    for ti in cg_times:
        if ti < t:
            lam += alpha * beta * np.exp(-beta * (t - ti))
    intensities[i] = lam

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_grid, intensities, color="#1D3557", linewidth=1.5, label=r"$\lambda(t)$ (intensite)")
ax.axhline(mu + 0.02, color="#E63946", linestyle="--", alpha=0.8, label=r"Seuil de fin d'alerte ($\mu + \varepsilon$)")
ax.axhline(mu, color="#adb5bd", linestyle=":", alpha=0.8, label=r"Taux de fond $\mu$")
for ti in cg_times:
    ax.axvline(ti, color="#457B9D", alpha=0.3, linewidth=0.8)
ax.scatter(cg_times, [mu]*len(cg_times), color="#457B9D", s=30, zorder=5, label="Eclairs CG")
ax.set_xlabel("Temps (minutes)", fontsize=11)
ax.set_ylabel("Intensite conditionnelle", fontsize=11)
ax.set_title(f"Processus de Hawkes - Intensite conditionnelle (mu={mu:.3f}, alpha={alpha:.3f}, beta={beta:.3f})",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()
savefig("hawkes_intensity")

print(f"\n[DONE] Figures generees dans {OUT_DIR}/")
