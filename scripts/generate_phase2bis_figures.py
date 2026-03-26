"""
generate_phase2bis_figures.py – Figures pour le rapport Phase 2bis.
Thème identique aux figures Phase 2 (couleurs Météorage, styles).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Thème Météorage (identique à generate_model_figures.py) ──
COLORS = {
    "meteoBlue": "#1D3557",
    "meteoCyan": "#457B9D",
    "meteoSky":  "#A8DADC",
    "meteoRed":  "#E63946",
    "meteoOrange": "#E07A5F",
    "gray":      "#adb5bd",
}

OUT_DIR = Path(__file__).parent.parent / "report" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent.parent / "data"


def savefig(name):
    path = OUT_DIR / f"{name}.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [OK] {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 : Comparaison ablative (bar chart)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_ablation():
    variants = [
        ("Baseline\n(30 min)",      60.26, -52.32, COLORS["gray"]),
        ("XGBoost\nAFT",            57.28, -26.74, COLORS["meteoCyan"]),
        ("GRU v1\n(Phase 2)",       22.11,  -1.61, COLORS["meteoSky"]),
        ("GRU v2\nmulti-tâche",     26.59, +11.04, COLORS["meteoOrange"]),
        ("Transformer\nmulti-tâche", 29.17, +14.10, COLORS["meteoRed"]),
        ("GRU\nGaussian",           21.33,  +1.96, COLORS["meteoBlue"]),
        ("Transformer\nGaussian",   23.25,  +4.23, COLORS["meteoCyan"]),
        ("Ensemble\npar aéroport",  21.93,  +0.76, COLORS["meteoSky"]),
    ]
    
    names = [v[0] for v in variants]
    maes = [v[1] for v in variants]
    biases = [v[2] for v in variants]
    colors = [v[3] for v in variants]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE
    bars = ax1.bar(names, maes, color=colors, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, maes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("MAE (minutes)", fontsize=11)
    ax1.set_title("Erreur Absolue Moyenne", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, max(maes) * 1.15)
    ax1.axhline(y=22.11, color=COLORS["meteoSky"], linestyle="--", alpha=0.5, label="GRU v1 (réf.)")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(axis='x', labelsize=8)
    
    # Biais
    bars2 = ax2.bar(names, biases, color=colors, edgecolor="white", linewidth=1.2)
    ax2.axhline(0, color="black", linewidth=0.8)
    for bar, v in zip(bars2, biases):
        ypos = v + 2 if v >= 0 else v - 4
        ax2.text(bar.get_x() + bar.get_width()/2, ypos, f"{v:+.1f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Biais (minutes)", fontsize=11)
    ax2.set_title("Biais des prédictions", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(axis='x', labelsize=8)
    
    plt.tight_layout()
    savefig("phase2bis_ablation")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 : Performance par aéroport
# ═══════════════════════════════════════════════════════════════════════════════

def fig_per_airport():
    airports = ["Ajaccio", "Bastia", "Biarritz", "Nantes", "Pise"]
    gru_v3 = [21.47, 21.77, 22.74, 21.00, 20.17]
    per_ap = [19.62, 21.14, 23.42, 20.92, 22.85]
    
    x = np.arange(len(airports))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, gru_v3, width, label="GRU Gaussian (global)",
                   color=COLORS["meteoBlue"], edgecolor="white", linewidth=1.2)
    bars2 = ax.bar(x + width/2, per_ap, width, label="Modèle local (par aéroport)",
                   color=COLORS["meteoCyan"], edgecolor="white", linewidth=1.2)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                   f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    ax.set_ylabel("MAE (minutes)", fontsize=11)
    ax.set_title("Performance par aéroport — Modèle global vs local", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(airports, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 30)
    
    plt.tight_layout()
    savefig("phase2bis_per_airport")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 : Distribution des erreurs (violin)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_error_distribution():
    fig, ax = plt.subplots(figsize=(12, 5))
    
    data_files = {
        "GRU\nGaussian": "errors_v3_gru_gaussian.parquet",
        "Transformer\nGaussian": "errors_v3_transformer_gaussian.parquet",
        "TF Large\nGaussian": "errors_v3_transformer_large_gaussian.parquet",
    }
    
    all_errors = []
    labels = []
    
    for label, fname in data_files.items():
        path = DATA_DIR / fname
        if path.exists():
            df = pd.read_parquet(path)
            errs = (df["pred"] - df["true"]).values
            all_errors.append(np.clip(errs, -100, 100))
            labels.append(label)
    
    if not all_errors:
        print("  ⚠ Pas de fichiers d'erreurs, figure ignorée")
        plt.close(fig)
        return
    
    colors_v = [COLORS["meteoBlue"], COLORS["meteoCyan"], COLORS["meteoOrange"]]
    parts = ax.violinplot(all_errors, showmedians=True, showextrema=False)
    for pc, c in zip(parts["bodies"], colors_v):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Erreur de prédiction (minutes)", fontsize=11)
    ax.set_title("Distribution des erreurs — Variantes Gaussian", fontsize=13, fontweight="bold")
    ax.axhline(y=0, color=COLORS["meteoRed"], linestyle="--", alpha=0.5, label="Prédiction parfaite")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    savefig("phase2bis_error_dist")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 : Scatter pred vs true
# ═══════════════════════════════════════════════════════════════════════════════

def fig_scatter():
    path = DATA_DIR / "errors_v3_gru_gaussian.parquet"
    if not path.exists():
        print("  ⚠ Fichier non trouvé, scatter ignoré")
        return
    
    df = pd.read_parquet(path)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    airport_colors = {
        "Ajaccio": COLORS["meteoRed"],
        "Bastia": COLORS["meteoOrange"],
        "Biarritz": COLORS["meteoCyan"],
        "Nantes": COLORS["meteoSky"],
        "Pise": COLORS["meteoBlue"],
    }
    
    for ap, color in airport_colors.items():
        mask = df["airport"] == ap
        ax.scatter(df.loc[mask, "true"], df.loc[mask, "pred"],
                  c=color, alpha=0.4, s=15, label=ap, edgecolors="none")
    
    lim = max(df["true"].max(), df["pred"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.6, linewidth=1.5, label="Prédiction parfaite")
    
    ax.set_xlabel("Temps restant réel (min)", fontsize=11)
    ax.set_ylabel("Temps restant prédit (min)", fontsize=11)
    ax.set_title("GRU Gaussian — Prédiction vs Réalité", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    savefig("phase2bis_scatter")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 5 : Calibration (reliability diagram)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_calibration():
    path = DATA_DIR / "errors_v3_gru_gaussian.parquet"
    if not path.exists():
        print("  ⚠ Fichier non trouvé, calibration ignorée")
        return
    
    df = pd.read_parquet(path)
    if "uncertainty" not in df.columns:
        print("  ⚠ Pas d'incertitude, calibration ignorée")
        return
    
    errors = np.abs(df["true"] - df["pred"])
    uncertainties = df["uncertainty"]
    
    from scipy.stats import norm
    confidence_levels = np.linspace(0.1, 0.99, 20)
    observed_coverage = []
    for cl in confidence_levels:
        z = norm.ppf((1 + cl) / 2)
        in_interval = errors <= z * uncertainties
        observed_coverage.append(in_interval.mean())
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=1.5, label="Calibration parfaite")
    ax.plot(confidence_levels, observed_coverage, "o-", color=COLORS["meteoBlue"],
            markersize=5, linewidth=1.5, label="GRU Gaussian")
    ax.fill_between(confidence_levels, confidence_levels, observed_coverage,
                    alpha=0.15, color=COLORS["meteoCyan"])
    
    ax.set_xlabel("Niveau de confiance attendu", fontsize=11)
    ax.set_ylabel("Couverture observée", fontsize=11)
    ax.set_title("Diagramme de fiabilité — Calibration", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    
    plt.tight_layout()
    savefig("phase2bis_calibration")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 6 : MAE par fraction de session
# ═══════════════════════════════════════════════════════════════════════════════

def fig_mae_by_fraction():
    path = DATA_DIR / "errors_v3_gru_gaussian.parquet"
    if not path.exists():
        print("  ⚠ Fichier non trouvé")
        return
    
    df = pd.read_parquet(path)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    fracs = sorted(df["frac"].unique())
    maes = [np.mean(np.abs(df[df["frac"]==f]["true"] - df[df["frac"]==f]["pred"])) for f in fracs]
    biases = [np.mean(df[df["frac"]==f]["pred"] - df[df["frac"]==f]["true"]) for f in fracs]
    
    frac_labels = [f"{int(f*100)}% de la session" for f in fracs]
    x = np.arange(len(fracs))
    
    bars = ax.bar(x, maes, color=COLORS["meteoBlue"], edgecolor="white", linewidth=1.2, width=0.6, label="MAE")
    for bar, v in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    ax2 = ax.twinx()
    ax2.plot(x, biases, "o-", color=COLORS["meteoRed"], markersize=8, linewidth=1.5, label="Biais")
    ax2.axhline(y=0, color=COLORS["gray"], linestyle=":", alpha=0.5)
    
    ax.set_xlabel("Progression dans la session", fontsize=11)
    ax.set_ylabel("MAE (minutes)", fontsize=11)
    ax2.set_ylabel("Biais (minutes)", fontsize=11, color=COLORS["meteoRed"])
    ax.set_title("Performance selon l'avancement de la session", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(frac_labels, fontsize=10)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    savefig("phase2bis_mae_by_fraction")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  Génération des figures Phase 2bis")
    print("=" * 50)
    
    print("\nFig: Comparaison ablative...")
    fig_ablation()
    
    print("Fig: Performance par aéroport...")
    fig_per_airport()
    
    print("Fig: Distribution des erreurs...")
    fig_error_distribution()
    
    print("Fig: Scatter prédiction vs réalité...")
    fig_scatter()
    
    print("Fig: Calibration de l'incertitude...")
    fig_calibration()
    
    print("Fig: MAE par fraction de session...")
    fig_mae_by_fraction()
    
    print(f"\n[DONE] Figures générées dans {OUT_DIR}/")
