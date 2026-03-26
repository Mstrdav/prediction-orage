"""
generate_eda_figures.py - Genere toutes les figures du rapport EDA.
Outputs: report/figures/*.pdf  (format vectoriel pour LaTeX)
"""
import sys, os
# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import seaborn as sns
from data_loader import load_raw, load_alerts, get_alert_sessions, AIRPORT_COLORS

# ── Répertoire de sortie ──────────────────────────────────────────────────────
OUT = "report/figures"
os.makedirs(OUT, exist_ok=True)

# ── Style global ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = list(AIRPORT_COLORS.values())
fig_files = []

def savefig(name):
    p = f"{OUT}/{name}.pdf"
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    fig_files.append(p)
    print(f"  [OK] {p}")

# ═══════════════════════════════════════════════════════════════════════════════
print("Chargement des données…")
df = load_raw()
alerts = load_alerts(df)
sessions = get_alert_sessions(df)
print(f"  {len(df):,} eclairs | {df['airport'].nunique()} aeroports | {len(sessions)} sessions")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 – Distribution des éclairs par aéroport (bar + pie)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 1 – Éclairs par aéroport…")
counts = df.groupby("airport").size().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(counts.index, counts.values,
            color=[AIRPORT_COLORS[a] for a in counts.index], edgecolor="white", linewidth=0.5)
axes[0].set_title("Nombre total d'éclairs par aéroport")
axes[0].set_ylabel("Nombre d'éclairs")
axes[0].set_xlabel("")
for i, (a, v) in enumerate(counts.items()):
    axes[0].text(i, v + 500, f"{v:,}", ha="center", va="bottom", fontsize=8)

axes[1].pie(counts.values, labels=counts.index,
            colors=[AIRPORT_COLORS[a] for a in counts.index],
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 0.8})
axes[1].set_title("Répartition des éclairs")

plt.suptitle("Répartition des éclairs par aéroport", fontweight="bold", y=1.01)
plt.tight_layout()
savefig("fig1_airport_counts")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 – Distribution temporelle (mensuelle et horaire)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 2 – Distribution temporelle…")
df["month"] = df["date"].dt.month
df["hour"] = df["date"].dt.hour
df["year"] = df["date"].dt.year

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Par mois
monthly = df.groupby("month").size()
month_names = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
axes[0].bar(monthly.index, monthly.values, color="#457B9D", edgecolor="white")
axes[0].set_xticks(range(1,13))
axes[0].set_xticklabels(month_names)
axes[0].set_title("Distribution mensuelle des éclairs")
axes[0].set_ylabel("Nombre d'éclairs")
axes[0].set_xlabel("Mois")

# Par heure
hourly = df.groupby("hour").size()
axes[1].bar(hourly.index, hourly.values, color="#E63946", edgecolor="white")
axes[1].set_title("Distribution horaire des éclairs (UTC)")
axes[1].set_ylabel("Nombre d'éclairs")
axes[1].set_xlabel("Heure (UTC)")
axes[1].set_xticks(range(0, 24, 2))

plt.suptitle("Saisonnalité et cycle diurne des éclairs", fontweight="bold", y=1.01)
plt.tight_layout()
savefig("fig2_temporal")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 – Heatmap mois × heure
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 3 – Heatmap mois × heure…")
heat = df.groupby(["month", "hour"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(13, 4))
cmap = LinearSegmentedColormap.from_list("storm", ["#f0f4f8","#457B9D","#1d3557","#E63946"])
sns.heatmap(heat, ax=ax, cmap=cmap, linewidths=0,
            xticklabels=range(0,24), yticklabels=month_names,
            cbar_kws={"label": "Nombre d'éclairs"})
ax.set_title("Densité des éclairs : mois × heure (UTC)", fontweight="bold")
ax.set_xlabel("Heure (UTC)")
ax.set_ylabel("Mois")
plt.tight_layout()
savefig("fig3_heatmap_month_hour")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 – Heatmap par aéroport et mois
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 4 – Heatmap aéroport × mois…")
heat2 = df.groupby(["airport", "month"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12, 3.5))
sns.heatmap(heat2, ax=ax, cmap=cmap, linewidths=0.4, linecolor="white",
            annot=True, fmt="d", annot_kws={"size": 7},
            xticklabels=month_names,
            cbar_kws={"label": "Nombre d'éclairs"})
ax.set_title("Activité orageuse mensuelle par aéroport", fontweight="bold")
ax.set_xlabel("Mois")
ax.set_ylabel("")
plt.tight_layout()
savefig("fig4_heatmap_airport_month")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 – Caractéristiques physiques des éclairs
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 5 – Caractéristiques physiques…")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Amplitude
axes[0,0].hist(df["amplitude"].clip(-100, 100), bins=80, color="#457B9D",
               edgecolor="none", density=True)
axes[0,0].axvline(0, color="red", linestyle="--", linewidth=0.8, label="0 kA")
axes[0,0].set_title("Distribution de l'amplitude (kA)")
axes[0,0].set_xlabel("Amplitude (kA)")
axes[0,0].set_ylabel("Densité")
axes[0,0].legend()

# Maxis
axes[0,1].hist(df["maxis"].clip(0, 6), bins=60, color="#F4A261",
               edgecolor="none", density=True)
axes[0,1].set_title("Distribution de maxis")
axes[0,1].set_xlabel("Maxis")
axes[0,1].set_ylabel("Densité")

# Distance
axes[1,0].hist(df["dist"], bins=60, color="#2A9D8F", edgecolor="none", density=True)
axes[1,0].set_title("Distribution de la distance à l'aéroport (km)")
axes[1,0].set_xlabel("Distance (km)")
axes[1,0].set_ylabel("Densité")

# icloud ratio par aéroport
ic_ratio = df.groupby("airport")["icloud"].mean().sort_values(ascending=False)
bars = axes[1,1].bar(ic_ratio.index, ic_ratio.values * 100,
                     color=[AIRPORT_COLORS[a] for a in ic_ratio.index],
                     edgecolor="white")
axes[1,1].set_title("Part des éclairs intra-cloud (IC) par aéroport")
axes[1,1].set_ylabel("% éclairs IC")
axes[1,1].set_ylim(0, 100)
for bar, v in zip(bars, ic_ratio.values):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, v*100 + 1,
                   f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)

plt.suptitle("Caractéristiques physiques des éclairs", fontweight="bold", y=1.01)
plt.tight_layout()
savefig("fig5_physics")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 – Rose des azimuts par aéroport
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 6 – Rose des azimuts…")
airports = df["airport"].unique()
n = len(airports)
fig = plt.figure(figsize=(14, 3.5))

for i, airport in enumerate(sorted(airports)):
    ax = fig.add_subplot(1, n, i+1, polar=True)
    sub = df[df["airport"] == airport]["azimuth"]
    bins = np.linspace(0, 360, 37)
    counts_az, bin_edges = np.histogram(sub, bins=bins)
    angles = np.deg2rad(bin_edges[:-1])
    width = np.deg2rad(10)
    bars = ax.bar(angles, counts_az, width=width, bottom=0,
                  color=AIRPORT_COLORS[airport], alpha=0.85, edgecolor="none")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(airport, pad=12, fontsize=9, fontweight="bold")
    ax.set_yticklabels([])
    ax.tick_params(labelsize=7)

plt.suptitle("Rose des azimuts des éclairs autour de chaque aéroport", fontweight="bold", y=1.02)
plt.tight_layout()
savefig("fig6_azimuth_rose")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 – Carte de dispersion géographique
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 7 – Carte géographique…")
airport_coords = {
    "Ajaccio":  (41.9236, 8.8029),
    "Bastia":   (42.5527, 9.4836),
    "Biarritz": (43.4683, -1.5311),
    "Nantes":   (47.1569, -1.6110),
    "Pise":     (43.6839, 10.3927),
}

fig, axes = plt.subplots(1, n, figsize=(14, 3.2))
for i, airport in enumerate(sorted(airports)):
    ax = axes[i]
    sub = df[df["airport"] == airport]
    clat, clon = airport_coords[airport]
    
    # ── Paramètres géographiques (35 km)
    # 1 degré de latitude ~ 110.574 km. 1 degré de longitude ~ 111.320 * cos(lat)
    r_lon = 35 / (111.320 * np.cos(np.deg2rad(clat)))
    r_lat = 35 / 110.574

    # ── Hexbin (densité spatiale)
    hb = ax.hexbin(sub["lon"], sub["lat"], gridsize=40, cmap="viridis", 
                   mincnt=1, bins="log", rasterized=True)
    ax.scatter([clon], [clat], s=80, color="red", zorder=5, marker="*")
            
    # Zoom dynamique autour des 35 km (petite marge pour aérer)
    margin = 1.05
    ax.set_xlim(clon - r_lon * margin, clon + r_lon * margin)
    ax.set_ylim(clat - r_lat * margin, clat + r_lat * margin)
    
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude", fontsize=7)
    ax.tick_params(labelsize=6)
        
    ax.set_title(airport, fontsize=9, fontweight="bold")
    if i == 0:
        ax.set_ylabel("Latitude", fontsize=7)



fig.suptitle("Densité spatiale des éclairs - Fenêtre de 35 km", fontweight="bold")
plt.tight_layout()
savefig("fig7_geo_scatter")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 – Analyse des sessions d'alerte
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 8 – Sessions d'alerte…")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Durée des sessions (filtre des sessions > 1 éclair pour la durée)
valid_sessions = sessions[sessions["n_lightnings"] > 1]
axes[0].hist(valid_sessions["duration_min"].clip(0, 300), bins=50,
             color="#8338EC", edgecolor="none", density=True)
axes[0].set_title("Durée des sessions d'alerte (>1 éclair)")
axes[0].set_xlabel("Durée (minutes)")
axes[0].set_ylabel("Densité")
med_dur = valid_sessions["duration_min"].median()
axes[0].axvline(med_dur, color="red", linestyle="--", label=f"Médiane: {med_dur:.0f} min")
axes[0].legend(fontsize=8)

# Nb éclairs par session
axes[1].hist(sessions["n_lightnings"].clip(0, 500), bins=50,
             color="#F4A261", edgecolor="none", density=True)
axes[1].set_title("Nombre d'éclairs par session")
axes[1].set_xlabel("Nombre d'éclairs")
axes[1].set_ylabel("Densité")

# Sessions par aéroport
sess_counts = sessions.groupby("airport").size().sort_values(ascending=False)
axes[2].bar(sess_counts.index, sess_counts.values,
            color=[AIRPORT_COLORS[a] for a in sess_counts.index], edgecolor="white")
axes[2].set_title("Nombre de sessions par aéroport")
axes[2].set_ylabel("Nombre de sessions d'alerte")
for j, (a, v) in enumerate(sess_counts.items()):
    axes[2].text(j, v + 0.5, str(v), ha="center", va="bottom", fontsize=8)

plt.suptitle("Analyse des sessions d'alerte orage", fontweight="bold", y=1.01)
plt.tight_layout()
savefig("fig8_sessions")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 – Durée médiane des sessions par aéroport (Boxplot -> ViolinPlot)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 9 – Durée sessions par aéroport…")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

airports_sorted = sorted(sessions["airport"].unique())
valid_sessions = sessions[sessions["n_lightnings"] > 1]
data_dur = [valid_sessions[valid_sessions["airport"]==a]["duration_min"].values for a in airports_sorted]

vp1 = axes[0].violinplot(data_dur, showmeans=False, showextrema=False, showmedians=True)
for i, body in enumerate(vp1["bodies"]):
    body.set_facecolor(AIRPORT_COLORS[airports_sorted[i]])
    body.set_alpha(0.8)
axes[0].set_xticks(range(1, len(airports_sorted)+1))
axes[0].set_xticklabels(airports_sorted)
axes[0].set_title("Durée des sessions (> 1 éclair)")
axes[0].set_ylabel("Durée (minutes) - Echelle Log")
axes[0].set_yscale("log")

data_nl = [sessions[sessions["airport"]==a]["n_lightnings"].values for a in airports_sorted]
vp2 = axes[1].violinplot(data_nl, showmeans=False, showextrema=False, showmedians=True)
for i, body in enumerate(vp2["bodies"]):
    body.set_facecolor(AIRPORT_COLORS[airports_sorted[i]])
    body.set_alpha(0.8)
axes[1].set_xticks(range(1, len(airports_sorted)+1))
axes[1].set_xticklabels(airports_sorted)
axes[1].set_title("Nombre d'éclairs par session")
axes[1].set_ylabel("Nombre d'éclairs - Echelle Log")
axes[1].set_yscale("log")

plt.suptitle("Comparaison des sessions d'alerte par aéroport", fontweight="bold", y=1.01)
plt.tight_layout()
savefig("fig9_sessions_by_airport")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 – Analyse de la variable cible
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 10 – Variable cible…")
cg_only = alerts[~alerts["icloud"]]  # éclairs cloud-to-ground dans sessions

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Répartition de la cible (Bar chart en log-scale)
target_counts = cg_only["is_last_cg"].value_counts(dropna=True)
labels_map = {True: "Dernier éclair CG", False: "Éclair CG non final"}
labels_str = [labels_map.get(k, str(k)) for k in target_counts.index]

axes[0].bar(labels_str, target_counts.values, color=["#457B9D", "#E63946"])
axes[0].set_yscale("log")
axes[0].set_ylabel("Nombre d'éclairs (log scale)")
axes[0].set_title("Répartition de la cible\n(éclairs CG en session)")
for i, v in enumerate(target_counts.values):
    pct = v / target_counts.sum() * 100
    axes[0].text(i, v * 1.2, f"{v:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
axes[0].set_ylim(top=target_counts.max() * 5)

# Distribution du temp depuis dernier CG jusqu'à la fin
# Pour chaque session, calculer le temps entre chaque CG et le dernier CG
def inter_arrival_stats(group):
    cg = group[~group["icloud"]].sort_values("date")
    if len(cg) < 2:
        return None
    last_time = cg["date"].iloc[-1]
    cg = cg.iloc[:-1]  # tous sauf le dernier
    times_to_end = (last_time - cg["date"]).dt.total_seconds() / 60
    return times_to_end

time_to_ends = []
for (airport, alert_id), grp in alerts.groupby(["airport", "airport_alert_id"]):
    r = inter_arrival_stats(grp)
    if r is not None:
        time_to_ends.extend(r.tolist())

time_to_ends = pd.Series(time_to_ends)

axes[1].hist(time_to_ends.clip(0, 120), bins=80, color="#2A9D8F",
             edgecolor="none", density=True)
axes[1].axvline(30, color="red", linestyle="--", linewidth=1.2, label="Règle 30 min (actuelle)")
axes[1].axvline(60, color="orange", linestyle="--", linewidth=1.2, label="Règle 60 min")
axes[1].set_title("Temps restant jusqu'à fin de session\n(depuis chaque éclair CG non-final)")
axes[1].set_xlabel("Minutes restantes")
axes[1].set_ylabel("Densité")
axes[1].legend(fontsize=8)

# Inter-arrival times CG dans une session
def cg_inter_arrivals(group):
    cg = group[~group["icloud"]].sort_values("date")
    if len(cg) < 2:
        return []
    return (cg["date"].diff().dropna().dt.total_seconds() / 60).tolist()

all_ia = []
for (airport, alert_id), grp in alerts.groupby(["airport", "airport_alert_id"]):
    all_ia.extend(cg_inter_arrivals(grp))

all_ia = pd.Series(all_ia)
axes[2].hist(all_ia.clip(0, 60), bins=80, color="#8338EC",
             edgecolor="none", density=True)
axes[2].set_title("Intervalles inter-éclairs CG\ndans une session d'alerte")
axes[2].set_xlabel("Intervalle (minutes)")
axes[2].set_ylabel("Densité")

plt.suptitle("Analyse de la variable cible et des dynamiques temporelles", fontweight="bold", y=1.01)
plt.tight_layout()
savefig("fig10_target_analysis")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 11 – Exemple de session d'alerte (timeline)
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 11 – Exemple de session…")
# Choisir une session de taille représentative AVEC des IC (critères relâchés)
good_sessions = sessions[(sessions["n_lightnings"] > 50) & (sessions["n_ic"] >= 5) & (sessions["duration_min"] > 30) & (sessions["duration_min"] < 150)]
if len(good_sessions) > 0:
    ex = good_sessions.sort_values("n_ic", ascending=False).iloc[0]
    ex_airport = ex["airport"]
    ex_alert_id = ex["airport_alert_id"]
    ex_data = alerts[(alerts["airport"] == ex_airport) & (alerts["airport_alert_id"] == ex_alert_id)].sort_values("date")

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    t0 = ex_data["date"].min()
    ex_data = ex_data.copy()
    ex_data["t_min"] = (ex_data["date"] - t0).dt.total_seconds() / 60

    cg = ex_data[~ex_data["icloud"]]
    ic = ex_data[ex_data["icloud"]]

    axes[0].scatter(ic["t_min"], ic["dist"], s=12, alpha=0.5, color="#8338EC", label="Intra-cloud", zorder=2)
    axes[0].scatter(cg["t_min"], cg["dist"], s=18, alpha=0.8, color="#E63946", label="Cloud-to-ground", zorder=3)
    last_cg = cg[cg["is_last_cg"] == True]
    if len(last_cg) > 0:
        axes[0].axvline(last_cg["t_min"].values[0], color="black", linestyle="--",
                        linewidth=1.5, label=f"Dernier éclair CG ({last_cg['t_min'].values[0]:.1f} min)")
    axes[0].set_ylabel("Distance à l'aéroport (km)")
    axes[0].set_title(f"Session d'alerte – {ex_airport} (alerte #{int(ex_alert_id)})", fontweight="bold")
    axes[0].legend(fontsize=8)

    axes[1].bar(cg["t_min"], cg["amplitude"].abs(), width=0.5,
                color="#E63946", alpha=0.7, label="Amplitude |CG|")
    axes[1].set_xlabel("Temps depuis début de session (minutes)")
    axes[1].set_ylabel("|Amplitude| (kA)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    savefig("fig11_session_example")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 12 – Matrice de corrélation
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 12 – Matrice de corrélation…")
num_cols = ["amplitude", "maxis", "dist", "azimuth"]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(6, 5))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
ax.set_title("Matrice de corrélation – variables physiques", fontweight="bold")
plt.tight_layout()
savefig("fig12_correlation")

print(f"\n{'='*60}")
print(f"[DONE] {len(fig_files)} figures generees dans {OUT}/")
for f in fig_files:
    print(f"   {f}")
