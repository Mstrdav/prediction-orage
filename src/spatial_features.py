"""
spatial_features.py – Phase 2ter : Features spatiales enrichies pour MoE.

Ajoute 8 features spatiales aux 12 features existantes de prepare_sessions_v2,
soit 20 features par événement.

Features additionnelles :
  12  radial_velocity     : dR/dt sur les 5 derniers éclairs (km/min)
  13  angular_spread      : écart-type de l'azimuth sur les 10 derniers éclairs
  14  dist_5km_ratio      : ratio d'éclairs dans rayon 5km (fenêtre 10)
  15  retreat_indicator   : 1 si radial_velocity > 0 ET dist_5km_ratio en baisse
  16  receding_duration   : minutes continues pendant lesquelles l'orage recule
  17  centroid_vx         : composante x de la vitesse du centroïde récent
  18  centroid_vy         : composante y de la vitesse du centroïde récent
  19  min_dist_5min       : distance minimale dans les 5 dernières minutes
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from data_loader import load_raw, load_alerts
from neural_hawkes_v2 import prepare_sessions_v2


def add_spatial_features(sessions):
    """
    Enrichit chaque session avec 8 features spatiales supplémentaires.
    Modifie le champ 'features' de chaque session (12 → 20 colonnes).
    """
    enriched = []
    
    for s in sessions:
        n = len(s["times"])
        features_base = s["features"]  # (n, 12)
        times = s["times"]
        
        # Colonnes existantes dans features_base :
        #  0=icloud, 1=amp, 2=dist(norm/50), 3=maxis, 4=az_sin, 5=az_cos,
        #  6=dt, 7=dt_cg, 8=density, 9=ic_ratio, 10=dist_centroid, 11=dist_trend
        
        dists_raw = features_base[:, 2] * 50.0  # dénormalise la distance (km)
        az_sin = features_base[:, 4]
        az_cos = features_base[:, 5]
        
        # --- 12. Radial velocity (dR/dt sur fenêtre de 5) ---
        f_radial_vel = np.zeros(n)
        for i in range(1, n):
            start_idx = max(0, i - 4)
            dt = times[i] - times[start_idx]
            if dt > 0.01:
                dr = dists_raw[i] - dists_raw[start_idx]
                f_radial_vel[i] = dr / dt  # km/min, positif = éloignement
        f_radial_vel = np.clip(f_radial_vel / 5.0, -1, 1)  # normaliser
        
        # --- 13. Angular spread (std de l'azimuth sur 10 derniers) ---
        azimuths = np.arctan2(az_sin, az_cos)  # retrouver l'azimuth en radians
        f_ang_spread = np.zeros(n)
        for i in range(n):
            start_idx = max(0, i - 9)
            window_az = azimuths[start_idx:i+1]
            if len(window_az) >= 2:
                # Circular std
                mean_sin = np.mean(np.sin(window_az))
                mean_cos = np.mean(np.cos(window_az))
                R = np.sqrt(mean_sin**2 + mean_cos**2)
                f_ang_spread[i] = np.sqrt(-2 * np.log(max(R, 1e-8)))
        f_ang_spread = np.clip(f_ang_spread / np.pi, 0, 1)  # normaliser
        
        # --- 14. dist_5km_ratio (ratio d'éclairs < 5km sur fenêtre 10) ---
        f_5km_ratio = np.zeros(n)
        for i in range(n):
            start_idx = max(0, i - 9)
            window_dist = dists_raw[start_idx:i+1]
            f_5km_ratio[i] = np.mean(window_dist < 5.0)
        
        # --- 15. Retreat indicator ---
        # 1 si radial_velocity > 0 ET 5km_ratio actuel < 5km_ratio il y a 5 éclairs
        f_retreat = np.zeros(n)
        for i in range(5, n):
            if f_radial_vel[i] > 0.02 and f_5km_ratio[i] < f_5km_ratio[i - 5]:
                f_retreat[i] = 1.0
        
        # --- 16. Receding duration (minutes continues de recul) ---
        f_receding_dur = np.zeros(n)
        for i in range(1, n):
            if f_radial_vel[i] > 0.01:
                f_receding_dur[i] = f_receding_dur[i-1] + (times[i] - times[i-1])
            else:
                f_receding_dur[i] = 0.0
        f_receding_dur = np.clip(f_receding_dur / 30.0, 0, 1)  # normaliser
        
        # --- 17-18. Centroid velocity x/y ---
        f_cvx = np.zeros(n)
        f_cvy = np.zeros(n)
        # Composantes cartésiennes approx (lat, lon → km)
        # dist * cos(azimuth) = composante Nord, dist * sin(azimuth) = composante Est
        x_pos = dists_raw * az_sin  # Est
        y_pos = dists_raw * az_cos  # Nord
        
        for i in range(2, n):
            start_idx = max(0, i - 4)
            # Centroïde à start_idx et à i
            mid = (start_idx + i) // 2
            cx_old = np.mean(x_pos[start_idx:mid+1]) if mid > start_idx else x_pos[start_idx]
            cy_old = np.mean(y_pos[start_idx:mid+1]) if mid > start_idx else y_pos[start_idx]
            cx_new = np.mean(x_pos[mid:i+1])
            cy_new = np.mean(y_pos[mid:i+1])
            dt = times[i] - times[start_idx]
            if dt > 0.01:
                f_cvx[i] = (cx_new - cx_old) / dt
                f_cvy[i] = (cy_new - cy_old) / dt
        f_cvx = np.clip(f_cvx / 5.0, -1, 1)
        f_cvy = np.clip(f_cvy / 5.0, -1, 1)
        
        # --- 19. Min distance in last 5 minutes ---
        f_min_dist = np.zeros(n)
        for i in range(n):
            window_start = times[i] - 5.0
            mask = (times >= window_start) & (times <= times[i])
            f_min_dist[i] = np.min(dists_raw[mask])
        f_min_dist = np.clip(f_min_dist / 50.0, 0, 1)  # normaliser
        
        # Stack the new features
        new_features = np.column_stack([
            features_base,        # 12 colonnes existantes
            f_radial_vel,         # 12
            f_ang_spread,         # 13
            f_5km_ratio,          # 14
            f_retreat,            # 15
            f_receding_dur,       # 16
            f_cvx,                # 17
            f_cvy,                # 18
            f_min_dist,           # 19
        ])
        
        enriched_session = dict(s)
        enriched_session["features"] = new_features
        enriched.append(enriched_session)
    
    return enriched


def prepare_sessions_spatial(alerts_df):
    """Pipeline complet : sessions v2 (12 features) → spatial (20 features)."""
    sessions_v2 = prepare_sessions_v2(alerts_df)
    return add_spatial_features(sessions_v2)


if __name__ == "__main__":
    print("Loading data...")
    df = load_raw()
    alerts = load_alerts(df)
    
    print("Preparing spatial sessions (20 features)...")
    sessions = prepare_sessions_spatial(alerts)
    print(f"  {len(sessions)} sessions")
    print(f"  Features par événement : {sessions[0]['features'].shape[1]}")
    
    # Quick stats
    for i, name in enumerate([
        "icloud", "amp", "dist", "maxis", "az_sin", "az_cos",
        "dt", "dt_cg", "density", "ic_ratio", "dist_centroid", "dist_trend",
        "radial_vel", "ang_spread", "5km_ratio", "retreat", "receding_dur",
        "centroid_vx", "centroid_vy", "min_dist"
    ]):
        vals = np.concatenate([s["features"][:, i] for s in sessions])
        print(f"  [{i:2d}] {name:16s} | min={vals.min():.3f} | max={vals.max():.3f} | mean={vals.mean():.3f}")
