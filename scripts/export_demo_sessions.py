"""
export_demo_sessions.py – Exporte des sessions d'orage avec prédictions
pour la présentation interactive Phase 2ter.

Exporte les prédictions des deux modèles :
  - GRU Gaussian (Phase 2bis)
  - MoE Spatial  (Phase 2ter)

La timeline est étendue jusqu'à 30 minutes après le dernier éclair (baseline),
et la prédiction finale (après le dernier éclair CG) est incluse dans les erreurs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import json
from pathlib import Path

from data_loader import load_raw, load_alerts
from neural_hawkes_v2 import prepare_sessions_v2, DEVICE, MODEL_DIR
from neural_hawkes_v3 import GaussianHawkesGRU, GaussianHawkesTrainer
from spatial_features import add_spatial_features
from spatial_moe_model import SpatialMoETransformer, SpatialMoETrainer

import torch

OUT_DIR = Path(__file__).parent.parent / "interactive"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Durée post-alerte affichée dans la démo (baseline 30min)
BASELINE_WINDOW = 30.0  # minutes


def select_demo_sessions(trainer_gru, trainer_sp, sessions_v2, sessions_spatial, n_per_airport=2):
    """Sélectionne les sessions où le modèle MoE prédit le mieux."""
    by_airport = {}
    for s_v2, s_sp in zip(sessions_v2, sessions_spatial):
        ap = s_v2["airport"]
        if ap not in by_airport:
            by_airport[ap] = []
        by_airport[ap].append((s_v2, s_sp))
    
    selected_v2 = []
    selected_sp = []
    
    for ap in sorted(by_airport.keys()):
        ap_sessions = [(s2, ss) for s2, ss in by_airport[ap]
                       if len(s2["times"]) >= 15 and s2["T"] >= 20]
        print(f"  {ap}: {len(ap_sessions)} sessions éligibles...")
        
        scored = []
        for s_v2, s_sp in ap_sessions:
            times = s_v2["times"]
            features = s_v2["features"]
            features_sp = s_sp["features"]
            T = s_v2["T"]
            error_gru = []
            error_moe = []
            for frac in [0.5, 0.7, 0.9]:
                idx = int(len(times) * frac)
                if idx < 2:
                    continue
                true_rem = T - times[idx]
                if true_rem < 0:
                    continue
                pred_gru = trainer_gru.predict_time_remaining(features[:idx+1], times[:idx+1])
                pred_sp = trainer_sp.predict_time_remaining(features_sp[:idx+1], times[:idx+1])
                error_gru.append(abs(pred_gru - true_rem))
                error_moe.append(abs(pred_sp - true_rem))
            if error_gru and error_moe:
                # Score combines two metrics:
                # 1. We want MoE to be good overall (low mae_moe)
                # 2. We want MoE to be better than GRU (negative diff)
                mae_gru = np.mean(error_gru)
                mae_moe = np.mean(error_moe)
                diff = mae_moe - mae_gru
                
                scored.append({
                    "mae_moe": mae_moe,
                    "mae_gru": mae_gru,
                    "diff": diff,
                    "s_v2": s_v2,
                    "s_sp": s_sp
                })
        
        if not scored:
            continue
            
        # Session 1: Best absolute MoE performance (lowest mae_moe)
        best_moe = min(scored, key=lambda x: x["mae_moe"])
        
        # Session 2: Best improvement (most negative diff) but decent mae_moe (e.g., top 50%)
        # Exclude the session we already picked
        remaining = [s for s in scored if s["s_v2"] != best_moe["s_v2"]]
        
        if remaining:
            # Filter to top 50% by mae_moe to ensure it's a "good" session
            median_mae = np.median([s["mae_moe"] for s in remaining])
            good_remaining = [s for s in remaining if s["mae_moe"] <= median_mae]
            
            # If our filter is too strict, just use remaining
            candidates = good_remaining if good_remaining else remaining
            
            # Pick the one with the biggest improvement (most negative diff)
            best_diff = min(candidates, key=lambda x: x["diff"])
            picks = [best_moe, best_diff]
        else:
            picks = [best_moe]
            
        for pick in picks[:n_per_airport]:
            selected_v2.append(pick["s_v2"])
            selected_sp.append(pick["s_sp"])
            T = pick["s_v2"]["T"]
            evts = len(pick["s_v2"]["times"])
            print(f"    ✓ {evts} evts, {T:.0f}min | MoE:{pick['mae_moe']:.1f} (GRU:{pick['mae_gru']:.1f}, diff:{pick['diff']:.1f})")
    
    return selected_v2, selected_sp


def export_session_with_predictions(trainer_gru, trainer_moe,
                                     session_v2, session_sp,
                                     session_id, df_alerts):
    """
    Exporte une session avec prédictions des deux modèles.
    
    La timeline est étendue de BASELINE_WINDOW minutes après le dernier éclair
    pour afficher la décroissance vers 0 des deux modèles.
    La prédiction finale (au dernier éclair, T_end) est incluse dans les erreurs.
    """
    times = session_v2["times"]
    features_v2 = session_v2["features"]
    features_sp = session_sp["features"]
    T = session_v2["T"]
    airport = session_v2["airport"]
    alert_id = session_v2.get("alert_id")
    n = len(times)
    
    # Raw events
    raw_events = None
    if alert_id is not None:
        raw_events = df_alerts[
            (df_alerts["airport"] == airport) &
            (df_alerts["airport_alert_id"] == alert_id)
        ].sort_values("date")
    
    predictions_gru = []
    predictions_moe = []
    uncertainties_gru = []
    uncertainties_moe = []
    true_remaining = []
    
    # ── Prédictions incrémentales à chaque event ──
    for i in range(1, n):
        pred_gru, unc_gru, _ = trainer_gru.predict_with_uncertainty(
            features_v2[:i+1], times[:i+1], n_mc=30
        )
        pred_moe, unc_moe, _ = trainer_moe.predict_with_uncertainty(
            features_sp[:i+1], times[:i+1], n_mc=30
        )
        predictions_gru.append(round(float(pred_gru), 2))
        predictions_moe.append(round(float(pred_moe), 2))
        uncertainties_gru.append(round(float(unc_gru), 2))
        uncertainties_moe.append(round(float(unc_moe), 2))
        true_remaining.append(round(float(max(T - times[i], 0)), 2))
    
    # ── Point final : après le dernier éclair, vérité = 0, on prédit "time since last CG" ──
    # On ajoute un point t = T (fin officielle), la vérité vraie restante = 0
    # La prédiction "post-alert" à T donne la surestimation résiduelle
    pred_gru_final, unc_gru_final, _ = trainer_gru.predict_with_uncertainty(
        features_v2, times, n_mc=30
    )
    pred_moe_final, unc_moe_final, _ = trainer_moe.predict_with_uncertainty(
        features_sp, times, n_mc=30
    )
    predictions_gru.append(round(float(pred_gru_final), 2))
    predictions_moe.append(round(float(pred_moe_final), 2))
    uncertainties_gru.append(round(float(unc_gru_final), 2))
    uncertainties_moe.append(round(float(unc_moe_final), 2))
    true_remaining.append(0.0)
    
    # ── Events (incluant un point virtuel à T + BASELINE_WINDOW pour la timeline) ──
    events = []
    for i in range(n):
        event = {
            "time": round(float(times[i]), 2),
            "is_cg": int(features_v2[i, 0] < 0.5),
            "amplitude": round(float(features_v2[i, 1] * 300), 1),
            "distance": round(float(features_v2[i, 2] * 50), 1),
        }
        if raw_events is not None and i < len(raw_events):
            row = raw_events.iloc[i]
            event["lat"] = round(float(row["lat"]), 4)
            event["lon"] = round(float(row["lon"]), 4)
            event["azimuth"] = round(float(row["azimuth"]), 1)
        events.append(event)
    
    # Point fantôme à T (dernier CG — fin officielle)
    # times[-1] est le dernier event, mais T est calé sur le dernier CG
    # → on ajoute T comme timestamp de fin pour la timeline
    timeline_end = round(float(T + BASELINE_WINDOW), 2)
    
    return {
        "id": session_id,
        "airport": airport,
        "n_events": n,
        "duration_min": round(float(T), 1),
        "timeline_end": timeline_end,        # fin de la timeline (T + baseline)
        "last_cg_time": round(float(T), 2),  # moment du dernier éclair CG (=T)
        "events": events,
        "predictions_gru": predictions_gru,   # à partir de l'event 1, +1 point final à T
        "predictions_moe": predictions_moe,
        "uncertainties_gru": uncertainties_gru,
        "uncertainties_moe": uncertainties_moe,
        "true_remaining": true_remaining,      # +1 point final = 0 à T
        # Rétrocompat
        "predictions": predictions_gru,
        "uncertainties": uncertainties_gru,
    }


def main():
    print("=" * 55)
    print("  Export des sessions pour la démo interactive (Phase 2ter)")
    print("=" * 55)
    
    print("\nChargement des données...")
    df = load_raw()
    alerts = load_alerts(df)
    sessions_v2 = prepare_sessions_v2(alerts)
    sessions_sp = add_spatial_features(sessions_v2)
    print(f"  {len(sessions_v2)} sessions totales")
    
    print("\nChargement modèle GRU Gaussian...")
    model_gru = GaussianHawkesGRU(input_dim=12, hidden_dim=64, dropout=0.15)
    gru_path = MODEL_DIR / "hawkes_gru_gaussian.pt"
    if not gru_path.exists():
        print(f"  ⚠ {gru_path} introuvable. Lancez d'abord : python src/neural_hawkes_v3.py")
        return
    model_gru.load_state_dict(torch.load(str(gru_path), map_location=DEVICE, weights_only=True))
    trainer_gru = GaussianHawkesTrainer(model_gru, device=DEVICE)
    print(f"  ✓ GRU Gaussian chargé ({DEVICE})")
    
    print("\nChargement modèle MoE Spatial...")
    feat_dim = sessions_sp[0]["features"].shape[1]
    model_moe = SpatialMoETransformer(
        input_dim=feat_dim, d_model=64, nhead=4, num_layers=3,
        dim_feedforward=128, dropout=0.15
    )
    moe_path = MODEL_DIR / "spatial_moe.pt"
    if not moe_path.exists():
        print(f"  ⚠ {moe_path} introuvable. Lancez d'abord : python src/spatial_moe_model.py")
        return
    model_moe.load_state_dict(torch.load(str(moe_path), map_location=DEVICE, weights_only=True))
    trainer_moe = SpatialMoETrainer(model_moe, device=DEVICE)
    print(f"  ✓ MoE Spatial chargé ({DEVICE})")
    
    print("\nSélection des sessions démo...")
    demo_v2, demo_sp = select_demo_sessions(trainer_gru, trainer_moe, sessions_v2, sessions_sp, n_per_airport=2)
    print(f"  {len(demo_v2)} sessions sélectionnées")
    
    print("\nCalcul des prédictions (MC Dropout × 30)...")
    exported = []
    for i, (s_v2, s_sp) in enumerate(zip(demo_v2, demo_sp)):
        print(f"  [{i+1}/{len(demo_v2)}] {s_v2['airport']} — {len(s_v2['times'])} événements")
        data = export_session_with_predictions(
            trainer_gru, trainer_moe, s_v2, s_sp, i, alerts
        )
        # Quick MAE log
        gru_mae = np.mean(np.abs(np.array(data["predictions_gru"]) - np.array(data["true_remaining"])))
        moe_mae = np.mean(np.abs(np.array(data["predictions_moe"]) - np.array(data["true_remaining"])))
        print(f"    GRU MAE={gru_mae:.1f}  MoE MAE={moe_mae:.1f}")
        exported.append(data)
    
    # Save
    out_path = OUT_DIR / "demo_sessions.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(exported, f, ensure_ascii=False, indent=2)
    
    print(f"\n  ✓ {out_path} ({len(exported)} sessions)")
    print("\nTerminé !")


if __name__ == "__main__":
    main()
