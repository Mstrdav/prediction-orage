import sys
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

# Add src to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from data_loader import load_raw, load_alerts
from neural_hawkes_v2 import prepare_sessions_v2
from spatial_features import add_spatial_features
from spatial_moe_model import SpatialMoETransformer, SpatialMoETrainer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = Path("models")

def get_confidence_from_uncertainty(sigma):
    """
    Convertie l'incertitude (sigma en minutes) en un score de confiance [0, 1].
    Un sigma de 0 -> confiance 1.0.
    Un sigma de 30min -> confiance 0.5.
    """
    # Exponentielle inverse ou simple ratio
    return max(0.01, min(1.0, 30.0 / (30.0 + sigma)))

def main():
    print("Export des prédictions pour Data Battle")
    print("-" * 55)
    
    print("\nChargement des données...")
    df = pd.read_csv("dataset_set.csv")
    print(f"  Dataset chargé: {len(df)} lignes")
    
    # On convertit les dates si nécessaire
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    # Extraire les alertes (format attendu par le code existant)
    print("  Extraction des alertes et calcul des features spatiales...")
    alerts = load_alerts(df)
    sessions_v2 = prepare_sessions_v2(alerts)
    sessions_sp = add_spatial_features(sessions_v2)
    print(f"  {len(sessions_sp)} sessions préparées")
    
    print("\nChargement modèle MoE Spatial...")
    feat_dim = sessions_sp[0]["features"].shape[1]
    model_moe = SpatialMoETransformer(
        input_dim=feat_dim, d_model=64, nhead=4, num_layers=3,
        dim_feedforward=128, dropout=0.15
    )
    moe_path = MODEL_DIR / "spatial_moe.pt"
    if not moe_path.exists():
        print(f"Erreur: modèle introuvable : {moe_path}")
        return
    model_moe.load_state_dict(torch.load(str(moe_path), map_location=DEVICE, weights_only=True))
    trainer_moe = SpatialMoETrainer(model_moe, device=DEVICE)
    print(f"  ✓ MoE Spatial chargé ({DEVICE})")
    
    print("\nCalcul des prédictions pour chaque événement...")
    rows = []
    
    for s_sp in tqdm(sessions_sp, desc="Sessions"):
        airport = s_sp["airport"]
        alert_id = s_sp["alert_id"]
        times = s_sp["times"]  # Minutes since first event
        times = s_sp["times"]  # Minutes since first event
        features = s_sp["features"]
        start_time = s_sp["start_time"]
        
        # We need to predict at every timestamp
        for i in range(len(times)):
            idx = i + 1
            # Current event true date
            pred_date = start_time + timedelta(minutes=float(times[i]))
            
            # Predict
            pred_rem, uncert, _ = trainer_moe.predict_with_uncertainty(features[:idx], times[:idx], n_mc=30)
            
            # End date prediction
            end_date = pred_date + timedelta(minutes=float(pred_rem))
            
            # Confidence
            conf = get_confidence_from_uncertainty(float(uncert))
            
            rows.append({
                "airport": airport,
                "airport_alert_id": alert_id,
                "prediction_date": pred_date,
                "predicted_date_end_alert": end_date,
                "confidence": conf
            })
    
    print("\nSauvegarde dans predictions.csv...")
    df_preds = pd.DataFrame(rows)
    df_preds.to_csv("predictions.csv", index=False)
    print(f"✓ Fichier predictions.csv généré avec {len(df_preds)} prédictions.")

if __name__ == "__main__":
    main()
