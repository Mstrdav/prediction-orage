"""
data_loader.py – Chargement et préparation du dataset Météorage.
"""
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "segment_alerts_all_airports_train.csv"

AIRPORT_COLORS = {
    "Ajaccio": "#E63946", # meteoRed
    "Bastia":  "#E07A5F", # warm orange
    "Biarritz":"#457B9D", # meteoCyan
    "Nantes":  "#A8DADC", # light meteoCyan
    "Pise":    "#1D3557", # meteoBlue
}

def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Charge le CSV brut et parse les dates."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["icloud"] = df["icloud"].astype(bool)
    # is_last_lightning_cloud_ground: True/False/NaN
    df["is_last_cg"] = df["is_last_lightning_cloud_ground"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    return df


def load_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Garde uniquement les lignes faisant partie d'une session d'alerte."""
    return df[df["airport_alert_id"].notna()].copy()


def get_alert_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les éclairs par session d'alerte.
    Retourne un DataFrame avec une ligne par session.
    """
    alerts = load_alerts(df)
    grp = alerts.groupby(["airport", "airport_alert_id"])

    sessions = grp.agg(
        n_lightnings=("lightning_id", "count"),
        n_cg=("icloud", lambda x: (~x).sum()),
        n_ic=("icloud", "sum"),
        start_time=("date", "min"),
        end_time=("date", "max"),
        mean_dist=("dist", "mean"),
        mean_amplitude=("amplitude", "mean"),
    ).reset_index()

    sessions["duration_min"] = (
        (sessions["end_time"] - sessions["start_time"]).dt.total_seconds() / 60
    )
    sessions["icloud_ratio"] = sessions["n_ic"] / sessions["n_lightnings"]
    return sessions
