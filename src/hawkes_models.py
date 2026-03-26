"""
hawkes_models.py - Processus de Hawkes Classique et Neural Hawkes Process.

Le processus de Hawkes modélise l'intensité d'arrivée des éclairs comme :
    lambda(t) = mu + sum_{t_i < t} alpha * beta * exp(-beta * (t - t_i))

Quand lambda(t) tombe sous un seuil, on prédit la fin de l'orage.

Le Neural Hawkes remplace la fonction de déclenchement exponentielle par un
réseau de neurones (GRU) qui apprend la dynamique temporelle directement.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import GroupShuffleSplit
from data_loader import load_raw, load_alerts
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------═══════════════════════════
# 1. HAWKES CLASSIQUE (Maximum de Vraisemblance)
# ----------------------------------------------------═══════════════════════════

class HawkesClassique:
    """
    Processus de Hawkes univarié avec noyau exponentiel.
    Paramètres : mu (background rate), alpha (excitation), beta (decay).
    """
    def __init__(self):
        self.mu = None
        self.alpha = None
        self.beta = None
    
    def neg_log_likelihood(self, params, timestamps, T):
        """
        Log-vraisemblance négative du processus de Hawkes.
        timestamps : array de temps d'événements (en minutes)
        T : horizon temporel total
        """
        mu, alpha, beta = params
        if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
            return 1e10
        
        n = len(timestamps)
        if n == 0:
            return mu * T
        
        # Terme 1 : somme des log-intensités aux temps d'événement
        log_intensities = 0.0
        A = 0.0  # Processus récursif
        for i in range(n):
            if i > 0:
                dt = timestamps[i] - timestamps[i-1]
                A = np.exp(-beta * dt) * (1 + A)
            lam_i = mu + alpha * beta * A
            if lam_i > 0:
                log_intensities += np.log(lam_i)
            else:
                return 1e10
        
        # Terme 2 : intégrale compensatrice
        compensator = mu * T
        for i in range(n):
            compensator += alpha * (1 - np.exp(-beta * (T - timestamps[i])))
        
        return -log_intensities + compensator
    
    def fit(self, sessions_timestamps, sessions_T):
        """
        Ajuste les paramètres sur un ensemble de sessions.
        sessions_timestamps : list de arrays de timestamps
        sessions_T : list de durées totales
        """
        def total_nll(params):
            total = 0.0
            for ts, T in zip(sessions_timestamps, sessions_T):
                total += self.neg_log_likelihood(params, ts, T)
            return total / len(sessions_timestamps)
        
        # Initialisation
        x0 = [0.1, 0.5, 1.0]
        bounds = [(1e-4, 10), (1e-4, 10), (1e-2, 50)]
        
        result = minimize(total_nll, x0, bounds=bounds, method="L-BFGS-B",
                         options={"maxiter": 500, "ftol": 1e-8})
        
        self.mu, self.alpha, self.beta = result.x
        print(f"  Hawkes Classique: mu={self.mu:.4f}, alpha={self.alpha:.4f}, beta={self.beta:.4f}")
        print(f"  NLL finale: {result.fun:.4f}")
        return self
    
    def intensity(self, t, past_times):
        """Calcule l'intensité lambda(t) étant donné les événements passés."""
        lam = self.mu
        for ti in past_times:
            if ti < t:
                lam += self.alpha * self.beta * np.exp(-self.beta * (t - ti))
        return lam
    
    def predict_end(self, past_times, excitation_threshold=0.02, max_horizon=120, dt=0.5):
        """
        Prédit le temps restant avant la fin de l'orage.
        On scanne le futur jusqu'à ce que la composante d'excitation
        (lambda(t) - mu) tombe sous excitation_threshold.
        Quand l'intensité revient quasi au taux de fond mu, l'orage est considéré fini.
        """
        if len(past_times) == 0:
            return 0.0
        
        effective_threshold = self.mu + excitation_threshold
        
        t0 = past_times[-1]
        t = t0
        while t - t0 < max_horizon:
            t += dt
            lam = self.intensity(t, past_times)
            if lam < effective_threshold:
                return t - t0
        return max_horizon


# ----------------------------------------------------═══════════════════════════
# 2. NEURAL HAWKES PROCESS (GRU-based)
# ----------------------------------------------------═══════════════════════════

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class NeuralHawkesGRU(nn.Module):
    """
    Neural Hawkes Process avec GRU.
    L'intensité est modélisée par :
        lambda(t) = softplus(W_h * h(t_last) + w_t * (t - t_last) + b)
    où h(t_last) est l'état caché du GRU après le dernier événement.
    """
    def __init__(self, input_dim=4, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.intensity_layer = nn.Linear(hidden_dim + 1, 1)  # +1 pour le delta_t
        self.softplus = nn.Softplus()
    
    def forward(self, event_features, delta_times, hidden=None):
        """
        event_features : (batch, seq_len, input_dim) - features de chaque événement
        delta_times : (batch, seq_len, 1) - temps inter-événements
        """
        # GRU sur la séquence d'événements
        gru_out, hidden = self.gru(event_features, hidden)
        
        # Intensité = f(h, dt)
        combined = torch.cat([gru_out, delta_times], dim=-1)
        intensity = self.softplus(self.intensity_layer(combined))
        
        return intensity.squeeze(-1), hidden


class HawkesDataset(Dataset):
    """Dataset pour l'entraînement du Neural Hawkes."""
    def __init__(self, sessions, max_len=200):
        self.sessions = []
        self.max_len = max_len
        
        for s in sessions:
            if len(s["times"]) < 3:
                continue
            # Tronquer/padder
            n = min(len(s["times"]), max_len)
            times = s["times"][:n]
            features = s["features"][:n]
            self.sessions.append({"times": times, "features": features, "n": n})
    
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        s = self.sessions[idx]
        n = s["n"]
        
        times = np.zeros(self.max_len)
        features = np.zeros((self.max_len, s["features"].shape[1]))
        mask = np.zeros(self.max_len)
        
        times[:n] = s["times"]
        features[:n] = s["features"]
        mask[:n] = 1.0
        
        # Delta times (inter-event)
        dt = np.zeros(self.max_len)
        dt[1:n] = np.diff(times[:n])
        
        return {
            "times": torch.FloatTensor(times),
            "features": torch.FloatTensor(features),
            "delta_times": torch.FloatTensor(dt).unsqueeze(-1),
            "mask": torch.FloatTensor(mask),
            "n": n
        }


class NeuralHawkes:
    """Wrapper d'entraînement pour le Neural Hawkes Process."""
    
    def __init__(self, input_dim=4, hidden_dim=32, lr=1e-3, device="cpu"):
        self.model = NeuralHawkesGRU(input_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
    
    def nll_loss(self, intensity, delta_times, mask):
        """
        NLL pour processus ponctuel :
          -sum log(lambda(t_i)) + integral lambda(t) dt
        On approxime l'intégrale avec la méthode trapézoïdale.
        """
        # Log-intensité aux points d'événement
        log_lambda = torch.log(intensity + 1e-8) * mask
        
        # Compensateur (approximation trapézoïdale)
        compensator = intensity * delta_times.squeeze(-1) * mask
        
        loss = -log_lambda.sum() + compensator.sum()
        n_events = mask.sum()
        
        return loss / (n_events + 1e-8)
    
    def fit(self, train_sessions, n_epochs=30, batch_size=32):
        """Entraîne le modèle sur les sessions d'entraînement."""
        dataset = HawkesDataset(train_sessions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            n_batches = 0
            for batch in loader:
                features = batch["features"].to(self.device)
                delta_times = batch["delta_times"].to(self.device)
                mask = batch["mask"].to(self.device)
                
                intensity, _ = self.model(features, delta_times)
                loss = self.nll_loss(intensity, delta_times, mask)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {total_loss/n_batches:.4f}")
    
    def predict_time_remaining(self, session_features, session_times, threshold=0.05, max_horizon=120, dt_step=0.5):
        """
        Prédit le temps restant en scannant l'intensité dans le futur.
        """
        self.model.eval()
        with torch.no_grad():
            n = len(session_times)
            if n < 2:
                return 30.0  # fallback
            
            # Préparer la séquence
            features = torch.FloatTensor(session_features).unsqueeze(0).to(self.device)
            dt = np.zeros(n)
            dt[1:] = np.diff(session_times)
            delta_times = torch.FloatTensor(dt).unsqueeze(0).unsqueeze(-1).to(self.device)
            
            # Obtenir le dernier état caché
            intensity, hidden = self.model(features, delta_times)
            last_intensity = intensity[0, -1].item()
            
            # Scanner le futur
            if last_intensity < threshold:
                return 0.0
            
            # Décroissance monotone approximée (simple extrapolation)
            # L'intensité au dernier step donne le point de départ
            t_remaining = 0.0
            current_intensity = last_intensity
            while current_intensity > threshold and t_remaining < max_horizon:
                t_remaining += dt_step
                # Approximation: décroissance exponentielle sans nouvel événement
                current_intensity *= np.exp(-0.1 * dt_step)
            
            return t_remaining


# ----------------------------------------------------═══════════════════════════
# 3. PREPARATION DES DONNÉES & ÉVALUATION
# ----------------------------------------------------═══════════════════════════

def prepare_hawkes_sessions(df: pd.DataFrame):
    """
    Prépare les données pour les modèles de Hawkes.
    Retourne une liste de dictionnaires {times, features, last_cg_offset}.
    """
    sessions = []
    
    for (airport, alert_id), group in df.groupby(["airport", "airport_alert_id"]):
        group = group.sort_values("date")
        
        # Tous les éclairs CG de la session
        cg = group[~group["icloud"]].copy()
        if len(cg) < 2:
            continue
        
        start = group["date"].iloc[0]
        last_cg = cg["date"].iloc[-1]
        
        # Timestamps relatifs en minutes
        times = (group["date"] - start).dt.total_seconds().values / 60.0
        cg_times = (cg["date"] - start).dt.total_seconds().values / 60.0
        
        # Features par événement : [icloud, amplitude_norm, dist_norm, maxis_norm]
        features = np.column_stack([
            group["icloud"].astype(float).values,
            group["amplitude"].values / 100.0,  # normaliser
            group["dist"].values / 50.0,
            group["maxis"].values / 10.0,
        ])
        
        T = (last_cg - start).total_seconds() / 60.0
        
        sessions.append({
            "airport": airport,
            "alert_id": alert_id,
            "times": times,
            "cg_times": cg_times,
            "features": features,
            "T": T,
            "session_key": f"{airport}_{alert_id}",
        })
    
    return sessions


def evaluate_hawkes_models(sessions, test_ratio=0.2):
    """Entraîne et évalue les deux modèles de Hawkes."""
    
    # Split par session
    np.random.seed(42)
    n = len(sessions)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_ratio))
    train_sessions = [sessions[i] for i in idx[:split]]
    test_sessions = [sessions[i] for i in idx[split:]]
    
    print(f"Train: {len(train_sessions)} sessions | Test: {len(test_sessions)} sessions")
    
    # ─── Hawkes Classique ──────────────────
    print("\n=== Hawkes Classique ===")
    hawkes = HawkesClassique()
    
    # On utilise uniquement les inter-arrivées CG pour calibrer
    train_cg_times = [s["cg_times"] for s in train_sessions]
    train_T = [s["T"] for s in train_sessions]
    
    hawkes.fit(train_cg_times, train_T)
    
    # Évaluation : pour chaque éclair CG dans le test set, prédire le temps restant
    hawkes_errors = []
    for session in test_sessions:
        cg_times = session["cg_times"]
        T = session["T"]
        for i in range(len(cg_times) - 1):
            past = cg_times[:i+1]
            true_remaining = T - cg_times[i]
            pred_remaining = hawkes.predict_end(past)
            hawkes_errors.append({"true": true_remaining, "pred": pred_remaining})
    
    hawkes_df = pd.DataFrame(hawkes_errors)
    if len(hawkes_df) > 0:
        mae_h = np.mean(np.abs(hawkes_df["true"] - hawkes_df["pred"]))
        rmse_h = np.sqrt(np.mean((hawkes_df["true"] - hawkes_df["pred"])**2))
        med_h = np.median(np.abs(hawkes_df["true"] - hawkes_df["pred"]))
        bias_h = np.mean(hawkes_df["pred"] - hawkes_df["true"])
        p90_h = np.percentile(np.abs(hawkes_df["true"] - hawkes_df["pred"]), 90)
        
        print(f"\n{'='*50}")
        print(f"  Hawkes Classique")
        print(f"{'='*50}")
        print(f"  MAE :        {mae_h:.2f} min")
        print(f"  RMSE :       {rmse_h:.2f} min")
        print(f"  Mediane :    {med_h:.2f} min")
        print(f"  P90 :        {p90_h:.2f} min")
        print(f"  Biais :      {bias_h:+.2f} min")
        print(f"{'='*50}")
    
    # ─── Neural Hawkes ─────────────────────
    print("\n=== Neural Hawkes Process (GRU) ===")
    neural_hawkes = NeuralHawkes(input_dim=4, hidden_dim=32, lr=1e-3)
    neural_hawkes.fit(train_sessions, n_epochs=30, batch_size=32)
    
    # Évaluation
    neural_errors = []
    for session in test_sessions:
        times = session["times"]
        features = session["features"]
        T = session["T"]
        
        # Prédire à différents points dans la session
        for frac in [0.3, 0.5, 0.7, 0.9]:
            idx = int(len(times) * frac)
            if idx < 2:
                continue
            true_remaining = T - times[idx]
            if true_remaining < 0:
                continue
            pred_remaining = neural_hawkes.predict_time_remaining(
                features[:idx+1], times[:idx+1], threshold=0.05
            )
            neural_errors.append({"true": true_remaining, "pred": pred_remaining})
    
    neural_df = pd.DataFrame(neural_errors)
    if len(neural_df) > 0:
        mae_n = np.mean(np.abs(neural_df["true"] - neural_df["pred"]))
        rmse_n = np.sqrt(np.mean((neural_df["true"] - neural_df["pred"])**2))
        med_n = np.median(np.abs(neural_df["true"] - neural_df["pred"]))
        bias_n = np.mean(neural_df["pred"] - neural_df["true"])
        p90_n = np.percentile(np.abs(neural_df["true"] - neural_df["pred"]), 90)
        
        print(f"\n{'='*50}")
        print(f"  Neural Hawkes (GRU)")
        print(f"{'='*50}")
        print(f"  MAE :        {mae_n:.2f} min")
        print(f"  RMSE :       {rmse_n:.2f} min")
        print(f"  Mediane :    {med_n:.2f} min")
        print(f"  P90 :        {p90_n:.2f} min")
        print(f"  Biais :      {bias_n:+.2f} min")
        print(f"{'='*50}")
    
    return hawkes, neural_hawkes


if __name__ == "__main__":
    print("Loading data...")
    df = load_raw()
    alerts = load_alerts(df)
    
    print("Preparing Hawkes sessions...")
    sessions = prepare_hawkes_sessions(alerts)
    print(f"  {len(sessions)} sessions preparees")
    
    hawkes, neural_hawkes = evaluate_hawkes_models(sessions)
    
    # Sauvegarde des paramètres du Hawkes classique
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    
    params = {"mu": hawkes.mu, "alpha": hawkes.alpha, "beta": hawkes.beta}
    pd.Series(params).to_json(str(model_dir / "hawkes_classique_params.json"))
    
    # Sauvegarde du modèle Neural Hawkes
    torch.save(neural_hawkes.model.state_dict(), str(model_dir / "neural_hawkes_gru.pt"))
    
    print(f"\nModeles sauvegardes dans {model_dir}/")
