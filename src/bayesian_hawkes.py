"""
bayesian_hawkes.py – Bayesian Neural Hawkes Process (multi-tâche).

Deux approches :
  Option A : MC Dropout sur le Transformer Hawkes
  Option B : Couches variationnelles (ELBO)

Les deux utilisent la tête de régression directe (pas d'extrapolation).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import math
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader import load_raw, load_alerts
from neural_hawkes_v2 import (
    prepare_sessions_v2, HawkesDatasetV2,
    ContinuousPositionalEncoding,
    DEVICE, MODEL_DIR
)

# ----------------------------------------------------═══════════════════════════
# OPTION A : MC DROPOUT TRANSFORMER (multi-tâche)
# ----------------------------------------------------═══════════════════════════

class MCDropoutHawkesTransformer(nn.Module):
    """
    Transformer Hawkes multi-tâche avec MC Dropout.
    Dropout reste actif pendant l'inférence pour N passages stochastiques.
    """
    def __init__(self, input_dim=12, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        
        self.feature_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = ContinuousPositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.intensity_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        self.softplus = nn.Softplus()
        
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()
        )
        self.dropout_input = nn.Dropout(dropout)
    
    def generate_causal_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    
    def forward(self, event_features, timestamps, delta_times, padding_mask=None):
        B, S, _ = event_features.shape
        x = self.feature_proj(event_features)
        pe = self.pos_encoder(timestamps)
        x = self.dropout_input(x + pe)
        causal_mask = self.generate_causal_mask(S).to(x.device)
        h = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        combined = torch.cat([h, delta_times], dim=-1)
        intensity = self.softplus(self.intensity_head(combined)).squeeze(-1)
        time_pred = self.regression_head(h).squeeze(-1)
        return intensity, time_pred, h


class BayesianHawkesTrainer:
    """Trainer multi-tâche + prédiction avec incertitude MC Dropout."""
    
    def __init__(self, model, lr=5e-4, reg_weight=1.0, device=DEVICE):
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        self.device = device
        self.reg_weight = reg_weight
        self.huber = nn.HuberLoss(delta=20.0, reduction='none')
    
    def combined_loss(self, intensity, time_pred, delta_times, time_to_end, mask):
        log_lambda = torch.log(intensity + 1e-8) * mask
        compensator = intensity * delta_times.squeeze(-1) * mask
        nll = (-log_lambda.sum() + compensator.sum()) / (mask.sum() + 1e-8)
        reg_loss = (self.huber(time_pred, time_to_end) * mask).sum() / (mask.sum() + 1e-8)
        return nll + self.reg_weight * reg_loss, nll.item(), reg_loss.item()
    
    def fit(self, train_sessions, n_epochs=60, batch_size=32, verbose=True):
        dataset = HawkesDatasetV2(train_sessions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=True)
        
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(n_epochs):
            total_nll = 0
            total_reg = 0
            n_batches = 0
            
            for batch in loader:
                features = batch["features"].to(self.device)
                times = batch["times"].to(self.device)
                delta_times = batch["delta_times"].to(self.device)
                time_to_end = batch["time_to_end"].to(self.device)
                mask = batch["mask"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                
                intensity, time_pred, _ = self.model(features, times, delta_times, padding_mask)
                loss, nll_val, reg_val = self.combined_loss(
                    intensity, time_pred, delta_times, time_to_end, mask
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_nll += nll_val
                total_reg += reg_val
                n_batches += 1
            
            avg_reg = total_reg / n_batches
            self.scheduler.step()
            
            if avg_reg < best_loss - 1e-4:
                best_loss = avg_reg
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_nll = total_nll / n_batches
                print(f"  Epoch {epoch+1}/{n_epochs} | NLL: {avg_nll:.4f} | Reg: {avg_reg:.4f}")
            
            if patience_counter >= 15:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self
    
    def predict_time_remaining(self, session_features, session_times):
        """Prédiction ponctuelle (mode eval)."""
        self.model.eval()
        with torch.no_grad():
            n = len(session_times)
            if n < 2:
                return 30.0
            features = torch.FloatTensor(session_features).unsqueeze(0).to(self.device)
            times_t = torch.FloatTensor(session_times).unsqueeze(0).to(self.device)
            dt = np.zeros(n); dt[1:] = np.diff(session_times)
            delta_times = torch.FloatTensor(dt).unsqueeze(0).unsqueeze(-1).to(self.device)
            padding_mask = torch.zeros(1, n, dtype=torch.bool, device=self.device)
            _, time_pred, _ = self.model(features, times_t, delta_times, padding_mask)
            return time_pred[0, -1].item()
    
    def predict_with_uncertainty(self, session_features, session_times, n_samples=50):
        """N passages forward avec dropout actif → (mean, std, predictions)."""
        self.model.train()  # Garder dropout actif !
        predictions = []
        
        with torch.no_grad():
            n = len(session_times)
            if n < 2:
                return 30.0, 15.0, [30.0] * n_samples
            
            features = torch.FloatTensor(session_features).unsqueeze(0).to(self.device)
            times_t = torch.FloatTensor(session_times).unsqueeze(0).to(self.device)
            dt = np.zeros(n); dt[1:] = np.diff(session_times)
            delta_times = torch.FloatTensor(dt).unsqueeze(0).unsqueeze(-1).to(self.device)
            padding_mask = torch.zeros(1, n, dtype=torch.bool, device=self.device)
            
            for _ in range(n_samples):
                _, time_pred, _ = self.model(features, times_t, delta_times, padding_mask)
                predictions.append(time_pred[0, -1].item())
        
        preds = np.array(predictions)
        return preds.mean(), preds.std(), predictions


# ----------------------------------------------------═══════════════════════════
# OPTION B : VARIATIONAL BAYESIAN LAYERS
# ----------------------------------------------------═══════════════════════════

class VariationalLinear(nn.Module):
    """Couche linéaire bayésienne variationnelle avec reparametrization trick."""
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.w_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.w_log_sigma = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_log_sigma = nn.Parameter(torch.full((out_features,), -3.0))
        self.prior_sigma = prior_sigma
        nn.init.kaiming_normal_(self.w_mu, nonlinearity='relu')
    
    def forward(self, x):
        if self.training:
            w = self.w_mu + torch.exp(self.w_log_sigma) * torch.randn_like(self.w_log_sigma)
            b = self.b_mu + torch.exp(self.b_log_sigma) * torch.randn_like(self.b_log_sigma)
        else:
            w, b = self.w_mu, self.b_mu
        return F.linear(x, w, b)
    
    def kl_divergence(self):
        w_sigma = torch.exp(self.w_log_sigma)
        b_sigma = torch.exp(self.b_log_sigma)
        kl_w = 0.5 * ((w_sigma**2 + self.w_mu**2) / self.prior_sigma**2 - 1 
                       + 2 * (math.log(self.prior_sigma) - self.w_log_sigma)).sum()
        kl_b = 0.5 * ((b_sigma**2 + self.b_mu**2) / self.prior_sigma**2 - 1
                       + 2 * (math.log(self.prior_sigma) - self.b_log_sigma)).sum()
        return kl_w + kl_b


class VariationalHawkesTransformer(nn.Module):
    """Transformer avec tête de régression variationnelle."""
    def __init__(self, input_dim=12, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=128, dropout=0.1, prior_sigma=1.0):
        super().__init__()
        self.d_model = d_model
        
        self.feature_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = ContinuousPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout_input = nn.Dropout(dropout)
        
        # Tête intensité standard
        self.intensity_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        self.softplus = nn.Softplus()
        
        # Tête régression variationnelle
        self.var_layer1 = VariationalLinear(d_model, d_model // 2, prior_sigma)
        self.var_layer2 = VariationalLinear(d_model // 2, 1, prior_sigma)
    
    def generate_causal_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    
    def forward(self, event_features, timestamps, delta_times, padding_mask=None):
        B, S, _ = event_features.shape
        x = self.feature_proj(event_features)
        pe = self.pos_encoder(timestamps)
        x = self.dropout_input(x + pe)
        causal_mask = self.generate_causal_mask(S).to(x.device)
        h = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        combined = torch.cat([h, delta_times], dim=-1)
        intensity = self.softplus(self.intensity_head(combined)).squeeze(-1)
        
        time_pred = self.softplus(self.var_layer2(F.gelu(self.var_layer1(h)))).squeeze(-1)
        return intensity, time_pred, h
    
    def kl_divergence(self):
        return self.var_layer1.kl_divergence() + self.var_layer2.kl_divergence()


class VariationalHawkesTrainer:
    """Trainer ELBO = NLL + reg + β * KL."""
    
    def __init__(self, model, lr=5e-4, kl_weight=1e-4, reg_weight=1.0, device=DEVICE):
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=60, eta_min=1e-6
        )
        self.device = device
        self.kl_weight = kl_weight
        self.reg_weight = reg_weight
        self.huber = nn.HuberLoss(delta=20.0, reduction='none')
    
    def elbo_loss(self, intensity, time_pred, delta_times, time_to_end, mask, n_data):
        log_lambda = torch.log(intensity + 1e-8) * mask
        compensator = intensity * delta_times.squeeze(-1) * mask
        nll = (-log_lambda.sum() + compensator.sum()) / (mask.sum() + 1e-8)
        reg_loss = (self.huber(time_pred, time_to_end) * mask).sum() / (mask.sum() + 1e-8)
        kl = self.model.kl_divergence() / n_data
        total = nll + self.reg_weight * reg_loss + self.kl_weight * kl
        return total, nll.item(), reg_loss.item(), kl.item()
    
    def fit(self, train_sessions, n_epochs=60, batch_size=32, verbose=True):
        dataset = HawkesDatasetV2(train_sessions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=True)
        n_data = len(dataset)
        
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(n_epochs):
            total_nll = total_reg = total_kl = 0
            n_batches = 0
            
            for batch in loader:
                features = batch["features"].to(self.device)
                times = batch["times"].to(self.device)
                delta_times = batch["delta_times"].to(self.device)
                time_to_end = batch["time_to_end"].to(self.device)
                mask = batch["mask"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                
                intensity, time_pred, _ = self.model(features, times, delta_times, padding_mask)
                loss, nll_v, reg_v, kl_v = self.elbo_loss(
                    intensity, time_pred, delta_times, time_to_end, mask, n_data
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_nll += nll_v; total_reg += reg_v; total_kl += kl_v
                n_batches += 1
            
            avg_reg = total_reg / n_batches
            self.scheduler.step()
            
            if avg_reg < best_loss - 1e-4:
                best_loss = avg_reg
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs} | NLL: {total_nll/n_batches:.4f} | Reg: {avg_reg:.4f} | KL: {total_kl/n_batches:.4f}")
            
            if patience_counter >= 15:
                if verbose: print(f"  Early stopping at epoch {epoch+1}")
                break
        
        if best_state: self.model.load_state_dict(best_state)
        return self
    
    def predict_time_remaining(self, session_features, session_times):
        self.model.eval()
        with torch.no_grad():
            n = len(session_times)
            if n < 2: return 30.0
            features = torch.FloatTensor(session_features).unsqueeze(0).to(self.device)
            times_t = torch.FloatTensor(session_times).unsqueeze(0).to(self.device)
            dt = np.zeros(n); dt[1:] = np.diff(session_times)
            delta_times = torch.FloatTensor(dt).unsqueeze(0).unsqueeze(-1).to(self.device)
            padding_mask = torch.zeros(1, n, dtype=torch.bool, device=self.device)
            _, time_pred, _ = self.model(features, times_t, delta_times, padding_mask)
            return time_pred[0, -1].item()
    
    def predict_with_uncertainty(self, session_features, session_times, n_samples=50):
        self.model.train()
        predictions = []
        with torch.no_grad():
            n = len(session_times)
            if n < 2: return 30.0, 15.0, [30.0] * n_samples
            features = torch.FloatTensor(session_features).unsqueeze(0).to(self.device)
            times_t = torch.FloatTensor(session_times).unsqueeze(0).to(self.device)
            dt = np.zeros(n); dt[1:] = np.diff(session_times)
            delta_times = torch.FloatTensor(dt).unsqueeze(0).unsqueeze(-1).to(self.device)
            padding_mask = torch.zeros(1, n, dtype=torch.bool, device=self.device)
            for _ in range(n_samples):
                _, time_pred, _ = self.model(features, times_t, delta_times, padding_mask)
                predictions.append(time_pred[0, -1].item())
        preds = np.array(predictions)
        return preds.mean(), preds.std(), predictions


# ----------------------------------------------------═══════════════════════════
# ÉVALUATION BAYÉSIENNE
# ----------------------------------------------------═══════════════════════════

def evaluate_bayesian(trainer, test_sessions, label="Bayesian", n_samples=50):
    """Évalue un modèle bayésien avec incertitude."""
    results = []
    for session in test_sessions:
        times = session["times"]
        features = session["features"]
        T = session["T"]
        airport = session["airport"]
        
        for frac in [0.3, 0.5, 0.7, 0.9]:
            idx = int(len(times) * frac)
            if idx < 2: continue
            true_remaining = T - times[idx]
            if true_remaining < 0: continue
            mean_pred, std_pred, _ = trainer.predict_with_uncertainty(
                features[:idx+1], times[:idx+1], n_samples=n_samples
            )
            results.append({
                "true": true_remaining, "pred": mean_pred,
                "uncertainty": std_pred, "airport": airport, "frac": frac
            })
    
    df = pd.DataFrame(results)
    if len(df) == 0:
        print(f"  {label}: Pas de prédictions")
        return None
    
    errors = np.abs(df["true"] - df["pred"])
    mae = errors.mean()
    rmse = np.sqrt(np.mean((df["true"] - df["pred"])**2))
    median = np.median(errors)
    bias = np.mean(df["pred"] - df["true"])
    p90 = np.percentile(errors, 90)
    mean_unc = df["uncertainty"].mean()
    in_1std = np.mean(errors <= df["uncertainty"])
    in_2std = np.mean(errors <= 2 * df["uncertainty"])
    
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  MAE :       {mae:.2f} min")
    print(f"  RMSE :      {rmse:.2f} min")
    print(f"  Médiane :   {median:.2f} min")
    print(f"  P90 :       {p90:.2f} min")
    print(f"  Biais :     {bias:+.2f} min")
    print(f"  Incert. σ : {mean_unc:.2f} min")
    print(f"  Calibration :")
    print(f"    % dans 1σ : {in_1std*100:.1f}% (idéal 68.3%)")
    print(f"    % dans 2σ : {in_2std*100:.1f}% (idéal 95.4%)")
    print(f"{'='*55}")
    
    print(f"\n  Par aéroport :")
    airport_metrics = {}
    for ap, g in df.groupby("airport"):
        ap_mae = np.mean(np.abs(g["true"] - g["pred"]))
        ap_unc = g["uncertainty"].mean()
        print(f"    {ap:12s} | MAE={ap_mae:.2f} | σ={ap_unc:.2f} | n={len(g)}")
        airport_metrics[ap] = {"mae": ap_mae, "uncertainty": ap_unc, "n": len(g)}
    
    return {
        "label": label, "mae": mae, "rmse": rmse, "median": median,
        "bias": bias, "p90": p90, "mean_uncertainty": mean_unc,
        "calibration_1std": in_1std, "calibration_2std": in_2std,
        "errors_df": df, "airport_metrics": airport_metrics
    }


def run_bayesian_experiments(sessions, test_ratio=0.2):
    np.random.seed(42)
    n = len(sessions)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_ratio))
    train_sessions = [sessions[i] for i in idx[:split]]
    test_sessions = [sessions[i] for i in idx[split:]]
    
    print(f"Train: {len(train_sessions)} | Test: {len(test_sessions)}\n")
    results = {}
    
    # ─── Option A : MC Dropout ────────────────────────────────────────
    print("=" * 60)
    print("  OPTION A : MC Dropout Transformer (multi-tâche)")
    print("=" * 60)
    model_mc = MCDropoutHawkesTransformer(
        input_dim=12, d_model=64, nhead=4, num_layers=3,
        dim_feedforward=128, dropout=0.2
    )
    trainer_mc = BayesianHawkesTrainer(model_mc, lr=5e-4, reg_weight=1.0)
    trainer_mc.fit(train_sessions, n_epochs=80, batch_size=32)
    r = evaluate_bayesian(trainer_mc, test_sessions, "MC Dropout Transformer", n_samples=50)
    if r: results["mc_dropout"] = r
    torch.save(model_mc.state_dict(), str(MODEL_DIR / "bayesian_hawkes_mc.pt"))
    
    # ─── Option B : Variational ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  OPTION B : Variational Bayesian Transformer")
    print("=" * 60)
    model_var = VariationalHawkesTransformer(
        input_dim=12, d_model=64, nhead=4, num_layers=3,
        dim_feedforward=128, dropout=0.1, prior_sigma=1.0
    )
    trainer_var = VariationalHawkesTrainer(model_var, lr=5e-4, kl_weight=1e-4, reg_weight=1.0)
    trainer_var.fit(train_sessions, n_epochs=80, batch_size=32)
    r = evaluate_bayesian(trainer_var, test_sessions, "Variational Bayesian Transformer", n_samples=50)
    if r: results["variational"] = r
    torch.save(model_var.state_dict(), str(MODEL_DIR / "bayesian_hawkes_var.pt"))
    
    # ─── Résumé ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RÉSUMÉ BAYESIAN NEURAL HAWKES")
    print("=" * 70)
    print(f"  {'Méthode':35s} | {'MAE':>6s} | {'Biais':>7s} | {'σ moy':>6s} | {'1σ':>6s} | {'2σ':>6s}")
    print("-" * 70)
    for key in ["mc_dropout", "variational"]:
        if key in results:
            r = results[key]
            print(f"  {r['label']:35s} | {r['mae']:6.2f} | {r['bias']:+7.2f} | {r['mean_uncertainty']:6.2f} | {r['calibration_1std']*100:5.1f}% | {r['calibration_2std']*100:5.1f}%")
    print("=" * 70)
    
    # Sauvegarde
    for key, r in results.items():
        if "errors_df" in r:
            r["errors_df"].to_parquet(
                str(Path(__file__).parent.parent / "data" / f"errors_bayesian_{key}.parquet"), index=False
            )
    summary = {k: {kk: v for kk, v in r.items() if kk not in ("errors_df","airport_metrics")} for k, r in results.items()}
    with open(str(MODEL_DIR / "bayesian_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    return results, trainer_mc, trainer_var


if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 2bis – Bayesian Neural Hawkes Process")
    print("=" * 60)
    
    print("\nLoading data...")
    df = load_raw()
    alerts = load_alerts(df)
    
    print("Preparing enriched sessions (12 features)...")
    sessions = prepare_sessions_v2(alerts)
    print(f"  {len(sessions)} sessions préparées")
    
    results, trainer_mc, trainer_var = run_bayesian_experiments(sessions)
    
    print(f"\nRésultats sauvegardés dans {MODEL_DIR}/")
    print("Terminé !")
