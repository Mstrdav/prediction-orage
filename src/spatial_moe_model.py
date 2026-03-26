"""
spatial_moe_model.py - Phase 2ter : Spatial Mixture of Experts.

Architecture MoE avec 3 experts spécialisés :
  - Expert A (Approaching) : orage qui s'approche
  - Expert B (Receding)    : orage qui s'éloigne → meilleure prédiction de fin
  - Expert C (Stationary)  : situation intermédiaire

Le gating network utilise les features spatiales pour router les prédictions.
Sortie gaussienne (mu, sigma) en log-space, comme neural_hawkes_v3.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import math
import json
from pathlib import Path
from data_loader import load_raw, load_alerts
from neural_hawkes_v2 import ContinuousPositionalEncoding, DEVICE, MODEL_DIR
from spatial_features import prepare_sessions_spatial
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------------------------------═══════════════════════════
# 1. DATASET
# ----------------------------------------------------═══════════════════════════

class SpatialMoEDataset(Dataset):
    """Dataset avec 20 features spatiales et log-target."""
    def __init__(self, sessions, max_len=200):
        self.sessions = []
        self.max_len = max_len
        self.feat_dim = None
        
        for s in sessions:
            if len(s["times"]) < 3:
                continue
            n = min(len(s["times"]), max_len)
            self.sessions.append({
                "times": s["times"][:n],
                "features": s["features"][:n],
                "time_to_end": s["time_to_end"][:n],
                "n": n,
                "airport": s["airport"]
            })
            if self.feat_dim is None:
                self.feat_dim = s["features"].shape[1]
    
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        s = self.sessions[idx]
        n = s["n"]
        
        times = np.zeros(self.max_len)
        features = np.zeros((self.max_len, self.feat_dim))
        time_to_end = np.zeros(self.max_len)
        log_tte = np.zeros(self.max_len)
        mask = np.zeros(self.max_len)
        
        times[:n] = s["times"]
        features[:n] = s["features"]
        time_to_end[:n] = s["time_to_end"]
        log_tte[:n] = np.log1p(s["time_to_end"])
        mask[:n] = 1.0
        
        dt = np.zeros(self.max_len)
        dt[1:n] = np.diff(times[:n])
        
        padding_mask = np.ones(self.max_len, dtype=bool)
        padding_mask[:n] = False
        
        return {
            "times": torch.FloatTensor(times),
            "features": torch.FloatTensor(features),
            "delta_times": torch.FloatTensor(dt).unsqueeze(-1),
            "time_to_end": torch.FloatTensor(time_to_end),
            "log_tte": torch.FloatTensor(log_tte),
            "mask": torch.FloatTensor(mask),
            "padding_mask": torch.BoolTensor(padding_mask),
            "n": n
        }


# ----------------------------------------------------═══════════════════════════
# 2. EXPERT MODULE
# ----------------------------------------------------═══════════════════════════

class ExpertHead(nn.Module):
    """Un expert gaussien : prédit (mu, log_sigma) depuis les hidden states."""
    def __init__(self, d_model, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(d_model // 2, 1)
        self.log_sigma_head = nn.Linear(d_model // 2, 1)
    
    def forward(self, h):
        z = self.net(h)
        mu = self.mu_head(z).squeeze(-1)
        log_sigma = torch.clamp(self.log_sigma_head(z).squeeze(-1), min=-4, max=3)
        return mu, log_sigma


# ----------------------------------------------------═══════════════════════════
# 3. SPATIAL MIXTURE OF EXPERTS
# ----------------------------------------------------═══════════════════════════

class SpatialMoETransformer(nn.Module):
    """
    Mixture of Experts Transformer avec gating spatial.
    
    3 experts : Approaching (A), Receding (B), Stationary (C).
    Le gating utilise les hidden states + features spatiales clés.
    """
    N_EXPERTS = 3
    
    # Indices des features spatiales clés pour le gating
    # 2=dist, 11=dist_trend, 12=radial_vel, 14=5km_ratio, 15=retreat, 16=receding_dur
    SPATIAL_FEAT_INDICES = [2, 11, 12, 14, 15, 16]
    
    def __init__(self, input_dim=20, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=128, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Shared encoder
        self.feature_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = ContinuousPositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Intensity head (unchanged from v3)
        self.intensity_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        self.softplus = nn.Softplus()
        
        # 3 expert heads
        self.experts = nn.ModuleList([
            ExpertHead(d_model, dropout=dropout) for _ in range(self.N_EXPERTS)
        ])
        
        # Gating network : utilise hidden_states + features spatiales
        n_spatial = len(self.SPATIAL_FEAT_INDICES)
        self.gating = nn.Sequential(
            nn.Linear(d_model + n_spatial, d_model // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.N_EXPERTS),
        )
        
        self.dropout_input = nn.Dropout(dropout)
    
    def generate_causal_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    
    def forward(self, event_features, timestamps, delta_times, padding_mask=None):
        B, S, _ = event_features.shape
        
        # Shared encoder
        x = self.feature_proj(event_features)
        pe = self.pos_encoder(timestamps)
        x = self.dropout_input(x + pe)
        
        causal_mask = self.generate_causal_mask(S).to(x.device)
        h = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        # Intensity
        combined = torch.cat([h, delta_times], dim=-1)
        intensity = self.softplus(self.intensity_head(combined)).squeeze(-1)
        
        # Extract spatial features for gating
        spatial_feats = event_features[:, :, self.SPATIAL_FEAT_INDICES]  # (B, S, n_spatial)
        gate_input = torch.cat([h, spatial_feats], dim=-1)  # (B, S, d_model + n_spatial)
        gate_logits = self.gating(gate_input)  # (B, S, N_EXPERTS)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # (B, S, N_EXPERTS)
        
        # Expert predictions
        expert_mus = []
        expert_log_sigmas = []
        for expert in self.experts:
            mu_e, ls_e = expert(h)
            expert_mus.append(mu_e)
            expert_log_sigmas.append(ls_e)
        
        # Stack: (B, S, N_EXPERTS)
        expert_mus = torch.stack(expert_mus, dim=-1)
        expert_log_sigmas = torch.stack(expert_log_sigmas, dim=-1)
        
        # Weighted mixture of Gaussians (moment matching)
        # mu_mix = sum(w_k * mu_k)
        mu = (gate_weights * expert_mus).sum(dim=-1)  # (B, S)
        
        # sigma_mix² = sum(w_k * (sigma_k² + mu_k²)) - mu_mix²
        expert_sigmas = torch.exp(expert_log_sigmas)
        var_mix = (gate_weights * (expert_sigmas**2 + expert_mus**2)).sum(dim=-1) - mu**2
        log_sigma = 0.5 * torch.log(var_mix + 1e-8)
        log_sigma = torch.clamp(log_sigma, min=-4, max=3)
        
        return intensity, mu, log_sigma, h, gate_weights


# ----------------------------------------------------═══════════════════════════
# 4. TRAINER
# ----------------------------------------------------═══════════════════════════

class SpatialMoETrainer:
    """Trainer pour le modèle MoE spatial avec load balancing."""
    
    def __init__(self, model, lr=5e-4, reg_weight=0.5, balance_weight=0.1, device=DEVICE):
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=80, eta_min=1e-6
        )
        self.device = device
        self.reg_weight = reg_weight
        self.balance_weight = balance_weight
    
    def gaussian_nll(self, mu, log_sigma, target, mask):
        sigma = torch.exp(log_sigma)
        nll = 0.5 * ((target - mu)**2 / (sigma**2 + 1e-8) + 2 * log_sigma + math.log(2 * math.pi))
        return (nll * mask).sum() / (mask.sum() + 1e-8)
    
    def load_balance_loss(self, gate_weights, mask):
        """Encourage une utilisation équilibrée des experts."""
        # gate_weights: (B, S, N_EXPERTS), mask: (B, S)
        mask_exp = mask.unsqueeze(-1)  # (B, S, 1)
        # Fraction du trafic par expert
        f = (gate_weights * mask_exp).sum(dim=[0, 1]) / (mask.sum() + 1e-8)  # (N_EXPERTS,)
        # Idéal : 1/N pour chaque expert
        n_experts = gate_weights.shape[-1]
        return n_experts * (f * f).sum()  # penalise la concentration
    
    def combined_loss(self, intensity, mu, log_sigma, gate_weights, delta_times, log_tte, mask):
        # NLL processus ponctuel
        log_lambda = torch.log(intensity + 1e-8) * mask
        compensator = intensity * delta_times.squeeze(-1) * mask
        pp_nll = (-log_lambda.sum() + compensator.sum()) / (mask.sum() + 1e-8)
        
        # Gaussian NLL
        g_nll = self.gaussian_nll(mu, log_sigma, log_tte, mask)
        
        # Load balancing
        lb = self.load_balance_loss(gate_weights, mask)
        
        total = pp_nll + self.reg_weight * g_nll + self.balance_weight * lb
        return total, pp_nll.item(), g_nll.item(), lb.item()
    
    def fit(self, train_sessions, n_epochs=80, batch_size=32, verbose=True):
        dataset = SpatialMoEDataset(train_sessions)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=True)
        
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(n_epochs):
            total_pp = total_g = total_lb = 0
            n_batches = 0
            
            for batch in loader:
                features = batch["features"].to(self.device)
                times = batch["times"].to(self.device)
                delta_times = batch["delta_times"].to(self.device)
                log_tte = batch["log_tte"].to(self.device)
                mask = batch["mask"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                
                intensity, mu, log_sigma, _, gate_weights = self.model(
                    features, times, delta_times, padding_mask
                )
                loss, pp_val, g_val, lb_val = self.combined_loss(
                    intensity, mu, log_sigma, gate_weights, delta_times, log_tte, mask
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_pp += pp_val
                total_g += g_val
                total_lb += lb_val
                n_batches += 1
            
            avg_g = total_g / n_batches
            self.scheduler.step()
            
            if avg_g < best_loss - 1e-4:
                best_loss = avg_g
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_pp = total_pp / n_batches
                avg_lb = total_lb / n_batches
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}/{n_epochs} | PP: {avg_pp:.4f} | G-NLL: {avg_g:.4f} | LB: {avg_lb:.4f} | LR: {lr:.2e}")
            
            if patience_counter >= 20:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
        
        if best_state:
            self.model.load_state_dict(best_state)
        return self
    
    def predict_time_remaining(self, session_features, session_times):
        """Prédiction ponctuelle : exp(mu) - 1."""
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
            _, mu, _, _, _ = self.model(features, times_t, delta_times, padding_mask)
            return max(0.0, np.expm1(mu[0, -1].item()))
    
    def predict_with_uncertainty(self, session_features, session_times, n_mc=50):
        """Prédiction avec incertitude via MC Dropout."""
        self.model.train()  # dropout actif
        mu_samples = []
        sigma_samples = []
        
        with torch.no_grad():
            n = len(session_times)
            if n < 2:
                return 30.0, 15.0, [30.0] * n_mc
            
            features = torch.FloatTensor(session_features).unsqueeze(0).to(self.device)
            times_t = torch.FloatTensor(session_times).unsqueeze(0).to(self.device)
            dt = np.zeros(n); dt[1:] = np.diff(session_times)
            delta_times = torch.FloatTensor(dt).unsqueeze(0).unsqueeze(-1).to(self.device)
            padding_mask = torch.zeros(1, n, dtype=torch.bool, device=self.device)
            
            for _ in range(n_mc):
                _, mu, log_sigma, _, _ = self.model(
                    features, times_t, delta_times, padding_mask
                )
                mu_val = mu[0, -1].item()
                sigma_val = np.exp(log_sigma[0, -1].item())
                mu_samples.append(mu_val)
                sigma_samples.append(sigma_val)
        
        mu_arr = np.array(mu_samples)
        sigma_arr = np.array(sigma_samples)
        preds_minutes = np.maximum(np.expm1(mu_arr), 0)
        
        mean_pred = preds_minutes.mean()
        epistemic = preds_minutes.std()
        aleatoric = np.mean(sigma_arr) * mean_pred
        total_std = np.sqrt(epistemic**2 + aleatoric**2)
        
        return mean_pred, total_std, preds_minutes.tolist()
    
    def predict_with_gate_info(self, session_features, session_times, n_mc=30):
        """Prédiction avec incertitude + info de gating (quel expert domine)."""
        self.model.train()
        mu_samples = []
        sigma_samples = []
        gate_samples = []
        
        with torch.no_grad():
            n = len(session_times)
            if n < 2:
                return 30.0, 15.0, [1/3, 1/3, 1/3]
            
            features = torch.FloatTensor(session_features).unsqueeze(0).to(self.device)
            times_t = torch.FloatTensor(session_times).unsqueeze(0).to(self.device)
            dt = np.zeros(n); dt[1:] = np.diff(session_times)
            delta_times = torch.FloatTensor(dt).unsqueeze(0).unsqueeze(-1).to(self.device)
            padding_mask = torch.zeros(1, n, dtype=torch.bool, device=self.device)
            
            for _ in range(n_mc):
                _, mu, log_sigma, _, gate_weights = self.model(
                    features, times_t, delta_times, padding_mask
                )
                mu_samples.append(mu[0, -1].item())
                sigma_samples.append(np.exp(log_sigma[0, -1].item()))
                gate_samples.append(gate_weights[0, -1].cpu().numpy())
        
        mu_arr = np.array(mu_samples)
        sigma_arr = np.array(sigma_samples)
        preds_minutes = np.maximum(np.expm1(mu_arr), 0)
        
        mean_pred = preds_minutes.mean()
        epistemic = preds_minutes.std()
        aleatoric = np.mean(sigma_arr) * mean_pred
        total_std = np.sqrt(epistemic**2 + aleatoric**2)
        
        avg_gates = np.mean(gate_samples, axis=0).tolist()
        
        return mean_pred, total_std, avg_gates


# ----------------------------------------------------═══════════════════════════
# 5. ÉVALUATION
# ----------------------------------------------------═══════════════════════════

def evaluate_spatial_moe(trainer, test_sessions, label="MoE Spatial", with_uncertainty=True, n_mc=50):
    """Évalue le MoE avec focus sur les prédictions à dist < 5km."""
    results = []
    for session in test_sessions:
        times = session["times"]
        features = session["features"]
        T = session["T"]
        airport = session["airport"]
        
        # Distance brute (feature index 2, dénormalisée)
        dists_raw = features[:, 2] * 50.0
        
        for frac in [0.3, 0.5, 0.7, 0.9]:
            idx = int(len(times) * frac)
            if idx < 2:
                continue
            true_remaining = T - times[idx]
            if true_remaining < 0:
                continue
            
            # Distance courante
            current_dist = dists_raw[idx]
            # Ratio 5km dans la fenêtre récente
            start_idx = max(0, idx - 9)
            ratio_5km = np.mean(dists_raw[start_idx:idx+1] < 5.0)
            
            if with_uncertainty:
                mean_pred, std_pred, _ = trainer.predict_with_uncertainty(
                    features[:idx+1], times[:idx+1], n_mc=n_mc
                )
                results.append({
                    "true": true_remaining, "pred": mean_pred,
                    "uncertainty": std_pred,
                    "airport": airport, "frac": frac,
                    "current_dist": current_dist,
                    "ratio_5km": ratio_5km,
                    "is_near_5km": current_dist < 5.0,
                })
            else:
                pred = trainer.predict_time_remaining(features[:idx+1], times[:idx+1])
                results.append({
                    "true": true_remaining, "pred": pred,
                    "airport": airport, "frac": frac,
                    "current_dist": current_dist,
                    "ratio_5km": ratio_5km,
                    "is_near_5km": current_dist < 5.0,
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
    
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  MAE :     {mae:.2f} min")
    print(f"  RMSE :    {rmse:.2f} min")
    print(f"  Médiane : {median:.2f} min")
    print(f"  P90 :     {p90:.2f} min")
    print(f"  Biais :   {bias:+.2f} min")
    
    if with_uncertainty and "uncertainty" in df.columns:
        mean_unc = df["uncertainty"].mean()
        in_1std = np.mean(errors <= df["uncertainty"])
        in_2std = np.mean(errors <= 2 * df["uncertainty"])
        print(f"  σ moyen : {mean_unc:.2f} min")
        print(f"  Cal. 1σ : {in_1std*100:.1f}% (idéal 68.3%)")
        print(f"  Cal. 2σ : {in_2std*100:.1f}% (idéal 95.4%)")
    
    # Focus 5km
    df_5km = df[df["is_near_5km"]]
    if len(df_5km) > 10:
        e5 = np.abs(df_5km["true"] - df_5km["pred"])
        mae_5km = e5.mean()
        bias_5km = np.mean(df_5km["pred"] - df_5km["true"])
        print(f"\n  ── Focus rayon 5km (n={len(df_5km)}) ──")
        print(f"  MAE 5km :  {mae_5km:.2f} min")
        print(f"  Biais 5km : {bias_5km:+.2f} min")
    
    # Analyse fin d'alerte (frac ≥ 0.7)
    df_end = df[df["frac"] >= 0.7]
    if len(df_end) > 5:
        e_end = np.abs(df_end["true"] - df_end["pred"])
        mae_end = e_end.mean()
        bias_end = np.mean(df_end["pred"] - df_end["true"])
        print(f"\n  ── Fin d'alerte (frac≥0.7, n={len(df_end)}) ──")
        print(f"  MAE fin :  {mae_end:.2f} min")
        print(f"  Biais fin : {bias_end:+.2f} min")
    
    print(f"{'='*65}")
    
    print(f"\n  Par aéroport :")
    airport_metrics = {}
    for ap, g in df.groupby("airport"):
        ap_mae = np.mean(np.abs(g["true"] - g["pred"]))
        ap_bias = np.mean(g["pred"] - g["true"])
        extra = ""
        if with_uncertainty and "uncertainty" in g.columns:
            extra = f" | σ={g['uncertainty'].mean():.2f}"
        print(f"    {ap:12s} | MAE={ap_mae:.2f} | Biais={ap_bias:+.2f}{extra} | n={len(g)}")
        airport_metrics[ap] = {"mae": ap_mae, "bias": ap_bias, "n": len(g)}
    
    result = {
        "label": label, "mae": mae, "rmse": rmse, "median": median,
        "bias": bias, "p90": p90, "errors_df": df,
        "airport_metrics": airport_metrics
    }
    if with_uncertainty and "uncertainty" in df.columns:
        result["mean_uncertainty"] = mean_unc
        result["calibration_1std"] = in_1std
        result["calibration_2std"] = in_2std
    
    return result


def run_moe_experiment(sessions, test_ratio=0.2):
    """Entraîne et évalue le modèle MoE Spatial."""
    
    np.random.seed(42)
    n = len(sessions)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_ratio))
    train_sessions = [sessions[i] for i in idx[:split]]
    test_sessions = [sessions[i] for i in idx[split:]]
    
    print(f"Train: {len(train_sessions)} | Test: {len(test_sessions)}\n")
    
    feat_dim = sessions[0]["features"].shape[1]
    print(f"Feature dimension: {feat_dim}\n")
    
    # ─── MoE Spatial ──────────────────────────────────────────────────
    print("=" * 65)
    print("  Phase 2ter : Spatial MoE Transformer")
    print("=" * 65)
    model = SpatialMoETransformer(
        input_dim=feat_dim, d_model=64, nhead=4, num_layers=3,
        dim_feedforward=128, dropout=0.15
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres : {n_params:,}")
    
    trainer = SpatialMoETrainer(model, lr=5e-4, reg_weight=0.5, balance_weight=0.1)
    trainer.fit(train_sessions, n_epochs=80, batch_size=32)
    
    result = evaluate_spatial_moe(trainer, test_sessions, "MoE Spatial (3 experts)", 
                                   with_uncertainty=True, n_mc=50)
    
    # Save model
    torch.save(model.state_dict(), str(MODEL_DIR / "spatial_moe.pt"))
    print(f"\n  Modèle sauvegardé : {MODEL_DIR / 'spatial_moe.pt'}")
    
    # Save results
    if result:
        summary = {k: v for k, v in result.items() if k not in ("errors_df", "airport_metrics")}
        with open(str(MODEL_DIR / "phase2ter_results.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        result["errors_df"].to_parquet(
            str(Path(__file__).parent.parent / "data" / "errors_moe_spatial.parquet"), index=False
        )
    
    # Comparison with v3 results
    v3_path = MODEL_DIR / "phase2bis_v3_results.json"
    if v3_path.exists():
        with open(str(v3_path)) as f:
            v3 = json.load(f)
        
        print(f"\n\n{'='*75}")
        print(f"  COMPARAISON Phase 2bis vs Phase 2ter")
        print(f"{'='*75}")
        print(f"  {'Modèle':40s} | {'MAE':>6s} | {'RMSE':>6s} | {'Biais':>7s} | {'P90':>6s}")
        print("-" * 75)
        
        for key in ["gru_gaussian", "transformer_gaussian", "transformer_large_gaussian"]:
            if key in v3:
                r = v3[key]
                lbl = r.get("label", key)[:40]
                print(f"  {lbl:40s} | {float(r['mae']):6.2f} | {float(r['rmse']):6.2f} | {float(r['bias']):+7.2f} | {float(r['p90']):6.2f}")
        
        if result:
            print("-" * 75)
            print(f"  {'MoE Spatial (Phase 2ter)':40s} | {result['mae']:6.2f} | {result['rmse']:6.2f} | {result['bias']:+7.2f} | {result['p90']:6.2f}")
        
        print("=" * 75)
    
    return result, trainer


if __name__ == "__main__":
    print("=" * 65)
    print("  PHASE 2ter - Spatial Mixture of Experts")
    print("=" * 65)
    
    print("\nLoading data...")
    df = load_raw()
    alerts = load_alerts(df)
    
    print("Preparing spatial sessions (20 features)...")
    sessions = prepare_sessions_spatial(alerts)
    print(f"  {len(sessions)} sessions")
    print(f"  Features : {sessions[0]['features'].shape[1]}")
    
    result, trainer = run_moe_experiment(sessions)
    
    print(f"\nTerminé !")
