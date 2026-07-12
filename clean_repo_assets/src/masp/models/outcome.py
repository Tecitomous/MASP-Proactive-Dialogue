"""
Outcome model G_κ (paper §A.2) — P4G donation prediction.

Per-head architecture:
    text  =  "[PROFILE] {p_u}\n[HISTORY] {h_T}"
    h     =  φ(text)                        # frozen SentenceEncoder backbone
    body  =  MLP(d → hidden → hidden)
    p     =  sigmoid(Linear(hidden → 1))    # P(donate > 0)
    a     =  sigmoid(Linear(hidden → 1))    # normalized donation amount in [0, 1]

K-ensemble (paper eq 33-35):
    p̄ = mean_k p^(k);  ā = mean_k a^(k);  σ² = var_k a^(k)
    R^P4G_out = λ_out · (w_p · p̄ + w_a · ā - w_σ · σ)              (eq 37)

Used as a TERMINAL reward bump on the system in self-play (paper eq 28).
The σ-term penalizes ensemble disagreement, discouraging reward-model
exploitation in out-of-distribution self-play regions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from .sentence_encoder import SentenceEncoder, SentenceEncoderConfig


@dataclass
class OutcomeHeadConfig:
    hidden: int = 512
    dropout: float = 0.1


def _format_outcome_text(history_text: str, profile_text: str) -> str:
    return (
        f"[USER PROFILE]\n{profile_text or '(none)'}\n\n"
        f"[FULL DIALOGUE HISTORY]\n{history_text or '(empty)'}"
    )


class P4GOutcomeHead(nn.Module):
    """One outcome head. Frozen SentenceEncoder backbone + trainable MLP."""

    def __init__(self, encoder: SentenceEncoder, cfg: OutcomeHeadConfig):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        d = int(encoder.hidden_size)
        h = int(cfg.hidden)
        self.body = nn.Sequential(
            nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(cfg.dropout),
        )
        self.cls = nn.Linear(h, 1)
        self.reg = nn.Linear(h, 1)

    def trainable_parameters(self):
        for m in (self.body, self.cls, self.reg):
            for p in m.parameters():
                yield p

    def forward(self, history_text: Sequence[str], profile_text: Sequence[str]):
        if len(history_text) != len(profile_text):
            raise ValueError("history_text and profile_text must align")
        texts = [_format_outcome_text(h, p) for h, p in zip(history_text, profile_text)]
        with torch.no_grad():
            h = self.encoder.encode(texts)            # (B, d) frozen
        x = self.body(h)
        p_donate = torch.sigmoid(self.cls(x)).squeeze(-1)        # (B,)
        amount_norm = torch.sigmoid(self.reg(x)).squeeze(-1)     # (B,)
        return {"p_donate": p_donate, "amount_norm": amount_norm}

    @staticmethod
    def loss(pred, target, mask=None,
             bce_weight: float = 1.0, mse_weight: float = 1.0) -> "torch.Tensor":
        """target = {"donated": (B,), "amount_norm": (B,)}."""
        m = mask if mask is not None else torch.ones_like(target["donated"])
        bce = F.binary_cross_entropy(pred["p_donate"], target["donated"], reduction="none")
        mse = F.mse_loss(pred["amount_norm"], target["amount_norm"], reduction="none")
        loss = bce_weight * bce + mse_weight * mse
        return (loss * m).sum() / m.sum().clamp(min=1.0)

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "cfg": {"hidden": int(self.cfg.hidden), "dropout": float(self.cfg.dropout)},
            "body": self.body.state_dict(),
            "cls": self.cls.state_dict(),
            "reg": self.reg.state_dict(),
        }, path)

    def load(self, path: str, map_location: str = "cpu") -> None:
        sd = torch.load(path, map_location=map_location)
        self.body.load_state_dict(sd["body"])
        self.cls.load_state_dict(sd["cls"])
        self.reg.load_state_dict(sd["reg"])


# ============================================================================
# Ensemble (paper eq 33-35)
# ============================================================================

@dataclass
class OutcomeRewardConfig:
    """Paper eq 37 weights."""
    lambda_out: float = 1.0
    w_p: float = 0.5
    w_a: float = 1.0
    w_sigma: float = 0.2


class OutcomeEnsemble:
    """
    K independent P4GOutcomeHead checkpoints. predict() runs all K, then
    returns aggregated (p̄, ā, σ) per paper eq 33-35.
    """

    def __init__(self, heads: List[P4GOutcomeHead]):
        if not heads:
            raise ValueError("OutcomeEnsemble needs ≥ 1 head")
        self.heads = heads
        for h in self.heads:
            h.eval()
            for p in h.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def predict(self, history_text: Sequence[str], profile_text: Sequence[str]):
        """Returns dict with keys {p_bar, a_bar, sigma}, each shape (B,)."""
        ps, as_ = [], []
        for head in self.heads:
            out = head(history_text, profile_text)
            ps.append(out["p_donate"])
            as_.append(out["amount_norm"])
        p_stack = torch.stack(ps, dim=0)     # (K, B)
        a_stack = torch.stack(as_, dim=0)
        p_bar = p_stack.mean(dim=0)          # (B,)
        a_bar = a_stack.mean(dim=0)
        sigma = a_stack.std(dim=0, unbiased=False) if len(self.heads) > 1 else torch.zeros_like(a_bar)
        return {"p_bar": p_bar, "a_bar": a_bar, "sigma": sigma}

    @torch.no_grad()
    def reward(self, history_text: str, profile_text: str, cfg: OutcomeRewardConfig) -> float:
        """
        Paper eq 37 — terminal scalar reward for one episode.
            R = λ_out · (w_p · p̄ + w_a · ā - w_σ · σ)
        """
        agg = self.predict([history_text], [profile_text])
        p = float(agg["p_bar"].item())
        a = float(agg["a_bar"].item())
        s = float(agg["sigma"].item())
        return float(cfg.lambda_out * (cfg.w_p * p + cfg.w_a * a - cfg.w_sigma * s))

    @classmethod
    def from_dir(cls, ensemble_dir: str, encoder: SentenceEncoder, head_cfg: OutcomeHeadConfig) -> "OutcomeEnsemble":
        """
        Load K heads from {ensemble_dir}/head_{seed}.pt files.
        """
        import os, glob
        paths = sorted(glob.glob(os.path.join(ensemble_dir, "head_*.pt")))
        if not paths:
            raise FileNotFoundError(f"no head_*.pt under {ensemble_dir}")
        heads = []
        for p in paths:
            head = P4GOutcomeHead(encoder, head_cfg).to(encoder.device)
            head.load(p, map_location=str(encoder.device))
            heads.append(head)
        return cls(heads)
