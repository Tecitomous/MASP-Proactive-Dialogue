"""
Mentalization Module  M_ψ : (dialogue_history) -> ẑ ∈ R^{3d + 3}

Architecture (paper §3.2 / eq 9)
--------------------------------
The latent state z_t = [φ(B); φ(D); φ(I); √α_ρ·ρ; √α_c·c; √α_v·v] has
3d + 3 dims. So M_ψ has six heads on top of a frozen SentenceEncoder
backbone:

    h = encoder(history)                                        # (B, H)
    ẑ_B = Linear(H, d)(h)       — text head (d-dim)
    ẑ_D = Linear(H, d)(h)
    ẑ_I = Linear(H, d)(h)
    ẑ_ρ = sigmoid(Linear(H, 1)(h))  → √α_ρ · ρ̂  (matches encode_bdi scaling)
    ẑ_c = sigmoid(Linear(H, 1)(h))  → √α_c · ĉ
    ẑ_v = tanh   (Linear(H, 1)(h))  → √α_v · v̂
    ẑ   = concat([ẑ_B, ẑ_D, ẑ_I, ẑ_ρ, ẑ_c, ẑ_v], dim=-1)        # (B, 3d + 3)

Loss (Phase 0)
--------------
    L_M = ||ẑ - z*||^2

where z* is obtained by encoding the 6-dim silver BDI labels through the same
frozen sentence encoder + the same √α scaling used at inference (encode_bdi),
so targets and predictions live in the exact same vector space.

During Phase 2 self-play the module is additionally updated via:
    L_SP = E_{self-play}[||ẑ - z_user||^2]
where z_user is the sampled ground-truth BDI carried by the user simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sentence_encoder import SentenceEncoder


@dataclass
class MentalizationConfig:
    hidden_size: int            # = SentenceEncoder.hidden_size
    proj_hidden: int = 512
    dropout: float = 0.1
    # Scalar-side metric weights (paper §3.3, A = diag(1/d·I_d, α_ρ, α_c, α_v)).
    # Mentalizer pre-multiplies its scalar predictions by √α so that the same
    # MSE loss matches encode_bdi targets bit-for-bit.
    alpha_rho: float = 1.0
    alpha_c: float = 1.0
    alpha_v: float = 1.0


class _ProjHead(nn.Module):
    def __init__(self, hidden: int, proj_hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, proj_hidden),
            nn.LayerNorm(proj_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MentalizationModule(nn.Module):
    """
    Belief tracker M_ψ. Wraps a *frozen* SentenceEncoder as the backbone and
    learns only three small projection heads that map the pooled history
    embedding to the three BDI component embeddings.

    The module stores a reference to the encoder but does NOT own its
    parameters — the encoder stays frozen.
    """

    def __init__(self, encoder: SentenceEncoder, cfg: MentalizationConfig):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        d = int(encoder.hidden_size)
        self.head_B = _ProjHead(d, cfg.proj_hidden, d, cfg.dropout)
        self.head_D = _ProjHead(d, cfg.proj_hidden, d, cfg.dropout)
        self.head_I = _ProjHead(d, cfg.proj_hidden, d, cfg.dropout)
        # Three scalar heads. Activation is applied in `forward_from_hidden`.
        self.head_rho = _ProjHead(d, cfg.proj_hidden, 1, cfg.dropout)
        self.head_c   = _ProjHead(d, cfg.proj_hidden, 1, cfg.dropout)
        self.head_v   = _ProjHead(d, cfg.proj_hidden, 1, cfg.dropout)
        # Pre-compute √α factors as buffers so they move with the module.
        import math as _math
        self.register_buffer("_sqrt_alpha_rho", torch.tensor(_math.sqrt(max(cfg.alpha_rho, 0.0))))
        self.register_buffer("_sqrt_alpha_c",   torch.tensor(_math.sqrt(max(cfg.alpha_c,   0.0))))
        self.register_buffer("_sqrt_alpha_v",   torch.tensor(_math.sqrt(max(cfg.alpha_v,   0.0))))

    @property
    def bdi_dim(self) -> int:
        return 3 * int(self.encoder.hidden_size) + 3

    def trainable_parameters(self):
        for head in (self.head_B, self.head_D, self.head_I,
                     self.head_rho, self.head_c, self.head_v):
            for p in head.parameters():
                yield p

    # ------------------------------------------------------------------ forward
    def forward_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, H) pooled history embedding (from SentenceEncoder).
        returns: (B, 3d + 3) BDI prediction matching encode_bdi's geometry:
                 text part is L2-normalized per-component;
                 ρ̂, ĉ pre-multiplied by √α and squashed to [0, √α];
                 v̂ pre-multiplied by √α and squashed to [-√α, √α].
        """
        b = F.normalize(self.head_B(h), p=2, dim=-1)
        d = F.normalize(self.head_D(h), p=2, dim=-1)
        i = F.normalize(self.head_I(h), p=2, dim=-1)
        rho = torch.sigmoid(self.head_rho(h)) * self._sqrt_alpha_rho
        c   = torch.sigmoid(self.head_c(h))   * self._sqrt_alpha_c
        v   = torch.tanh   (self.head_v(h))   * self._sqrt_alpha_v
        return torch.cat([b, d, i, rho, c, v], dim=-1)

    def forward(self, history_texts: Sequence[str]) -> torch.Tensor:
        """Encode a list of history strings, then project through heads."""
        with torch.no_grad():
            h = self.encoder.encode(list(history_texts))  # (B, H)
        return self.forward_from_hidden(h)

    # ------------------------------------------------------------------ losses
    @staticmethod
    def bdi_regression_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """
        Component-wise MSE in normalized embedding space. Because each of the
        three components is individually normalized, this is equivalent (up to
        an additive constant) to 3 - sum-of-cosine-similarities.
        """
        if z_pred.shape != z_target.shape:
            raise ValueError(f"shape mismatch: {z_pred.shape} vs {z_target.shape}")
        return F.mse_loss(z_pred, z_target)

    def save(self, path: str) -> None:
        import os
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        state = {
            "cfg": {
                "hidden_size": int(self.cfg.hidden_size),
                "proj_hidden": int(self.cfg.proj_hidden),
                "dropout": float(self.cfg.dropout),
                "alpha_rho": float(self.cfg.alpha_rho),
                "alpha_c": float(self.cfg.alpha_c),
                "alpha_v": float(self.cfg.alpha_v),
            },
            "head_B": self.head_B.state_dict(),
            "head_D": self.head_D.state_dict(),
            "head_I": self.head_I.state_dict(),
            "head_rho": self.head_rho.state_dict(),
            "head_c":   self.head_c.state_dict(),
            "head_v":   self.head_v.state_dict(),
        }
        torch.save(state, path)

    def load(self, path: str, map_location: str = "cpu") -> None:
        # Our own checkpoints — opt out of the upcoming weights_only=True
        # default (which would refuse the nested dict-of-state_dicts layout).
        sd = torch.load(path, map_location=map_location, weights_only=False)
        self.head_B.load_state_dict(sd["head_B"])
        self.head_D.load_state_dict(sd["head_D"])
        self.head_I.load_state_dict(sd["head_I"])
        # Backward-compat: legacy 3-head checkpoints (no ρ/c/v) get fresh heads.
        for k, head in [("head_rho", self.head_rho), ("head_c", self.head_c), ("head_v", self.head_v)]:
            if k in sd:
                head.load_state_dict(sd[k])
            else:
                print(f"[MentalizationModule] WARNING: legacy ckpt missing {k}; using freshly init head.")


# ============================================================================
# Teacher mentalizer F_ω (paper §3.2)
# ============================================================================

# Backward-compat alias — old name "MentalizationModule" is the STUDENT.
StudentMentalizationModule = MentalizationModule


class TeacherMentalizationModule(MentalizationModule):
    """
    Teacher F_ω : (h, p_u, g) -> z_t  ∈ R^{3d+3}

    Paper §3.2:
        z_t = F_ω(h_t, p_u, g)
    Conditions on dialogue history + user profile + task goal. Trained on
    silver oracle labels (Phase 0a output) via L_F (paper eq 30), then FROZEN.
    Used only on the reward side at inference / self-play; the system policy
    never sees its output directly (only sees student M_φ's ẑ_t).

    Architecture: identical to the student (same encoder backbone, same 6
    projection heads, same √α scaling) — the *only* difference is that the
    forward() composes (g, p_u, h) into a single composite input string
    before encoding, rather than encoding history alone.
    """

    _COMPOSITE_TEMPLATE = (
        "[TASK GOAL]\n{goal}\n\n"
        "[USER PROFILE]\n{profile}\n\n"
        "[DIALOGUE HISTORY]\n{history}"
    )

    def __init__(self, encoder: SentenceEncoder, cfg: MentalizationConfig):
        super().__init__(encoder, cfg)

    @staticmethod
    def _compose(history_text: str, profile_text: str, goal_text: str) -> str:
        return TeacherMentalizationModule._COMPOSITE_TEMPLATE.format(
            goal=goal_text or "(no task goal provided)",
            profile=profile_text or "(no user profile available)",
            history=history_text or "(no turns yet)",
        )

    def forward_with_context(
        self,
        history_texts: Sequence[str],
        profile_texts: Sequence[str],
        goal_text: str,
    ) -> torch.Tensor:
        """Composite forward: (h, p_u, g) → z_t. Returns (B, 3d+3)."""
        if len(history_texts) != len(profile_texts):
            raise ValueError(
                f"history_texts ({len(history_texts)}) and profile_texts "
                f"({len(profile_texts)}) must have the same length"
            )
        composites = [
            self._compose(h, p, goal_text)
            for h, p in zip(history_texts, profile_texts)
        ]
        with torch.no_grad():
            h = self.encoder.encode(composites)         # (B, H)
        return self.forward_from_hidden(h)

    # Override forward() to fail loudly if someone tries the (history-only)
    # student-style call on a teacher. The teacher *requires* (h, p_u, g).
    def forward(self, history_texts: Sequence[str]) -> torch.Tensor:  # type: ignore[override]
        raise RuntimeError(
            "TeacherMentalizationModule.forward(history_texts) is not "
            "supported — call forward_with_context(history_texts, "
            "profile_texts, goal_text) instead. The teacher conditions on "
            "(h, p_u, g) per paper §3.2."
        )

    # ------------------------------------------------------------------ losses

    @staticmethod
    def dynamic_loss(z_pred_t: torch.Tensor, z_pred_tp1: torch.Tensor,
                     z_targ_t: torch.Tensor, z_targ_tp1: torch.Tensor) -> torch.Tensor:
        """
        L_dyn = ||(F(h_{t+1}) - F(h_t)) - (z̄_{t+1} - z̄_t)||²_A   (paper eq 30)

        Pre/post deltas are computed in the same scaled space as the heads,
        so plain MSE is the A-weighted distance (since √α is already inside
        the scalar head outputs).
        """
        delta_pred = z_pred_tp1 - z_pred_t
        delta_targ = z_targ_tp1 - z_targ_t
        return F.mse_loss(delta_pred, delta_targ)
