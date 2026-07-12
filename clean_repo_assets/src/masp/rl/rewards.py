"""
Reward functions for Mentalized Adversarial Self-Play.

All rewards are built on top of a scalar *progress* measure in the BDI
latent space. Let

    z_init ∈ R^{3d}   be the initial-turn BDI embedding (sampled from p(z))
    z_goal ∈ R^{3d}   be the task goal BDI embedding
    z_t    ∈ R^{3d}   be the ground-truth BDI at time t (from OBU)
    ẑ_t   ∈ R^{3d}   be the mentalization module's estimate

Define the unit vector towards the goal in BDI space:

    u = (z_goal - z_init) / ||z_goal - z_init||₂                (eq. 1)

The progress of any BDI vector z is its scalar projection onto u, shifted so
that z_init has progress 0:

    prog(z) = ((z - z_init) · u) / ||z_goal - z_init||₂          (eq. 2)

Under this definition prog(z_init) = 0 and prog(z_goal) = 1. For embeddings
that land in-between, prog ∈ [0, 1] with some slack outside.

System policy reward (per step t, after user responds):

    r_S^t = α_task · 1[success at t]
          + α_shape · (prog(z*_{t+1}) - prog(z*_t))
          - α_ment  · ||ẑ_t - z*_t||^2                           (eq. 3)

User policy reward (adversarial):

    r_U^t = - α_shape · (prog(z*_{t+1}) - prog(z*_t))            # resistance
            + α_fid   · <φ(u_t), φ_init>_cos                     # voice fidelity
            + α_rat   · rationality(h_t, a_t, u_t) ∈ {-1, +1}    # rationality
                                                                  (eq. 4)

KL regularization is applied in PPO (see `masp/rl/ppo.py`) with per-agent
coefficients `beta_kl_S` and `beta_kl_U`.

Mentalization supervised loss (Phase 0 and Phase 2 co-training):

    L_M(ψ) = E[||M_ψ(h_t) - z*_t||^2]                            (eq. 5)

Terminal bonus:
    * if the dialogue ends with success, the system gets +α_term and the
      user gets -α_term (zero-sum boost on success).

See README.md §3 for the high-level story.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import re

import torch
import torch.nn.functional as F

from ..utils.llm_client import LLMClient


# ---------------------------------------------------------------------- config

@dataclass
class RewardConfig:
    """
    Reward weights for paper §3.4.

    Paper-aligned core (always on):
      r_S = α_shape · Δprog − α_ment · ‖ẑ − z‖²_A − w_kl^S · KL^S      (eq 23)
      r_U = − α_shape · Δprog + α_rat · q − w_kl^U · KL^U              (eq 25)
    KL terms live in masp/rl/ppo.py (β_kl_S / β_kl_U), not here.

    The fields below default to 0 unless they appear in eq 23 / eq 25.
    The non-paper extensions (alpha_task / alpha_term / alpha_fid / step_penalty)
    are KEPT as ablation switches: set them > 0 to re-enable for an
    experiment, leave at 0 to match paper.
    """
    # ===== paper §3.4 core terms =====
    alpha_shape: float = 1.0    # eq 23 / 25: w_prog
    alpha_ment: float = 0.25    # eq 23: w_inf
    alpha_rat: float = 0.5      # eq 25: w_rat

    # ===== non-paper ablation switches (default OFF) =====
    alpha_task: float = 0.0     # success bonus on system; NOT in paper, ablation only
    alpha_term: float = 0.0     # terminal zero-sum boost; NOT in paper, ablation only
    alpha_fid: float = 0.0      # voice fidelity for user; NOT in paper, ablation only
    step_penalty: float = 0.0   # AT regularizer; NOT in paper, ablation only
    alpha_safety: float = 0.0   # anti-coercion penalty for system; NOT in paper
    alpha_early_success: float = 0.0  # rationality-gated early success bonus
    alpha_close_quality: float = 0.0  # bounded reward for safe low-pressure close timing

    # ===== task-side helpers =====
    success_threshold: float = 0.6   # for the success judge, not the reward formula


# ----------------------------------------------------------- progress in BDI

def progress_score(
    z_t: torch.Tensor,
    z_init: torch.Tensor,
    z_goal: torch.Tensor,
    eps: float = 1e-6,
    n_scalar_dims: int = 3,
) -> torch.Tensor:
    """
    User-relative progress in the paper's A-weighted BDI latent space (eq 17):

        prog(z) = ((z - z_init)^T A (z_goal - z_init))
                  / ((z_goal - z_init)^T A (z_goal - z_init) + ε)

    A = diag(1/(3d)·I_{3d}, α_ρ, α_c, α_v).

    The √α factors on the scalar tail are already applied inside `encode_bdi`,
    so we only need to apply 1/√(3d) to the text part before standard L2 dot
    products. Accepts (3d+3,) or (B, 3d+3).
    """
    from ..mind.bdi_schema import apply_metric_sqrt
    z_t_pre    = apply_metric_sqrt(z_t,    n_scalar_dims=n_scalar_dims)
    z_init_pre = apply_metric_sqrt(z_init, n_scalar_dims=n_scalar_dims)
    z_goal_pre = apply_metric_sqrt(z_goal, n_scalar_dims=n_scalar_dims)

    if z_init_pre.dim() == 1:
        diff_goal = (z_goal_pre - z_init_pre).unsqueeze(0)
    else:
        diff_goal = z_goal_pre - z_init_pre
    if z_t_pre.dim() == 1:
        diff_t = (z_t_pre - z_init_pre).unsqueeze(0)
    else:
        diff_t = z_t_pre - z_init_pre
    denom = (diff_goal * diff_goal).sum(dim=-1).clamp(min=eps)
    num = (diff_t * diff_goal).sum(dim=-1)
    prog = num / denom
    return prog.squeeze()


def a_weighted_sq_distance(z_a: torch.Tensor, z_b: torch.Tensor, n_scalar_dims: int = 3) -> torch.Tensor:
    """
    Paper eq 23 second term (and Appendix L_F / L_M): ||z_a - z_b||_A^2.
    Equals plain L2 squared distance after applying `apply_metric_sqrt` to both.
    """
    from ..mind.bdi_schema import apply_metric_sqrt
    a = apply_metric_sqrt(z_a, n_scalar_dims=n_scalar_dims)
    b = apply_metric_sqrt(z_b, n_scalar_dims=n_scalar_dims)
    return ((a - b) ** 2).sum(dim=-1)


# ------------------------------------------------------ rationality judge

_RATIONALITY_PROMPT = """\
You are an impartial evaluator of dialogues.

Task context:
{task_description}

The user internally holds this mental state:
{user_bdi}

Here is the most recent exchange:
Assistant: {assistant_turn}
User:      {user_turn}

Question: Was it rational for a user with that internal mental state to
respond this way? A "rational" response means:
  (a) if the assistant gave a genuinely compelling, concrete, and honest
      reason to update their belief/desire/intention, the user reasonably
      updated toward it; OR
  (b) if the assistant did NOT give such a reason, the user reasonably
      resisted and kept their mental state stable.

A "not rational" response means the user either (i) conceded for no good
reason (sycophancy) or (ii) stubbornly refused despite a clear, compelling
argument.

Answer with EXACTLY one of:
  RATIONAL
  NOT_RATIONAL
"""


class RationalityJudge:
    """
    Wraps an LLMClient with the universal rationality prompt. Returns +1 for
    RATIONAL, -1 for NOT_RATIONAL, and 0 if the response is unparseable.
    """

    def __init__(
        self,
        llm: LLMClient,
        task_description: str,
        temperature: float = 0.0,
        max_tokens: int = 8,
    ):
        self.llm = llm
        self.task_description = task_description
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def __call__(
        self,
        assistant_turn: str,
        user_turn: str,
        user_bdi_text: str,
    ) -> int:
        prompt = _RATIONALITY_PROMPT.format(
            task_description=self.task_description.strip(),
            user_bdi=user_bdi_text,
            assistant_turn=assistant_turn,
            user_turn=user_turn,
        )
        try:
            out = self.llm.call(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception:
            return 0
        t = (out or "").strip().upper()
        if "NOT_RATIONAL" in t or "NOT RATIONAL" in t:
            return -1
        if "RATIONAL" in t:
            return 1
        return 0


# ---------------------------------------------------------- reward bundles

@dataclass
class RewardBundle:
    """One turn's worth of rewards from both sides, plus diagnostic parts."""

    r_system: float
    r_user: float
    success: bool
    progress_delta: float
    rationality: int
    mental_error: float
    fidelity: float
    task_label: str = ""
    task_reward: float = 0.0
    safety_penalty: float = 0.0
    close_quality: float = 0.0
    raw_judgments: List[str] = field(default_factory=list)


_REFUSAL_PATTERNS = (
    "no thank",
    "no thanks",
    "no thank you",
    "no thank",
    "not donate",
    "not give",
    "don't want to donate",
    "do not want to donate",
    "wouldn't donate",
    "wouldn't give",
    "would do nothing",
    "will donate 0",
    "donate 0",
    "0 dollars",
    "$0",
    "nothing",
    "nothing at all",
    "can't donate",
    "cannot donate",
    "not at this time",
    "don't plan on donating",
    "do not plan on donating",
    "would not like",
)

_DONATION_ASK_PATTERNS = (
    "how much",
    "would you like to donate",
    "would you be open",
    "want to donate",
    "willing to donate",
    "will you donate",
    "make a donation",
    "donation from",
    "donate any amount",
)

_COERCIVE_ASSIGN_PATTERNS = (
    "put you down",
    "mark you down",
    "sign you up",
    "count you in",
    "count you for",
    "i'll put",
    "i will put",
    "i'll still put",
    "i will still put",
    "record you for",
    "take your refusal as yes",
    "take your refusal as a yes",
    "take refusal as yes",
    "take refusal as a yes",
)

_DISRESPECT_PATTERNS = (
    "you don't care",
    "you do not care",
    "you sound like",
    "selfish",
    "shame on",
    "bad person",
    "don't care about the less fortunate",
    "do not care about the less fortunate",
)

_UNSUPPORTED_CLAIM_PATTERNS = (
    "100% tax deductible",
    "guarantee the funds",
    "guarantee your donation",
    "i will end our conversation now",
)

_SAFE_SMALL_CLOSE_RE = re.compile(
    r"("
    r"\$0?\.25|25\s+cents?|small\s+non[-\s]?zero|token\s+\$0?\.25|"
    r"from\s+your\s+task\s+payment"
    r")",
    re.IGNORECASE,
)
_HIGH_AMOUNT_RE = re.compile(
    r"\$\s*(?:[1-9]|\d{2,})(?:\b|\s|[.,])|"
    r"\b(?:all|entire|full)\s+(?:of\s+)?(?:your\s+)?(?:task\s+payment|\$\d)\b",
    re.IGNORECASE,
)
_GOODBYE_RE = re.compile(
    r"\b(?:goodbye|good bye|have\s+(?:a\s+)?(?:nice|good|wonderful)\s+day|"
    r"enjoy\s+(?:the\s+)?rest\s+of\s+your\s+day|thank\s+you\s+for\s+your\s+time)\b",
    re.IGNORECASE,
)


def system_safety_penalty(system_turn: str, pre_history: Sequence[str]) -> float:
    """
    Lightweight guardrail reward feature for P4G-style persuasion.

    It penalizes behavior that can inflate the task judge while violating the
    task contract: assigning a donation without consent, pressuring after a
    clear refusal, or using personal attacks. Kept separate from the paper
    reward and gated by RewardConfig.alpha_safety.
    """
    text_raw = str(system_turn or "").lower()
    text = re.sub(r"[^a-z0-9$]+", " ", text_raw)
    penalty = 0.0

    text_haystacks = (text_raw, text)

    if any(p in h for h in text_haystacks for p in _COERCIVE_ASSIGN_PATTERNS):
        penalty += 1.0
    if any(p in h for h in text_haystacks for p in _DISRESPECT_PATTERNS):
        penalty += 1.0
    if any(p in h for h in text_haystacks for p in _UNSUPPORTED_CLAIM_PATTERNS):
        penalty += 0.5

    last_user = ""
    last_user_raw = ""
    for line in reversed(list(pre_history or [])):
        if str(line).startswith("User:"):
            last_user_raw = str(line).lower()
            last_user = re.sub(r"[^a-z0-9$]+", " ", last_user_raw)
            break
    last_user_haystacks = (last_user_raw, last_user)
    if last_user and any(p in h for h in last_user_haystacks for p in _REFUSAL_PATTERNS):
        if any(p in h for h in text_haystacks for p in _DONATION_ASK_PATTERNS) or any(
            p in h for h in text_haystacks for p in _COERCIVE_ASSIGN_PATTERNS
        ):
            penalty += 1.0

    return float(min(penalty, 2.0))


def system_close_quality(
    system_turn: str,
    pre_history: Sequence[str],
    *,
    turn_idx: Optional[int],
    success: bool,
) -> float:
    """Bounded shaping feature for P4G close timing.

    Positive values reward explicit, low-pressure, small-dollar donation asks.
    Negative values punish high-dollar/full-payment asks and premature goodbye
    while the episode is still unsolved. The feature is only applied when
    RewardConfig.alpha_close_quality > 0.
    """
    text_raw = str(system_turn or "")
    text = text_raw.lower()
    has_ask = any(p in text for p in _DONATION_ASK_PATTERNS)
    has_safe_small = bool(_SAFE_SMALL_CLOSE_RE.search(text_raw))
    quality = 0.0

    if _HIGH_AMOUNT_RE.search(text_raw):
        quality -= 1.0
    if has_ask and has_safe_small:
        quality += 0.5
        if turn_idx is not None and 2 <= int(turn_idx) <= 5:
            quality += 0.25
    if not success and turn_idx is not None and int(turn_idx) < 8:
        if _GOODBYE_RE.search(text_raw) and not has_ask:
            quality -= 0.5

    last_user = ""
    for line in reversed(list(pre_history or [])):
        if str(line).startswith("User:"):
            last_user = str(line).lower()
            break
    if last_user and any(p in last_user for p in _REFUSAL_PATTERNS):
        if has_ask:
            quality -= 0.75

    return float(max(min(quality, 1.0), -1.0))


# ---------------------------------------------------------- system reward

def system_reward(
    *,
    success: bool,
    progress_t: float,
    progress_tp1: float,
    mental_error_sq: float,
    step_penalty: float,
    safety_penalty: float,
    close_quality: float,
    cfg: RewardConfig,
    terminal: bool = False,
    turn_idx: Optional[int] = None,
    max_turns: Optional[int] = None,
    rationality_signal: int = 0,
) -> float:
    """
    Paper §3.4 eq 23 (KL omitted — added inside PPO):
        r_S^t = α_shape · Δprog_t − α_ment · ‖ẑ_t − z_t‖²_A

    All other terms below default to 0 (RewardConfig defaults). They are
    ablation switches — set the corresponding RewardConfig field > 0 to
    re-enable, but the paper's r_S has only the two core terms above.
    """
    r = 0.0
    # ----- paper core (eq 23) -----
    r += cfg.alpha_shape * float(progress_tp1 - progress_t)
    r -= cfg.alpha_ment * float(mental_error_sq)
    # ----- non-paper ablation extensions (default 0) -----
    r -= float(step_penalty)                    # AT regularizer
    r -= cfg.alpha_safety * float(safety_penalty)
    r += cfg.alpha_close_quality * float(close_quality)
    if success:
        r += cfg.alpha_task                     # success bonus
        if (
            cfg.alpha_early_success > 0.0
            and rationality_signal > 0
            and turn_idx is not None
            and max_turns is not None
            and int(max_turns) > 0
        ):
            remaining = max(int(max_turns) - int(turn_idx) + 1, 1)
            r += cfg.alpha_early_success * (remaining / float(max_turns))
        if terminal:
            r += cfg.alpha_term                 # terminal zero-sum boost
    return float(r)


# ---------------------------------------------------------- user reward

def user_reward(
    *,
    progress_t: float,
    progress_tp1: float,
    fidelity_cos: float,
    rationality_signal: int,
    cfg: RewardConfig,
    success: bool = False,
    terminal: bool = False,
) -> float:
    """
    Paper §3.4 eq 25 (KL omitted — added inside PPO):
        r_U^t = − α_shape · Δprog_t + α_rat · q_t

    α_fid and α_term are non-paper extensions (default 0).
    """
    r = 0.0
    # ----- paper core (eq 25) -----
    r -= cfg.alpha_shape * float(progress_tp1 - progress_t)   # resistance
    r += cfg.alpha_rat * float(rationality_signal)            # rationality judge
    # ----- non-paper ablation extensions (default 0) -----
    r += cfg.alpha_fid * float(fidelity_cos)                  # voice fidelity
    if success and terminal:
        r -= cfg.alpha_term                                    # terminal zero-sum
    return float(r)


# ---------------------------------------------------- helper: cos fidelity

def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-row cosine similarity. Accepts (d,) or (B, d)."""
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return (a * b).sum(dim=-1).squeeze()


def fidelity_cosine(
    user_turn_emb: torch.Tensor,
    z_init_emb: torch.Tensor,
) -> float:
    """
    Voice fidelity: how similar is the user's latest utterance embedding to
    the initial mental state embedding? High value => the user is still
    "sounding like" its sampled mind.
    """
    with torch.no_grad():
        c = _cosine(user_turn_emb, z_init_emb)
        return float(c.item()) if c.dim() == 0 else float(c.mean().item())
