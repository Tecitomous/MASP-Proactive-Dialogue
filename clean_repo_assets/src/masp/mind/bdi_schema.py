"""
BDI (Belief-Desire-Intention) schema — 6-dim mental state per paper §3.2.

Following Bratman (1987) / Rao & Georgeff (1995), augmented with three scalar
affective variables as in our paper:

    Belief (B)      — what the user takes to be true about the world      [str]
    Desire (D)      — what the user wants (goals / preferences)            [str]
    Intention (I)   — what the user is currently committed to do           [str]
    receptivity ρ   — openness to assistant's proposals                    [0, 1]
    confidence  c   — commitment / certainty in current stance             [0, 1]
    valence     v   — emotional polarity (negative … positive)             [-1, 1]

Encoded into a dense vector via a frozen sentence encoder φ + a precondition
that makes standard L2 equal to the paper's A-weighted metric:

    z_t = [ φ(B)/√(3d); φ(D)/√(3d); φ(I)/√(3d); √α_ρ·ρ; √α_c·c; √α_v·v ]   ∈ R^{3d+3}

with this preconditioning, paper eq 12 `||x||_A² = x^T A x = (1/3d)||e||² +
α_ρ ρ² + α_c c² + α_v v²` reduces to plain L2 in the encoded space.

The (α_ρ, α_c, α_v) are stored on `BDITaskConfig`; default 1.0/1.0/1.0.

Backward compatibility: when ρ/c/v are absent from a data source, they
default to (0.5, 0.5, 0.0). Old data_cache files therefore still load.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import torch  # required by encode_bdi / apply_metric_sqrt only
except ImportError:  # allow schema to be importable for offline data work / unit tests
    torch = None  # type: ignore


# ---------------------------------------------------------------------- BDI

@dataclass
class BDI:
    """One instantiation of a (B, D, I, ρ, c, v) mental state."""
    belief: str
    desire: str
    intention: str
    # Paper §3.2 affective scalars; defaults are neutral so legacy code
    # that constructs `BDI(b, d, i)` with positional args still works.
    receptivity: float = 0.5     # ρ ∈ [0, 1]
    confidence: float = 0.5      # c ∈ [0, 1]
    valence: float = 0.0         # v ∈ [-1, 1]

    # ---------------------------------------------------------------- text

    def to_text(self, include_scalars: bool = True) -> str:
        """Canonical text form. Scalars shown to 2 decimals when included."""
        base = (
            f"Belief: {self.belief}\n"
            f"Desire: {self.desire}\n"
            f"Intention: {self.intention}"
        )
        if not include_scalars:
            return base
        return base + (
            f"\nReceptivity: {self.receptivity:.2f}"
            f"\nConfidence: {self.confidence:.2f}"
            f"\nValence: {self.valence:+.2f}"
        )

    def to_list(self) -> List[str]:
        """Three text components only — used by sentence encoder."""
        return [self.belief, self.desire, self.intention]

    # ---------------------------------------------------------------- dict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "belief": self.belief,
            "desire": self.desire,
            "intention": self.intention,
            "receptivity": float(self.receptivity),
            "confidence": float(self.confidence),
            "valence": float(self.valence),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BDI":
        """Tolerant loader: missing scalars default to neutral."""
        return cls(
            belief=str(d.get("belief", "Unknown.")),
            desire=str(d.get("desire", "Unknown.")),
            intention=str(d.get("intention", "Unknown.")),
            receptivity=_clip(_to_float(d.get("receptivity", 0.5)), 0.0, 1.0),
            confidence=_clip(_to_float(d.get("confidence", 0.5)), 0.0, 1.0),
            valence=_clip(_to_float(d.get("valence", 0.0)), -1.0, 1.0),
        )

    # ---------------------------------------------------------------- parse

    @classmethod
    def from_text(cls, text: str) -> "BDI":
        """
        Parse a BDI block produced by the OBU. The LLM is instructed to
        output 6 lines (Belief / Desire / Intention / Receptivity /
        Confidence / Valence). We are permissive about case / ordering;
        missing fields fall back to defaults.
        """
        b = d = i = ""
        rho: Optional[float] = None
        conf: Optional[float] = None
        val: Optional[float] = None

        for line in (text or "").splitlines():
            low = line.strip()
            if not low:
                continue
            lower = low.lower()
            if lower.startswith("belief"):
                b = low.split(":", 1)[-1].strip()
            elif lower.startswith("desire"):
                d = low.split(":", 1)[-1].strip()
            elif lower.startswith("intention"):
                i = low.split(":", 1)[-1].strip()
            elif lower.startswith("receptivity") or lower.startswith("openness"):
                rho = _safe_parse_scalar(low)
            elif lower.startswith("confidence") or lower.startswith("commitment"):
                conf = _safe_parse_scalar(low)
            elif lower.startswith("valence") or lower.startswith("emotion"):
                val = _safe_parse_scalar(low)

        if not (b or d or i):
            # Degenerate: dump entire blob into belief to avoid downstream NaNs
            b = (text or "").strip() or "Unknown."

        return cls(
            belief=b or "Unknown.",
            desire=d or "Unknown.",
            intention=i or "Unknown.",
            receptivity=_clip(rho if rho is not None else 0.5, 0.0, 1.0),
            confidence=_clip(conf if conf is not None else 0.5, 0.0, 1.0),
            valence=_clip(val if val is not None else 0.0, -1.0, 1.0),
        )


# ---------------------------------------------------------------- utilities

def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _safe_parse_scalar(line: str) -> Optional[float]:
    """Pull a number out of a line like 'Receptivity: 0.7' or '... 0.7 ...'."""
    import re
    m = re.search(r"-?\d+(?:\.\d+)?", line.split(":", 1)[-1])
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


# ---------------------------------------------------------------- task config

@dataclass
class BDITaskConfig:
    """All task-specific strings + scalar weights for the latent geometry."""

    task_name: str
    task_description: str
    assistant_role: str
    user_role: str
    goal_bdi: BDI
    extraction_context_hint: str = ""
    user_profile_template: str = ""
    # Paper §3.3 metric weights: A = diag(1/d * I_d, α_ρ, α_c, α_v)
    alpha_rho: float = 1.0
    alpha_c: float = 1.0
    alpha_v: float = 1.0


P4G_GOAL_BDI = BDI(
    belief=(
        "Save the Children is a trustworthy charity and even small "
        "donations make a real, concrete difference in children's lives."
    ),
    desire=(
        "The user wants to help children in need and feels morally "
        "motivated to contribute."
    ),
    intention=(
        "The user intends to donate now to Save the Children."
    ),
    # Paper §3.3 target z*: high receptivity, fairly committed, mildly positive
    receptivity=1.0,
    confidence=0.8,
    valence=0.5,
)

P4G_CFG = BDITaskConfig(
    task_name="p4g",
    task_description=(
        "A two-party online dialogue in which one speaker (the Persuader / "
        "Assistant) tries to convince the other speaker (the Persuadee / "
        "User) to donate to a charity called 'Save the Children'. The "
        "persuadee may be reluctant, skeptical, financially constrained, "
        "or uninterested at the beginning of the conversation."
    ),
    assistant_role="persuader",
    user_role="persuadee",
    goal_bdi=P4G_GOAL_BDI,
    extraction_context_hint=(
        "P4G-specific guidance:\n"
        "- Belief: What does the user think about the charity, donation, or "
        "the topic? E.g., 'The user is skeptical about whether donations "
        "actually reach children.' NOT a restatement of the assistant's pitch.\n"
        "- Desire: What does the USER want? Often this is NOT to donate. "
        "E.g., 'The user wants to protect their money' or 'The user wants "
        "to help but worries about affordability.'\n"
        "- Intention: What will the user likely do next? E.g., 'The user "
        "intends to decline' or 'The user is considering a small donation.'\n"
        "- Receptivity: Most persuadees START skeptical or neutral (0.2-0.5). "
        "Only raise above 0.6 when the user shows genuine engagement or "
        "softening. A user who says 'I don't like children' is ~0.1. "
        "A user who says 'Tell me more about the charity' is ~0.6.\n"
        "- Confidence: How firm is the user? 'No way' = 0.9. 'I'm not sure' "
        "= 0.3. A user who hasn't expressed a clear position = 0.2-0.4.\n"
        "- Valence: Reflect actual emotional tone. Polite small talk = 0.1. "
        "Enthusiasm about helping = 0.5+. Annoyance at being pressured = -0.3."
    ),
)


# =============================================================== ESConv

ESCONV_GOAL_BDI = BDI(
    belief=(
        "Talking with the supporter and trying their suggestions can "
        "genuinely help me cope with what I am feeling."
    ),
    desire=(
        "The user wants to feel less distressed and to find emotional or "
        "practical ways to cope with the situation."
    ),
    intention=(
        "Open up about the situation, accept emotional support, and try a "
        "concrete coping suggestion."
    ),
    receptivity=1.0,
    confidence=0.7,
    valence=0.4,
)

ESCONV_CFG = BDITaskConfig(
    task_name="esconv",
    task_description=(
        "A two-party emotional-support conversation in which one speaker "
        "(the Supporter / Assistant) tries to help the other speaker (the "
        "Help-Seeker / User) cope with an emotional distress they are "
        "currently experiencing (e.g. job loss, anxiety, grief, "
        "relationship issues). The help-seeker may feel overwhelmed, "
        "hopeless, anxious, guilty, or ashamed at the start of the "
        "conversation."
    ),
    assistant_role="supporter",
    user_role="help-seeker",
    goal_bdi=ESCONV_GOAL_BDI,
    extraction_context_hint=(
        "ESConv-specific guidance:\n"
        "- Belief: What does the user think about their situation, the "
        "supporter, or whether things can get better? E.g., 'The user "
        "feels the situation is hopeless' or 'The user believes the "
        "supporter understands them.' NOT a paraphrase of the supporter's "
        "advice.\n"
        "- Desire: What does the USER want emotionally? E.g., 'The user "
        "wants to feel understood' or 'The user wants concrete advice on "
        "what to do next.'\n"
        "- Intention: What is the user about to do — keep venting, ask for "
        "advice, push back on the supporter's suggestion, or close off?\n"
        "- Receptivity: Most help-seekers START guarded or skeptical "
        "(0.3-0.5). Move above 0.6 only when they explicitly accept "
        "advice, thank the supporter, or visibly soften.\n"
        "- Confidence: How firm is the user in their current emotional "
        "stance? 'I've tried everything, nothing works' = 0.8. 'I don't "
        "know what to do' = 0.2.\n"
        "- Valence: Help-seekers typically start NEGATIVE (-0.3 to -0.7). "
        "Move toward 0 / positive only when relief, gratitude, or warmth "
        "is explicitly expressed."
    ),
)


# =============================================================== Craigslist Bargain

CRAIGSLIST_GOAL_BDI = BDI(
    belief=(
        "The seller believes the buyer's offer can still be a mutually "
        "acceptable deal and that closing now is better than walking away."
    ),
    desire=(
        "The seller wants to close the sale at an acceptable price and is "
        "willing to accept a reasonable buyer-favorable offer."
    ),
    intention=(
        "Concede if needed, agree on the final price, and confirm the "
        "transaction."
    ),
    receptivity=0.8,
    confidence=0.7,
    valence=0.3,
)

CRAIGSLIST_CFG = BDITaskConfig(
    task_name="craigslist_bargain",
    task_description=(
        "A two-party bargaining dialogue over a used item listed online. "
        "For this buyer-system orientation, Assistant is always the buyer "
        "policy being trained/evaluated, and User is always the seller "
        "counterparty. Each side has a private target price, but the "
        "evaluation success metric is buyer utility, so infer the seller "
        "User's mental state as the buyer tries to reach a favorable deal."
    ),
    assistant_role="buyer / system policy",
    user_role="seller / counterparty",
    goal_bdi=CRAIGSLIST_GOAL_BDI,
    extraction_context_hint=(
        "Craigslist-Bargain-specific guidance:\n"
        "- Do not infer USER role from price direction. In this split, "
        "Assistant is the buyer/system and User is the seller/counterparty.\n"
        "- Extract the seller User's BDI only. Do not describe the buyer's "
        "desire to purchase cheaply as the User's desire.\n"
        "- Belief: What does the seller think about the item's value, the "
        "buyer's willingness to move, or the seller's own walk-away point?\n"
        "- Desire: What price / outcome is the seller actually aiming for, "
        "and how willing are they to accept the buyer's offer?\n"
        "- Intention: Are they about to make a counter-offer, hold firm, "
        "concede, accept, or walk away?\n"
        "- Receptivity: How willing is the seller to move from their "
        "current stated price toward the buyer? Holding firm = 0.1-0.3; "
        "starting to concede = 0.4-0.6; eager to close = 0.7+.\n"
        "- Confidence: How firm is the seller in their current offer? "
        "'Final offer, take it or leave it' = 0.9. 'Maybe we can work "
        "something out' = 0.4. Early opening offers = 0.3-0.5.\n"
        "- Valence: Bargaining is usually transactional and roughly "
        "neutral (-0.1 to +0.1). Mark frustration after refusals as "
        "-0.3 to -0.5; satisfaction at an accepted offer as +0.3 to +0.5."
    ),
)


# =============================================================== EmpatheticDialogues

EMPATHETIC_GOAL_BDI = BDI(
    belief=(
        "The listener genuinely understands and acknowledges what I am "
        "feeling about this experience."
    ),
    desire=(
        "The user wants to feel heard, validated, and emotionally "
        "connected with the listener."
    ),
    intention=(
        "Continue sharing the experience openly and engage with the "
        "listener's responses."
    ),
    receptivity=0.8,
    confidence=0.6,
    valence=0.3,
)

EMPATHETIC_CFG = BDITaskConfig(
    task_name="empathetic_dialogues",
    task_description=(
        "A short two-party conversation grounded in a real personal "
        "experience tied to a specific emotion (e.g. guilty, afraid, "
        "proud, joyful, lonely, surprised). One speaker (the Speaker / "
        "User) recounts the experience; the other speaker (the Listener / "
        "Assistant) responds empathetically — asking follow-ups, "
        "acknowledging feelings, or gently reframing. The conversation is "
        "casual and emotionally driven rather than goal-directed."
    ),
    assistant_role="listener",
    user_role="speaker",
    goal_bdi=EMPATHETIC_GOAL_BDI,
    extraction_context_hint=(
        "EmpatheticDialogues-specific guidance:\n"
        "- Belief: What does the user currently believe about the "
        "experience they're recounting, or about how the listener is "
        "responding to them?\n"
        "- Desire: What does the USER want emotionally — to vent, to be "
        "validated, to laugh it off, to make sense of what happened?\n"
        "- Intention: Are they about to elaborate further, change topic, "
        "ask the listener something, or wind down the conversation?\n"
        "- Receptivity: How open is the user to the listener's empathic "
        "responses? Brief one-liner replies = ~0.4; warm engagement and "
        "elaboration = 0.7+; deflection or short shutdown = 0.2-0.3.\n"
        "- Confidence: How sure is the user about the emotion or "
        "interpretation they are conveying? Vivid, detailed recall = 0.7+; "
        "'I don't really know how I felt' = 0.2-0.3.\n"
        "- Valence: Match the underlying emotion of the story. "
        "Sad / afraid / guilty / lonely should be clearly negative "
        "(-0.3 to -0.7); proud / joyful / excited / grateful should be "
        "clearly positive (+0.3 to +0.7); neutral reflection close to 0."
    ),
)


# =============================================================== CIMA

CIMA_GOAL_BDI = BDI(
    belief=(
        "The tutor's hints, questions, corrections, and confirmations are "
        "useful for producing the correct Italian translation."
    ),
    desire=(
        "The student wants to complete the translation correctly and "
        "understand the relevant Italian words or grammar."
    ),
    intention=(
        "Use the tutor's feedback to revise or provide the correct Italian "
        "translation."
    ),
    receptivity=1.0,
    confidence=0.7,
    valence=0.4,
)

CIMA_CFG = BDITaskConfig(
    task_name="cima",
    task_description=(
        "A two-party tutoring dialogue in which the Assistant is a tutor "
        "helping the User, a student, translate an English prepositional "
        "sentence into Italian. The tutor may ask questions, provide hints, "
        "give corrections, or confirm correct parts. The student may ask "
        "for vocabulary, make partial guesses, express uncertainty, or try "
        "to produce the target translation."
    ),
    assistant_role="tutor / teacher",
    user_role="student / learner",
    goal_bdi=CIMA_GOAL_BDI,
    extraction_context_hint=(
        "CIMA-specific guidance:\n"
        "- Belief: What does the student currently believe about the "
        "translation, vocabulary, grammar rule, or tutor feedback? "
        "Mention uncertainty when the student is guessing or asking for a "
        "word.\n"
        "- Desire: What does the STUDENT want? Usually to solve the current "
        "translation, get a missing word, check a grammar point, or confirm "
        "a guess.\n"
        "- Intention: What is the student likely to do next — ask for a "
        "hint, make another guess, revise a phrase, accept a correction, or "
        "finish the answer?\n"
        "- Receptivity: Openness to the tutor's guidance. Asking for help "
        "or using a correction is 0.6-0.8; repeated confusion but still "
        "engaged is 0.4-0.6; ignoring feedback or going off track is lower.\n"
        "- Confidence: How sure is the student about the current answer? "
        "Explicit uncertainty is 0.2-0.4; a fluent complete answer is "
        "0.6-0.8; a firm but wrong guess can still have high confidence.\n"
        "- Valence: Tutoring is often neutral. Mark frustration/confusion "
        "negative, and successful engagement or appreciation positive."
    ),
)


TASK_CONFIGS: Dict[str, BDITaskConfig] = {
    "p4g": P4G_CFG,
    "esconv": ESCONV_CFG,
    "craigslist_bargain": CRAIGSLIST_CFG,
    "empathetic_dialogues": EMPATHETIC_CFG,
    "cima": CIMA_CFG,
}


# ---------------------------------------------------------------- encoding

def encode_bdi(
    bdi: BDI,
    encoder,                       # masp.models.sentence_encoder.SentenceEncoder
    alpha_rho: float = 1.0,
    alpha_c: float = 1.0,
    alpha_v: float = 1.0,
):  # -> torch.Tensor (annotation omitted to keep import-light)
    """
    Encode a 6-dim BDI into the latent state z_t per paper eq 9.

        z_t = [ φ(B); φ(D); φ(I); √α_ρ·ρ; √α_c·c; √α_v·v ]   ∈ R^{3d+3}

    The √α factors absorb the scalar half of the metric A
    (eq 11) so that A-weighted operations only need to apply the text-side
    weight `1/(3d)` separately. See `apply_metric_sqrt()` below and
    `progress_score()` in masp/rl/rewards.py for the geometric side.

    Returns a (3*d + 3,) tensor on the encoder's device.
    """
    if torch is None:
        raise RuntimeError("encode_bdi requires torch; install it on the runtime node.")
    texts = bdi.to_list()                                  # 3 text components
    with torch.no_grad():
        embs = encoder.encode(texts)                       # (3, d)
    if embs.dim() != 2 or embs.shape[0] != 3:
        raise RuntimeError(f"sentence encoder returned unexpected shape: {embs.shape}")
    flat = embs.reshape(-1).float()                        # (3d,)
    flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

    scalars = torch.tensor(
        [
            math.sqrt(max(alpha_rho, 0.0)) * float(bdi.receptivity),
            math.sqrt(max(alpha_c,   0.0)) * float(bdi.confidence),
            math.sqrt(max(alpha_v,   0.0)) * float(bdi.valence),
        ],
        dtype=flat.dtype,
        device=flat.device,
    )

    z = torch.cat([flat, scalars], dim=0)                  # (3d + 3,)
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def apply_metric_sqrt(z, n_scalar_dims: int = 3):  # -> torch.Tensor
    """
    Apply the square-root of the paper's A-metric to z so that subsequent
    plain-L2 operations equal A-weighted operations:

        z_pre = [ z[: 3d] / √(3d) ; z[3d:] ]

    Standard L2 inner product on z_pre then equals
        z₁^T A z₂  =  (1/(3d)) * <text₁, text₂> + sum_i α_i scalar₁_i scalar₂_i
    (the √α factors are already inside z[3d:] from `encode_bdi`).

    Works on shapes (..., 3d + n_scalar_dims).
    """
    n = z.shape[-1]
    n_text = n - int(n_scalar_dims)
    if n_text <= 0:
        return z
    scale = 1.0 / math.sqrt(float(n_text))
    text = z[..., :n_text] * scale
    scalars = z[..., n_text:]
    return torch.cat([text, scalars], dim=-1)


def decode_bdi_text(bdi: BDI) -> str:
    return bdi.to_text(include_scalars=True)
