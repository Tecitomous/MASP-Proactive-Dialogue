"""
DialogXpert-P4G success critic.

Adapted from the previous project so we can keep the exact same evaluation
protocol:
  refused  -> -1.0
  neutral  -> -0.5
  positive ->  0.1
  agree    ->  1.0
success iff averaged reward > threshold (default 0.6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.llm_client import LLMClient, LLMConfig


P4G_REWARD_DICT: Dict[str, float] = {
    "refused": -1.0,
    "neutral": -0.5,
    "positive": 0.1,
    "agree": 1.0,
}


@dataclass
class RewardOutput:
    reward: float
    success: bool
    label: str
    raw_judgments: List[str] = field(default_factory=list)
    pos_hits: int = 0
    neg_hits: int = 0


def score_user_reply_heuristic(user_reply: str, step_penalty: float = 0.02) -> RewardOutput:
    """Cheap rule-based reward used for debugging."""
    text = (user_reply or "").strip().lower()
    pos_cues = {
        "yes", "sure", "okay", "i can", "i will", "sounds good",
        "i agree", "i'm in", "donate", "i can donate", "i will donate",
    }
    neg_cues = {
        "no", "can't", "cannot", "won't", "not interested",
        "i disagree", "not donate", "i won't",
    }
    pos = sum(1 for c in pos_cues if c in text)
    neg = sum(1 for c in neg_cues if c in text)
    success = pos > neg and pos > 0
    reward = 0.8 * pos - 0.8 * neg - step_penalty + (1.0 if success else 0.0)
    label = "agree" if success else ("refused" if neg > pos else "neutral")
    return RewardOutput(
        reward=float(reward),
        success=bool(success),
        label=label,
        pos_hits=int(pos),
        neg_hits=int(neg),
    )


def _normalize_label(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    if not t:
        return None
    if "explicitly refused" in t:
        return "refused"
    if "remains neutral" in t:
        return "neutral"
    if "positive attitude" in t:
        return "positive"
    if "decided to donate" in t:
        return "agree"
    if "refused" in t:
        return "refused"
    if "neutral" in t:
        return "neutral"
    if "positive" in t:
        return "positive"
    if "agree" in t:
        return "agree"
    if "donate" in t and ("decided" in t or "will" in t):
        return "agree"
    token = t.split()[0].strip(".,:;!?()[]{}\"'")
    return token if token in P4G_REWARD_DICT else None


def _aggregate(judgments: List[str], success_threshold: float) -> RewardOutput:
    labels: List[str] = []
    rewards: List[float] = []
    for item in judgments:
        lb = _normalize_label(item)
        if lb is None:
            continue
        labels.append(lb)
        rewards.append(float(P4G_REWARD_DICT[lb]))

    reward = float(sum(rewards) / len(rewards)) if rewards else 0.0
    success = bool(reward > float(success_threshold))
    if not labels:
        label = "unknown"
    else:
        order = ["refused", "neutral", "positive", "agree"]
        label = max(order, key=lambda x: labels.count(x))
    return RewardOutput(
        reward=reward,
        success=success,
        label=label,
        raw_judgments=list(judgments),
        pos_hits=int(labels.count("positive") + labels.count("agree")),
        neg_hits=int(labels.count("refused")),
    )


def _build_prompt(
    history_lines: List[str],
    assistant_action: str,
    user_reply: str,
    max_history_lines: int,
) -> str:
    use_lines = history_lines[-max_history_lines:] if max_history_lines > 0 else history_lines
    conv = "\n".join(use_lines)
    return (
        "You are an expert in evaluating persuasive conversations between a "
        "persuader and a persuadee.\n"
        "Your job is to determine if the persuader has successfully convinced "
        "the persuadee to donate to a charity called 'Save the Children'.\n\n"
        "Conversation so far:\n"
        f"{conv}\n"
        f"Assistant: {assistant_action}\n"
        f"Persuadee: {user_reply}\n\n"
        "Please evaluate the conversation and strictly choose ONE of the exact "
        "responses below.\n"
        "Question: Has the persuadee been successfully convinced to make a "
        "donation at the end of the conversation?\n\n"
        "Respond with one of:\n"
        "- The persuadee has explicitly refused.\n"
        "- The persuadee remains neutral about donating.\n"
        "- The persuadee has a positive attitude towards donating but hasn't "
        "decided yet.\n"
        "- The persuadee has decided to donate."
    )


class DialogXpertP4GJudge:
    """
    Wraps an LLMClient and multiplexes N samples to aggregate a stable label.
    """

    def __init__(
        self,
        llm: LLMClient,
        success_threshold: float = 0.6,
        num_samples: int = 10,
        temperature: float = 1.1,
        max_tokens: int = 16,
        max_history_lines: int = 20,
    ):
        self.llm = llm
        self.success_threshold = float(success_threshold)
        self.num_samples = int(num_samples)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_history_lines = int(max_history_lines)

    def score(
        self,
        history_lines: List[str],
        assistant_action: str,
        user_reply: str,
        case_meta: Optional[Dict[str, Any]] = None,
    ) -> RewardOutput:
        prompt = _build_prompt(
            history_lines=history_lines,
            assistant_action=assistant_action,
            user_reply=user_reply,
            max_history_lines=self.max_history_lines,
        )
        try:
            judgments = self.llm.call_many(
                prompt,
                n=self.num_samples,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return _aggregate(judgments, success_threshold=self.success_threshold)
        except Exception as exc:  # noqa: BLE001
            fallback = score_user_reply_heuristic(user_reply)
            fallback.raw_judgments = [f"p4g_judge_fallback:{type(exc).__name__}:{exc}"]
            return fallback
