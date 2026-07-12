"""DialogXpert-aligned success critic for EmpatheticDialogues."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .p4g_judge import RewardOutput
from ..utils.llm_client import LLMClient


EMPATHETIC_REWARD_DICT: Dict[str, float] = {
    "worse": -1.0,
    "same": -0.5,
    "somewhat_better": 0.5,
    "resolved": 1.0,
}


def score_empathetic_heuristic(
    user_reply: str,
    assistant_action: str = "",
) -> RewardOutput:
    """Cheap local scorer for fast empathetic-dialogue phase2 training."""
    text = (user_reply or "").strip().lower()
    action = (assistant_action or "").strip().lower()
    worse = (
        "you don't understand", "you do not understand", "not really",
        "that's not it", "that is not it", "whatever", "never mind",
        "i don't want to talk", "i do not want to talk", "no thanks",
        "worse", "more upset", "hurt",
    )
    same = (
        "same", "still", "not sure", "i don't know", "i do not know",
        "nothing changed", "not much",
    )
    better = (
        "thank", "thanks", "yeah", "yes", "exactly", "that's true",
        "that is true", "i know", "i felt", "i feel", "it was",
        "it is", "i remember", "i guess", "i appreciate",
        "you understand", "that means", "it meant", "it made me",
    )
    resolved = (
        "i feel heard", "you get it", "you understand", "i feel better",
        "i feel calmer", "i'm okay", "i am okay", "relieved",
        "that helps", "that helped", "i can move on", "clear now",
    )
    empathy = (
        "sounds", "understand", "makes sense", "that must", "i can see",
        "i hear", "sorry", "proud", "scary", "hard", "sweet",
        "exciting", "frustrating", "lonely", "valid", "meaningful",
    )
    question_or_space = ("?", "tell me", "what was", "how did", "do you")
    worse_hits = sum(1 for cue in worse if cue in text)
    same_hits = sum(1 for cue in same if cue in text)
    better_hits = sum(1 for cue in better if cue in text)
    resolved_hits = sum(1 for cue in resolved if cue in text)
    empathy_hits = sum(1 for cue in empathy if cue in action)
    space_hits = sum(1 for cue in question_or_space if cue in action)
    if (
        resolved_hits > 0
        and resolved_hits >= worse_hits + same_hits
        and empathy_hits > 0
        and (space_hits > 0 or len(text.split()) >= 6)
    ):
        return RewardOutput(
            reward=1.0,
            success=True,
            label="resolved",
            pos_hits=better_hits + resolved_hits,
            neg_hits=worse_hits + same_hits,
        )
    if better_hits > 0 and better_hits + resolved_hits >= worse_hits + same_hits:
        return RewardOutput(
            reward=0.5,
            success=True,
            label="somewhat_better",
            pos_hits=better_hits + resolved_hits,
            neg_hits=worse_hits + same_hits,
        )
    label = "worse" if worse_hits > 0 and worse_hits >= same_hits else "same"
    reward = -1.0 if label == "worse" else -0.5
    return RewardOutput(
        reward=float(reward),
        success=False,
        label=label,
        pos_hits=better_hits + resolved_hits,
        neg_hits=worse_hits + same_hits,
    )


def _normalize_label(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    if not t:
        return None
    t = t.replace("-", "_")
    token = t.split()[0].strip(".,:;!?()[]{}\"'")
    if token in EMPATHETIC_REWARD_DICT:
        return token
    if (
        "resolved" in t
        or "solved" in t
        or "emotionally connected" in t
        or "feels heard" in t
    ):
        return "resolved"
    if (
        token == "better"
        or "somewhat_better" in t
        or "somewhat better" in t
        or "feels better" in t
        or "speaker feels better" in t
        or "validated" in t
    ):
        return "somewhat_better"
    if "better" in t and "somewhat" in t:
        return "somewhat_better"
    if "worse" in t or "dismissed" in t or "misunderstood" in t or "shut down" in t:
        return "worse"
    if (
        "same" in t
        or "does not feel heard" in t
        or "unheard" in t
        or "generic" in t
        or "unchanged" in t
    ):
        return "same"
    return None


def _aggregate(judgments: List[str], success_threshold: float) -> RewardOutput:
    labels: List[str] = []
    rewards: List[float] = []
    for item in judgments:
        label = _normalize_label(item)
        if label is None:
            continue
        labels.append(label)
        rewards.append(float(EMPATHETIC_REWARD_DICT[label]))

    reward = float(sum(rewards) / len(rewards)) if rewards else 0.0
    success = bool(reward >= float(success_threshold))
    if labels:
        order = ["worse", "same", "somewhat_better", "resolved"]
        label = max(order, key=lambda x: labels.count(x))
    else:
        label = "unknown"
    return RewardOutput(
        reward=reward,
        success=success,
        label=label,
        raw_judgments=list(judgments),
        pos_hits=int(labels.count("somewhat_better") + labels.count("resolved")),
        neg_hits=int(labels.count("worse") + labels.count("same")),
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
        "You are an expert evaluator of empathetic listening conversations.\n"
        "Use the same four-level emotional-outcome reward style as "
        "PPDPP/DialogXpert: worse, same, somewhat_better, resolved. For this "
        "dataset, resolved means the speaker's immediate emotional need in the "
        "exchange is resolved: they show clear evidence of feeling accurately "
        "heard, validated, emotionally steadier, and willing to continue. A "
        "generic sympathetic response is at most somewhat_better.\n\n"
        "Conversation so far:\n"
        f"{conv}\n"
        f"Listener: {assistant_action}\n"
        f"Speaker: {user_reply}\n\n"
        "Question: Has the speaker's emotional state improved or been resolved?\n"
        "Reward mapping: worse=-1.0, same=-0.5, somewhat_better=0.5, "
        "resolved=1.0.\n\n"
        "Respond with ONE exact label only:\n"
        "worse\n"
        "same\n"
        "somewhat_better\n"
        "resolved"
    )


class DialogXpertEmpatheticJudge:
    """LLM-backed EmpatheticDialogues success judge."""

    def __init__(
        self,
        llm: LLMClient,
        success_threshold: float = 0.5,
        num_samples: int = 5,
        temperature: float = 1.0,
        max_tokens: int = 24,
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
        backend = str(getattr(getattr(self.llm, "cfg", None), "backend", "")).lower()
        if backend in {"heuristic", "none", "local_heuristic"}:
            out = score_empathetic_heuristic(user_reply, assistant_action=assistant_action)
            out.raw_judgments = [f"empathetic_heuristic:{out.label}"]
            return out
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
            fallback = score_empathetic_heuristic(user_reply, assistant_action)
            fallback.raw_judgments = [f"empathetic_judge_fallback:{type(exc).__name__}:{exc}"]
            return fallback
