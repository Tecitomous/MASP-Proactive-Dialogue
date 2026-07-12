"""DialogXpert/PPDPP-aligned success critic for ESConv dialogues."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .p4g_judge import RewardOutput
from ..utils.llm_client import LLMClient


ESCONV_REWARD_DICT: Dict[str, float] = {
    "worse": -1.0,
    "same": -0.5,
    "somewhat_better": 0.5,
    "resolved": 1.0,
}


def score_esconv_heuristic(
    user_reply: str,
    assistant_action: str = "",
) -> RewardOutput:
    """Cheap ESConv scorer for fallback and fast local phase2 training."""
    text = (user_reply or "").strip().lower()
    action = (assistant_action or "").strip().lower()
    worse = (
        "not helpful", "doesn't help", "does not help", "worse",
        "more upset", "more anxious", "hopeless", "alone",
        "you don't understand", "you do not understand", "that won't help",
        "that will not help", "unsafe", "hurt myself",
    )
    same = (
        "still upset", "still anxious", "still scared", "still sad",
        "still overwhelmed", "same", "i don't know", "i do not know",
        "i can't", "i cannot", "not sure", "nothing changed",
    )
    better = (
        "thank", "thanks", "that helps", "helpful", "feel better",
        "i feel better", "i can try", "i will try", "i'll try",
        "that makes sense", "i can do that", "good idea",
        "i'll do that", "i will do that", "i can start",
        "i'll start", "i will start", "i can take",
    )
    resolved = (
        "much better", "a lot better", "relieved", "calmer now",
        "i feel calm", "i know what to do", "i can handle",
        "i can cope", "i'm okay", "i am okay", "clear now",
        "this resolves", "problem is solved", "issue is solved",
    )
    empathy = (
        "makes sense", "understand", "sorry", "heavy", "hard",
        "painful", "overwhelming", "valid", "not alone",
    )
    better_hits = sum(1 for cue in better if cue in text)
    resolved_hits = sum(1 for cue in resolved if cue in text)
    worse_hits = sum(1 for cue in worse if cue in text)
    same_hits = sum(1 for cue in same if cue in text)
    empathy_hits = sum(1 for cue in empathy if cue in action)
    if resolved_hits > 0 and resolved_hits >= worse_hits + same_hits and empathy_hits > 0:
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
            success=False,
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
    if token in ESCONV_REWARD_DICT:
        return token
    if "resolved" in t or "issue has been solved" in t or "issues have been solved" in t:
        return "resolved"
    if "solved" in t and "not" not in t:
        return "resolved"
    if (
        token == "better"
        or "somewhat_better" in t
        or "somewhat better" in t
        or "feels better" in t
        or "patient feels better" in t
        or "help_seeker feels better" in t
    ):
        return "somewhat_better"
    if "worse" in t or "unsupported" in t or "harmful" in t:
        return "worse"
    if "same" in t or "unchanged" in t or "remains distressed" in t:
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
        rewards.append(float(ESCONV_REWARD_DICT[label]))

    reward = float(sum(rewards) / len(rewards)) if rewards else 0.0
    # PPDPP Env.step uses `reward > 0.5` for ESConv completion. Keep the
    # threshold configurable, but make the comparison strict for this task.
    success = bool(reward > float(success_threshold))
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
    case_meta: Optional[Dict[str, Any]] = None,
) -> str:
    use_lines = history_lines[-max_history_lines:] if max_history_lines > 0 else history_lines
    role_lines: List[str] = []
    for line in use_lines:
        s = str(line)
        if s.startswith("Assistant:"):
            role_lines.append("Therapist:" + s[len("Assistant:"):])
        elif s.startswith("User:"):
            role_lines.append("Patient:" + s[len("User:"):])
        else:
            role_lines.append(s)
    role_lines.extend([
        f"Therapist: {assistant_action}",
        f"Patient: {user_reply}",
    ])
    conv = " ".join(role_lines)
    meta = case_meta or {}
    emotion_type = str(meta.get("emotion_type") or meta.get("emotion") or "the patient's emotional state")
    problem_type = str(meta.get("problem_type") or meta.get("problem") or "the patient's situation")
    return (
        "Given a conversation between a Therapist and a Patient, please assess "
        "whether the Patient' emotional issue has been solved after the conversation.\n"
        "You can only reply with one of the following sentences: No, the Patient "
        "feels worse. No, the Patient feels the same. No, but the Patient feels "
        "better. Yes, the Patient's issue has been solved.\n\n"
        f"The following is a conversation about {emotion_type} regarding {problem_type}: {conv}\n"
        "Question: Has the Patient's issue been solved? Answer: "
    )


class DialogXpertESConvJudge:
    """LLM-backed ESConv success judge with the same interface as P4G."""

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
            out = score_esconv_heuristic(user_reply, assistant_action=assistant_action)
            out.raw_judgments = [f"esconv_heuristic:{out.label}"]
            return out
        prompt = _build_prompt(
            history_lines=history_lines,
            assistant_action=assistant_action,
            user_reply=user_reply,
            max_history_lines=self.max_history_lines,
            case_meta=case_meta,
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
            fallback = score_esconv_heuristic(user_reply)
            fallback.raw_judgments = [f"esconv_judge_fallback:{type(exc).__name__}:{exc}"]
            return fallback
