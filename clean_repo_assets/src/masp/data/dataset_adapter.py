"""
Dataset adapter abstractions per MASP+OGR design §8.1.

The MASP core (mentalization, reward, rollout, PPO) must stay
task-agnostic: anything dataset-specific is funneled through a
`DatasetAdapter`. Currently registered:

    p4g                    — Persuasion-for-Good (has donation outcome)
    esconv                 — Emotional Support Conversation (no observed outcome)
    craigslist_bargain     — Craigslist Bargain (no observed outcome yet)
    empathetic_dialogues   — EmpatheticDialogues (no observed outcome)
    cima                   — tutoring dialogue benchmark (no observed outcome)

Generic on-disk format (all 4 datasets match this):

    {
      "dialogue_id": "...",
      "task": "p4g|esconv|...",          # optional
      "is_annotated": false,              # optional
      "dialogue": [
          {"speaker": "user|assistant|persuader|persuadee|...", "text": "..."},
          ...
      ],
      "donation": 0.0,                    # P4G only
      "meta": {...}                        # dataset-specific extras
    }

Backward compatibility
----------------------
Callers that imported `P4GSession` / `P4GTurn` / `load_p4g_sessions` from
`masp.data.p4g_loader` keep working — those names are now aliases
defined in `p4g_loader.py` for the same generic types/loader here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable


# Speaker normalisation — covers the role labels seen in the 4 raw datasets.
ASSISTANT_SPEAKERS = {
    "persuader", "assistant", "system", "supporter", "listener", "seller",
    "sys", "teacher", "tutor",
}
USER_SPEAKERS = {
    "persuadee", "user", "human", "help-seeker", "help_seeker", "speaker",
    "buyer", "usr", "student", "learner",
}


# ------------------------------------------------------------------ data model

@dataclass
class DialogueTurn:
    speaker: str  # 'assistant' | 'user' (post-normalisation)
    text: str


@dataclass
class DialogueSession:
    """Format-agnostic dialogue session.

    Outcome handling: `donation` is kept as a top-level field for P4G
    backward compatibility (default 0.0 means "no donation observed");
    `outcome` is the canonical adapter-facing dict — None when the
    dataset has no observed outcome at all.
    """
    session_id: str
    turns: List[DialogueTurn] = field(default_factory=list)
    task: str = ""
    is_annotated: bool = False
    donation: float = 0.0
    outcome: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def history_lines(self) -> List[str]:
        out = []
        for t in self.turns:
            prefix = "Assistant" if t.speaker == "assistant" else "User"
            out.append(f"{prefix}: {t.text}")
        return out


def _normalize_speaker(raw: str) -> Optional[str]:
    r = (raw or "").strip().lower()
    if r in ASSISTANT_SPEAKERS:
        return "assistant"
    if r in USER_SPEAKERS:
        return "user"
    return None


def load_dialogue_sessions(
    path: str,
    max_sessions: Optional[int] = None,
) -> List[DialogueSession]:
    """Format-agnostic loader. All 4 MASP datasets share this JSON schema."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sessions: List[DialogueSession] = []
    for i, item in enumerate(data):
        if max_sessions is not None and len(sessions) >= max_sessions:
            break
        sid = str(item.get("dialogue_id", f"session_{i}"))
        raw_turns = item.get("dialogue") or item.get("turns") or []
        turns: List[DialogueTurn] = []
        for t in raw_turns:
            spk = _normalize_speaker(str(t.get("speaker", "")))
            if spk is None:
                continue
            text = str(t.get("text", "")).strip()
            if not text:
                continue
            turns.append(DialogueTurn(speaker=spk, text=text))
        if not turns:
            continue
        donation = float(item.get("donation", 0.0) or 0.0)
        outcome: Optional[Dict[str, Any]] = None
        if "donation" in item:
            outcome = {"donation": donation}
        sessions.append(DialogueSession(
            session_id=sid,
            turns=turns,
            task=str(item.get("task", "")),
            is_annotated=bool(item.get("is_annotated", False)),
            donation=donation,
            outcome=outcome,
            meta=dict(item.get("meta") or {}),
        ))
    return sessions


# ---------------------------------------------------------------- episode seeds

@dataclass
class EpisodeSeed:
    """A starting point for a self-play / eval episode."""
    session_id: str
    history_lines: List[str] = field(default_factory=list)
    prev_user_text: str = ""
    donation_label: float = 0.0  # P4G outcome; 0.0 for non-outcome datasets


def build_episode_seeds(
    sessions: Sequence[DialogueSession],
    all_user_starts: bool = False,
    max_seeds: Optional[int] = None,
) -> List[EpisodeSeed]:
    """For each dialogue create one (or many) starting points.

    With `all_user_starts=True` we additionally yield a seed at every
    position immediately after a user turn (DialogXpert-style replay).
    """
    seeds: List[EpisodeSeed] = []
    for s in sessions:
        seeds.append(EpisodeSeed(
            session_id=s.session_id,
            history_lines=[],
            prev_user_text="",
            donation_label=s.donation,
        ))
        if all_user_starts:
            history: List[str] = []
            prev_user = ""
            for t in s.turns:
                line = ("Assistant: " if t.speaker == "assistant"
                        else "User: ") + t.text
                history.append(line)
                if t.speaker == "user":
                    prev_user = t.text
                    seeds.append(EpisodeSeed(
                        session_id=s.session_id,
                        history_lines=list(history),
                        prev_user_text=prev_user,
                        donation_label=s.donation,
                    ))
        if max_seeds is not None and len(seeds) >= max_seeds:
            break
    return seeds


# =============================================================== DatasetAdapter

@runtime_checkable
class DatasetAdapter(Protocol):
    """Adapter Protocol per MASP+OGR design §8.1.

    Hides dataset-specific I/O, profile / goal text, and outcome
    extraction from the rest of the pipeline. The MASP core algorithm
    only ever depends on this protocol.
    """
    task_name: str

    def load_sessions(self, path: str, max_sessions: Optional[int] = None) -> List[DialogueSession]: ...
    def build_profile_text(self, session: DialogueSession) -> str: ...
    def build_goal_text(self, session: DialogueSession) -> str: ...
    def has_observed_outcome(self, session: DialogueSession) -> bool: ...
    def get_outcome(self, session: DialogueSession) -> Optional[Dict[str, Any]]: ...


class _BaseAdapter:
    """Shared default implementation. Subclasses override `task_name`,
    `goal_text`, and outcome handling as needed."""

    task_name: str = ""
    goal_text: str = ""

    def load_sessions(
        self,
        path: str,
        max_sessions: Optional[int] = None,
    ) -> List[DialogueSession]:
        return load_dialogue_sessions(path, max_sessions=max_sessions)

    def build_profile_text(self, session: DialogueSession) -> str:
        return ""

    def build_goal_text(self, session: DialogueSession) -> str:
        return self.goal_text

    def has_observed_outcome(self, session: DialogueSession) -> bool:
        return session.outcome is not None

    def get_outcome(self, session: DialogueSession) -> Optional[Dict[str, Any]]:
        return session.outcome


# ----- concrete adapters -----------------------------------------------------

class P4GAdapter(_BaseAdapter):
    task_name = "p4g"
    goal_text = (
        "Persuade the user to make a non-zero donation to Save the Children, "
        "while remaining honest, respectful, and non-coercive."
    )

    def has_observed_outcome(self, session: DialogueSession) -> bool:
        return session.outcome is not None and "donation" in session.outcome


class ESConvAdapter(_BaseAdapter):
    task_name = "esconv"
    goal_text = (
        "Provide emotional support to the help-seeker, helping them feel "
        "understood and identifying coping strategies for their distress."
    )

    def has_observed_outcome(self, session: DialogueSession) -> bool:
        return False  # ESConv has no per-session observed outcome in current splits


class CraigslistBargainAdapter(_BaseAdapter):
    task_name = "craigslist_bargain"
    goal_text = (
        "Reach an agreement on the item's price that satisfies both buyer "
        "and seller within their respective targets."
    )

    def has_observed_outcome(self, session: DialogueSession) -> bool:
        # An outcome (final agreed price vs. targets) could be derived from
        # `meta.agent_info`, but no observed-price label is materialised yet.
        return False


class EmpatheticDialoguesAdapter(_BaseAdapter):
    task_name = "empathetic_dialogues"
    goal_text = (
        "Listen empathetically and acknowledge the speaker's emotion."
    )

    def has_observed_outcome(self, session: DialogueSession) -> bool:
        return False


class CIMAAdapter(_BaseAdapter):
    task_name = "cima"
    goal_text = (
        "Tutor the student toward the correct Italian translation while "
        "using helpful pedagogical strategies such as hints, questions, "
        "corrections, and confirmations."
    )

    def has_observed_outcome(self, session: DialogueSession) -> bool:
        return False


# ----- registry --------------------------------------------------------------

_ADAPTERS: Dict[str, DatasetAdapter] = {
    "p4g": P4GAdapter(),
    "esconv": ESConvAdapter(),
    "craigslist_bargain": CraigslistBargainAdapter(),
    "empathetic_dialogues": EmpatheticDialoguesAdapter(),
    "cima": CIMAAdapter(),
}


def get_adapter(task_name: str) -> DatasetAdapter:
    """Look up the adapter for a task. Raises KeyError on unknown task."""
    key = (task_name or "").lower().strip()
    if key not in _ADAPTERS:
        raise KeyError(
            f"unknown task {task_name!r}; registered adapters: "
            f"{sorted(_ADAPTERS.keys())}"
        )
    return _ADAPTERS[key]


def register_adapter(adapter: DatasetAdapter) -> None:
    """Register a custom adapter for a new dataset."""
    _ADAPTERS[adapter.task_name] = adapter


def list_adapters() -> List[str]:
    return sorted(_ADAPTERS.keys())
