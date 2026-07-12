"""
Datasets that sit on top of the BDI label cache produced by `extract_bdi_labels.py`.

The label cache is a JSON file of the form:
    {
      "task": "p4g",
      "sessions": [
         {
           "session_id": "...",
           "turn_entries": [
              {
                 "turn_idx": 0,
                 "history_upto": [...lines...],
                 "bdi": {...},
                 "next_speaker": "assistant|user"
              },
              ...
           ],
           "initial_bdi": {...},
           "profile_text": "..."
         },
         ...
      ]
    }

We expose three derived datasets:

1. `BDITurnDataset` — (history_text, z_target) pairs used to pretrain the
   mentalization module in Phase 0.

2. `BCDataset` — (prompt, target) pairs used to BC warm-up the LoRA policies
   in Phase 1. Each BC example knows whether it is an assistant or a user
   turn so the two policies can be trained from disjoint subsets.

3. The mind-prior pool is built externally in `mind/mind_prior.py` from the
   `initial_bdi` entries.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

try:
    from torch.utils.data import Dataset
except ImportError:
    # Allow this file to import on machines without torch (e.g. Mac dev box
    # running offline data conversion). The Dataset subclasses can't be
    # instantiated without torch but BDILabelCache I/O still works.
    class Dataset:  # type: ignore
        pass

from ..mind.bdi_schema import BDI
from .dataset_adapter import DialogueSession
# Legacy alias — code below was written against this name.
P4GSession = DialogueSession


# ------------------------------------------------------------------------------
# label cache I/O
# ------------------------------------------------------------------------------

@dataclass
class BDITurnEntry:
    session_id: str
    turn_idx: int            # 0-based index in the full dialogue
    history_upto: List[str]  # history before this turn
    bdi: BDI                 # user's BDI at the moment just before this turn
    # speaker of the turn *just preceding* this hook point, useful for
    # diagnostics only.
    last_speaker: str = "none"
    # speaker of the upcoming turn at this hook point ('assistant'|'user').
    # This enables leakage-free BC lookup for user turns.
    next_speaker: str = "assistant"


@dataclass
class BDILabelCache:
    task: str
    entries: List[BDITurnEntry] = field(default_factory=list)
    initial_bdi: Dict[str, BDI] = field(default_factory=dict)  # session_id -> BDI
    profile_text: Dict[str, str] = field(default_factory=dict)

    # ---- serialization ----

    def to_jsonable(self) -> Dict:
        return {
            "task": self.task,
            "schema_version": "6dim",   # paper §3.2: (B, D, I, ρ, c, v)
            "sessions": [
                {
                    "session_id": sid,
                    "initial_bdi": self.initial_bdi[sid].to_dict(),
                    "profile_text": self.profile_text.get(sid, ""),
                }
                for sid in self.initial_bdi
            ],
            "turn_entries": [
                {
                    "session_id": e.session_id,
                    "turn_idx": int(e.turn_idx),
                    "history_upto": list(e.history_upto),
                    "bdi": e.bdi.to_dict(),
                    "last_speaker": e.last_speaker,
                    "next_speaker": e.next_speaker,
                }
                for e in self.entries
            ],
        }

    def save(self, path: str) -> None:
        """Atomically save the cache.

        BDI extraction can run for hours. Writing through a temp file plus
        os.replace keeps an interrupted incremental save from corrupting the
        last good cache on disk.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.to_jsonable(), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: str) -> "BDILabelCache":
        """
        Tolerant loader. Supports both:
          * old 3-field cache: {bdi: {belief, desire, intention}}        → ρ/c/v default to (0.5, 0.5, 0.0)
          * new 6-field cache: {bdi: {belief, desire, intention, receptivity, confidence, valence}}

        Print a one-time migration hint if old format is detected.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cache = cls(task=str(data.get("task", "p4g")))

        legacy = (str(data.get("schema_version", "")) != "6dim")

        for e in data.get("turn_entries", []):
            cache.entries.append(BDITurnEntry(
                session_id=str(e["session_id"]),
                turn_idx=int(e["turn_idx"]),
                history_upto=list(e["history_upto"]),
                bdi=BDI.from_dict(e["bdi"]),
                last_speaker=str(e.get("last_speaker", "none")),
                next_speaker=str(e.get("next_speaker", "assistant")),
            ))
        for s in data.get("sessions", []):
            sid = str(s["session_id"])
            cache.initial_bdi[sid] = BDI.from_dict(s["initial_bdi"])
            cache.profile_text[sid] = str(s.get("profile_text", ""))

        if legacy:
            print(
                f"[BDILabelCache] WARNING: {path} is in legacy 3-field format. "
                f"ρ/c/v defaulted to (0.5, 0.5, 0.0). To get real 6-dim labels, "
                f"either rerun extract_bdi_labels.py or use "
                f"persona_v2/migrate_cache_to_6dim.py for incremental upgrade."
            )
        return cache


# ------------------------------------------------------------------------------
# Phase 0 — mentalization supervised dataset
# ------------------------------------------------------------------------------

class BDITurnDataset(Dataset):
    """
    Yields (history_text, bdi_text) where:
        - history_text: newline-joined dialogue history up to turn k
        - bdi_text:     canonical string form of the target BDI
    The mentalization module will tokenize these strings itself.
    """

    def __init__(
        self,
        cache: BDILabelCache,
        max_history_lines: int = 30,
        next_speaker_filter: str = "assistant",
    ):
        if next_speaker_filter:
            self.entries = [
                e for e in cache.entries if (e.next_speaker == next_speaker_filter)
            ]
        else:
            self.entries = list(cache.entries)
        self.max_history_lines = int(max_history_lines)
        # Carry the per-session profile text through the dataset so the teacher
        # mentalizer (paper §3.2: F_ω(h, p_u, g)) can condition on it during
        # student-mimics-teacher training (eq 31).
        self._profile_text_by_sid: Dict[str, str] = dict(cache.profile_text)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        e = self.entries[idx]
        hist = e.history_upto[-self.max_history_lines:]
        history_text = "\n".join(hist) if hist else "(no turns yet)"
        return {
            "session_id": e.session_id,
            "turn_idx": int(e.turn_idx),
            "history_text": history_text,
            "profile_text": self._profile_text_by_sid.get(e.session_id, ""),
            "bdi_belief": e.bdi.belief,
            "bdi_desire": e.bdi.desire,
            "bdi_intention": e.bdi.intention,
            "bdi_receptivity": float(e.bdi.receptivity),
            "bdi_confidence": float(e.bdi.confidence),
            "bdi_valence": float(e.bdi.valence),
        }


def build_bdi_turn_dataset(
    cache: BDILabelCache,
    max_history_lines: int = 30,
    next_speaker_filter: str = "assistant",
) -> BDITurnDataset:
    return BDITurnDataset(
        cache,
        max_history_lines=max_history_lines,
        next_speaker_filter=next_speaker_filter,
    )


# ------------------------------------------------------------------------------
# Phase 1 — behavior cloning warm-up dataset
# ------------------------------------------------------------------------------

@dataclass
class BCExample:
    role: str                 # 'assistant' or 'user'
    history_lines: List[str]
    target_text: str
    bdi: Optional[BDI] = None  # only used for the user side


class BCDataset(Dataset):
    """
    Next-turn behavior cloning examples extracted from the raw P4G sessions.
    For user turns we also attach the BDI label (if available in the cache)
    so the user policy can be conditioned on its own "mind" during warmup.
    """

    def __init__(self, examples: List[BCExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        e = self.examples[idx]
        return {
            "role": e.role,
            "history_lines": list(e.history_lines),
            "target_text": e.target_text,
            "bdi_belief":      e.bdi.belief      if e.bdi is not None else "",
            "bdi_desire":      e.bdi.desire      if e.bdi is not None else "",
            "bdi_intention":   e.bdi.intention   if e.bdi is not None else "",
            "bdi_receptivity": float(e.bdi.receptivity) if e.bdi is not None else 0.5,
            "bdi_confidence":  float(e.bdi.confidence)  if e.bdi is not None else 0.5,
            "bdi_valence":     float(e.bdi.valence)     if e.bdi is not None else 0.0,
            "has_bdi": e.bdi is not None,
        }


def build_bc_dataset(
    sessions: Sequence[P4GSession],
    cache: Optional[BDILabelCache] = None,
    role: str = "both",  # 'assistant' | 'user' | 'both'
) -> BCDataset:
    examples: List[BCExample] = []
    want_assistant = role in {"assistant", "both"}
    want_user = role in {"user", "both"}

    # Build a fast lookup: (session_id, turn_idx, next_speaker) -> BDI
    #
    # We intentionally key by `next_speaker` to avoid leakage:
    # for a user BC target at index i we must condition on BDI inferred from
    # history *before* that user turn, not from history that already contains it.
    bdi_lookup: Dict[tuple, BDI] = {}
    if cache is not None:
        for e in cache.entries:
            bdi_lookup[(e.session_id, e.turn_idx, e.next_speaker)] = e.bdi

    for s in sessions:
        history: List[str] = []
        for idx, t in enumerate(s.turns):
            if t.speaker == "assistant" and want_assistant:
                examples.append(BCExample(
                    role="assistant",
                    history_lines=list(history),
                    target_text=t.text,
                    bdi=None,
                ))
            elif t.speaker == "user" and want_user:
                bdi = bdi_lookup.get((s.session_id, idx, "user"))
                if cache is not None and bdi is None:
                    raise RuntimeError(
                        "Missing pre-turn user BDI label for "
                        f"(session={s.session_id}, turn_idx={idx}). "
                        "Please regenerate cache with the updated "
                        "extract_bdi_labels.py."
                    )
                examples.append(BCExample(
                    role="user",
                    history_lines=list(history),
                    target_text=t.text,
                    bdi=bdi,
                ))
            prefix = "Assistant" if t.speaker == "assistant" else "User"
            history.append(f"{prefix}: {t.text}")
    return BCDataset(examples)
