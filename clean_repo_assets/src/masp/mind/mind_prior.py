"""
Mind Prior p(z) — a non-parametric distribution over initial user BDIs.

For each real training dialogue we extract the BDI from the first N persuadee
turns (default N=2) and put it in a pool. Sampling from p(z) is a uniform pick
from this pool. Optionally we also keep a per-dialogue mapping so the self-play
loop can occasionally "replay" a known user profile for curriculum purposes.

This prior is a natural replacement for UDP's hand-defined 16 personas:
  (a) it is continuous in the embedding space,
  (b) it reflects real human heterogeneity from the original dataset,
  (c) it does not require a hand-written persona schema.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .bdi_schema import BDI


@dataclass
class MindPriorEntry:
    session_id: str
    bdi: BDI
    profile_text: str = ""  # short natural-language user profile / persona


class MindPrior:
    def __init__(self, entries: Optional[List[MindPriorEntry]] = None, seed: int = 0):
        self.entries: List[MindPriorEntry] = list(entries or [])
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: MindPriorEntry) -> None:
        self.entries.append(entry)

    def sample(self) -> MindPriorEntry:
        if not self.entries:
            raise RuntimeError("MindPrior is empty. Did you run extract_bdi_labels?")
        return self._rng.choice(self.entries)

    def to_jsonable(self) -> List[Dict]:
        return [
            {
                "session_id": e.session_id,
                "bdi": e.bdi.to_dict(),
                "profile_text": e.profile_text,
            }
            for e in self.entries
        ]

    @classmethod
    def from_jsonable(cls, items: Sequence[Dict], seed: int = 0) -> "MindPrior":
        entries = [
            MindPriorEntry(
                session_id=str(x.get("session_id", "")),
                bdi=BDI.from_dict(x["bdi"]),
                profile_text=str(x.get("profile_text", "")),
            )
            for x in items
        ]
        return cls(entries=entries, seed=seed)
