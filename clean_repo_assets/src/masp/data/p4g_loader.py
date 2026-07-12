"""
P4G dataset loader — thin backward-compat shim over `dataset_adapter.py`.

All real logic now lives in `masp.data.dataset_adapter`. This module
exists so existing call sites that imported `P4GSession`, `P4GTurn`,
`load_p4g_sessions`, `EpisodeSeed`, `build_episode_seeds`,
`ASSISTANT_SPEAKERS`, `USER_SPEAKERS` keep working unchanged.

New code should prefer one of:

    from masp.data.dataset_adapter import (
        DialogueSession, DialogueTurn, load_dialogue_sessions, get_adapter,
    )
    sessions = get_adapter("p4g").load_sessions(path)        # adapter API
    sessions = load_dialogue_sessions(path)                  # raw API
"""
from __future__ import annotations

from typing import List, Optional

from .dataset_adapter import (
    ASSISTANT_SPEAKERS,
    USER_SPEAKERS,
    DialogueSession,
    DialogueTurn,
    EpisodeSeed,
    P4GAdapter,
    build_episode_seeds,
    get_adapter,
    load_dialogue_sessions,
)

# Names kept for backward compatibility with code written against the
# original P4G-only loader.
P4GSession = DialogueSession
P4GTurn = DialogueTurn


def load_p4g_sessions(
    path: str,
    max_sessions: Optional[int] = None,
) -> List[DialogueSession]:
    """Backward-compat alias. Equivalent to
    ``get_adapter("p4g").load_sessions(path, max_sessions)``.
    Format-agnostic — also works on esconv / craigslist_bargain /
    empathetic_dialogues files thanks to the shared schema."""
    return get_adapter("p4g").load_sessions(path, max_sessions=max_sessions)


__all__ = [
    "ASSISTANT_SPEAKERS",
    "USER_SPEAKERS",
    "DialogueSession",
    "DialogueTurn",
    "EpisodeSeed",
    "P4GAdapter",
    "P4GSession",
    "P4GTurn",
    "build_episode_seeds",
    "load_dialogue_sessions",
    "load_p4g_sessions",
]
