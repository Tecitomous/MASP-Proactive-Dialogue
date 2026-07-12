"""GRPO/DDP-only Phase 2 entrypoint.

This wrapper keeps the shared Phase 2 training code untouched while adding a
small runtime guard for GRPO rollouts: generated turns can occasionally decode
to an empty string, and Qwen3's tokenizer can produce a zero-token encoder
input for that text. The guard maps empty/whitespace text to a short sentinel
before sentence encoding.
"""
from __future__ import annotations

from typing import Sequence

import torch

from masp.models.sentence_encoder import SentenceEncoder


_ORIG_SENTENCE_ENCODE = SentenceEncoder.encode


@torch.no_grad()
def _encode_with_nonempty_text(self: SentenceEncoder, texts: Sequence[str]) -> torch.Tensor:
    if not texts:
        return _ORIG_SENTENCE_ENCODE(self, texts)
    cleaned = [
        text if isinstance(text, str) and text.strip() else "(empty response)"
        for text in texts
    ]
    return _ORIG_SENTENCE_ENCODE(self, cleaned)


SentenceEncoder.encode = _encode_with_nonempty_text


from train_phase2_selfplay import main as _phase2_main  # noqa: E402


def main() -> None:
    print("[phase2-grpo-ddp] empty-text sentence encoder guard enabled", flush=True)
    _phase2_main()


if __name__ == "__main__":
    main()
