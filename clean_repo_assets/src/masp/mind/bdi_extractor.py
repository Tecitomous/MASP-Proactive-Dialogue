"""
Oracle Belief Updater (OBU) — extracts (B, D, I) from a dialogue history
via an LLM.

The same prompt template is used for every task. Task adaptation is routed
through the `task_description` and `extraction_context_hint` fields of
BDITaskConfig. No per-task slot engineering.

We intentionally ask the LLM for EXACTLY three lines so that downstream
parsing is trivial and deterministic. If the LLM fails to obey, `BDI.from_text`
falls back to stuffing everything into the Belief field.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

from ..utils.llm_client import LLMClient
from .bdi_schema import BDI, BDITaskConfig


EXTRACTION_PROMPT_TEMPLATE = """\
You are an expert cognitive analyst. Your task is to infer a dialogue \
participant's CURRENT internal mental state from the conversation so far.

Task context:
{task_description}

Roles:
  Assistant = {assistant_role}
  User      = {user_role}

Dialogue so far (most recent last):
{history_block}

Infer the USER's mental state RIGHT NOW using these six dimensions:

  Belief       — What does the user currently believe to be true? Focus on \
beliefs REVEALED by their words and behavior, not what you think they \
"should" believe. If the user has said very little, their beliefs are \
largely unknown.

  Desire       — What does the user currently want? This should reflect \
THEIR goals, not the assistant's goals for them.

  Intention    — What is the user currently planning to do next? Base this \
on what they have actually expressed or implied, not on what a cooperative \
participant "would" do.

  Receptivity  — float in [0, 1]. How open is the user to the assistant's \
proposals RIGHT NOW, based on what they have actually said or implied?
    Calibration anchors:
      0.0-0.2  The user is actively resisting, dismissing, or hostile
      0.2-0.4  The user is skeptical, deflecting, or disengaged
      0.4-0.6  The user is neutral — neither resisting nor embracing
      0.6-0.8  The user is showing interest, asking questions, engaging
      0.8-1.0  The user is enthusiastically agreeing or already committed
    IMPORTANT: At the start of a conversation (0-2 turns), most users are \
in the 0.3-0.5 range (neutral/undecided) unless they explicitly show \
enthusiasm or hostility. Do NOT default to high values.

  Confidence   — float in [0, 1]. How certain/committed is the user in \
their CURRENT stance (whether that stance is positive or negative)?
    Calibration anchors:
      0.0-0.2  Very uncertain, hedging ("I'm not sure", "maybe")
      0.2-0.4  Leaning but open to change ("I guess", "probably")
      0.4-0.6  Moderate conviction, could go either way
      0.6-0.8  Fairly committed ("I think", "I believe", clear position)
      0.8-1.0  Extremely firm ("Absolutely not", "I'm certain")
    NOTE: Early in a conversation, confidence is typically LOW (0.2-0.5) \
because the user hasn't formed strong views yet.

  Valence      — float in [-1, 1]. The user's emotional tone right now.
    Calibration anchors:
      -1.0 to -0.5  Angry, frustrated, hostile
      -0.5 to -0.2  Annoyed, uncomfortable, mildly negative
      -0.2 to +0.2  Neutral, polite, matter-of-fact
      +0.2 to +0.5  Warm, friendly, mildly positive
      +0.5 to +1.0  Enthusiastic, excited, deeply moved

{hint}

CRITICAL RULES:
- Base your inference ONLY on what the user has actually said or clearly \
implied. Do not project or assume cooperative behavior.
- Scalar values MUST reflect the user's CURRENT state, not a desired end \
state. Users start conversations with diverse attitudes.
- Use the FULL range of each scale. Avoid clustering around any default.
- If the dialogue is very short or the user hasn't revealed much, use \
moderate/uncertain values (receptivity ~0.4, confidence ~0.3, valence ~0.0).

Output EXACTLY six lines. No bullets, no JSON, no explanation.

Belief: <one sentence>
Desire: <one sentence>
Intention: <one sentence>
Receptivity: <number>
Confidence: <number>
Valence: <number>
"""


def _format_history(history_lines: Sequence[str], max_lines: int = 30) -> str:
    lines = list(history_lines)[-int(max_lines):]
    return "\n".join(lines) if lines else "(no turns yet — beginning of conversation)"


class BDIExtractor:
    """
    Wraps an LLMClient with the universal extraction prompt. Supports caching
    at the (task, history-hash) level via an in-memory dict.
    """

    def __init__(
        self,
        llm: LLMClient,
        task_cfg: BDITaskConfig,
        temperature: float = 0.2,
        max_tokens: int = 320,    # 6-field schema needs a bit more headroom
        max_history_lines: int = 30,
        cache: Dict[str, BDI] | None = None,
    ):
        self.llm = llm
        self.task_cfg = task_cfg
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_history_lines = int(max_history_lines)
        self.cache: Dict[str, BDI] = cache if cache is not None else {}

    def _cache_key(self, history_lines: Sequence[str]) -> str:
        return self.task_cfg.task_name + "|" + "\n".join(history_lines[-20:])

    def build_prompt(self, history_lines: Sequence[str]) -> str:
        return EXTRACTION_PROMPT_TEMPLATE.format(
            task_description=self.task_cfg.task_description.strip(),
            assistant_role=self.task_cfg.assistant_role,
            user_role=self.task_cfg.user_role,
            history_block=_format_history(history_lines, self.max_history_lines),
            hint=self.task_cfg.extraction_context_hint.strip()
            or "None — use general common sense.",
        )

    def extract(self, history_lines: Sequence[str]) -> BDI:
        key = self._cache_key(history_lines)
        if key in self.cache:
            return self.cache[key]
        prompt = self.build_prompt(history_lines)
        raw = self.llm.call(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        bdi = BDI.from_text(raw)
        self.cache[key] = bdi
        return bdi

    def extract_batch(self, list_of_history_lines: List[Sequence[str]]) -> List[BDI]:
        return [self.extract(h) for h in list_of_history_lines]
