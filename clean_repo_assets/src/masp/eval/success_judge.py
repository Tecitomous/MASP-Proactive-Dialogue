"""Factory for task-specific DialogXpert-style success judges."""
from __future__ import annotations

from .craigslist_judge import DialogXpertCraigslistJudge
from .esconv_judge import DialogXpertESConvJudge
from .empathetic_judge import DialogXpertEmpatheticJudge
from .p4g_judge import DialogXpertP4GJudge
from ..utils.llm_client import LLMClient


def build_success_judge(
    task_name: str,
    llm: LLMClient,
    success_threshold: float,
    num_samples: int,
    temperature: float,
    max_tokens: int,
):
    task = (task_name or "p4g").lower().strip()
    if task == "esconv":
        return DialogXpertESConvJudge(
            llm=llm,
            success_threshold=success_threshold,
            num_samples=num_samples,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if task == "empathetic_dialogues":
        return DialogXpertEmpatheticJudge(
            llm=llm,
            success_threshold=success_threshold,
            num_samples=num_samples,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if task == "craigslist_bargain":
        return DialogXpertCraigslistJudge(
            llm=llm,
            success_threshold=success_threshold,
            num_samples=num_samples,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return DialogXpertP4GJudge(
        llm=llm,
        success_threshold=success_threshold,
        num_samples=num_samples,
        temperature=temperature,
        max_tokens=max_tokens,
    )
