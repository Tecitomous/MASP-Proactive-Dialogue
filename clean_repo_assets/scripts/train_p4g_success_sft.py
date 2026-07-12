#!/usr/bin/env python3
"""Continuation SFT for P4G short successful close turns.

This script trains only the assistant LoRA. It is intentionally narrow:
examples are mined from already audited evaluation episode dumps, filtered to
successful and rational short trajectories, then prompted with the same
no-BDI-hint assistant prompt used by the current P4G eval protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from masp.data.dataset_adapter import get_adapter
from masp.models.policy import (
    LoRAPolicy,
    PolicyConfig,
    build_chat_prompt_for_policy,
    infer_lora_config_from_adapter,
)
from masp.utils.io import dump_json, ensure_dir
from masp.utils.seed import set_seed


_DONATION_ASK_RE = re.compile(
    r"("
    r"\b(?:would|will|could|can)\s+you\b.{0,160}\b(?:donat|contribut)|"
    r"\b(?:willing|able)\b.{0,120}\b(?:donat|contribut)|"
    r"\bhow much\b.{0,100}\bdonat|"
    r"\bwhat amount\b.{0,100}\bdonat|"
    r"\bdonate\b.{0,100}\b(?:small|non-zero|any amount|\$|cent|task payment)"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_BAD_TEXT_RE = re.compile(
    r"\b("
    r"put you down|mark you down|sign you up|count you in|"
    r"take (?:your )?refusal as (?:a )?yes|need to get this done|"
    r"complete this task|need .*turns?|don'?t you want to|poor needy|"
    r"no con to donating|all (?:of )?the money goes|guarantee|"
    r"tax deductible|must ask|won't continue past this point|"
    r"need your donation more than you do|so you are willing|"
    r"how much do you like|top rated|100%|every penny|"
    r"you have to|we have to each donate"
    r")\b|"
    r"\bURL\b|https?://\S*URL\S*",
    re.IGNORECASE,
)
_GOOD_CLOSE_RE = re.compile(
    r"\b("
    r"would\s+you\s+be\s+willing|would\s+you\s+donate|"
    r"will\s+you\s+donate|could\s+you\s+donate|can\s+you\s+donate|"
    r"what\s+amount\s+would\s+you|how\s+much\s+would\s+you"
    r")\b",
    re.IGNORECASE,
)
_REFUSAL_RE = re.compile(
    r"\b(no thank|not donate|don'?t want|do not want|would not|will not|"
    r"not interested|choose 0|\$0|0 cents?|0 dollars|nothing|nope|nah|"
    r"not today|not now|maybe later|another time|in the future|"
    r"not ready|already donate|donated so much|numerous donations|"
    r"prefer local|local charities|prefer cash|need the money|"
    r"can'?t help|cannot help|still no|again no)\b",
    re.IGNORECASE,
)


@dataclass
class SuccessSFTExample:
    source_path: str
    session_id: str
    episode_idx: int
    assistant_turn: int
    success_turn: int
    avg_rationality: float
    progress_final: float
    history_lines: List[str]
    target_text: str


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokenize_target(tokenizer, target_text: str, max_target_len: int) -> torch.Tensor:
    ids = tokenizer(
        target_text or "",
        truncation=True,
        max_length=max(max_target_len - 1, 1),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    if tokenizer.eos_token_id is not None:
        eos = int(tokenizer.eos_token_id)
        if ids.numel() == 0 or int(ids[-1].item()) != eos:
            ids = torch.cat([ids, torch.tensor([eos], dtype=ids.dtype)], dim=0)
    return ids.long()


def _assistant_turns(history_lines: Sequence[str]) -> Iterable[Tuple[int, int, str]]:
    turn = 0
    for idx, line in enumerate(history_lines):
        if str(line).startswith("Assistant:"):
            turn += 1
            yield idx, turn, str(line)[10:].strip()


def _history_has_recent_refusal(history_lines: Sequence[str], max_lines: int = 4) -> bool:
    for line in history_lines[-max_lines:]:
        if str(line).startswith("User:") and _REFUSAL_RE.search(str(line)):
            return True
    return False


def _episode_turn_count(ep: Dict) -> int:
    value = ep.get("success_turn")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    value = ep.get("turns")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    bundles = ep.get("reward_bundles")
    if isinstance(bundles, list):
        return len(bundles)
    if isinstance(value, list):
        return len(value)
    return 99


def mine_examples(
    episode_paths: Sequence[str],
    *,
    max_success_turn: int,
    min_avg_rationality: float,
    min_progress: float,
    include_nonfinal_asks: bool,
    max_examples_per_episode: int,
) -> List[SuccessSFTExample]:
    examples: List[SuccessSFTExample] = []
    seen = set()
    for path in episode_paths:
        episodes = _load_json(path)
        if not isinstance(episodes, list):
            raise ValueError(f"{path} must be a list of episode dicts")
        for ep_idx, ep in enumerate(episodes):
            if not bool(ep.get("success")):
                continue
            success_turn = _episode_turn_count(ep)
            if success_turn > int(max_success_turn):
                continue
            avg_rat = float(ep.get("avg_rationality", 0.0) or 0.0)
            if avg_rat < float(min_avg_rationality):
                continue
            progress = float(ep.get("progress_final", 0.0) or 0.0)
            if progress < float(min_progress):
                continue
            hist = list(ep.get("history_lines", []))
            ep_examples: List[SuccessSFTExample] = []
            for line_idx, turn, text in _assistant_turns(hist):
                if turn > success_turn:
                    continue
                if not include_nonfinal_asks and turn != success_turn:
                    continue
                if not _DONATION_ASK_RE.search(text):
                    continue
                if not _GOOD_CLOSE_RE.search(text):
                    continue
                if _BAD_TEXT_RE.search(text):
                    continue
                pre_hist = hist[:line_idx]
                # Avoid training the model to keep pressing immediately after a
                # very recent refusal; eval-time repairs handle those separately.
                if _history_has_recent_refusal(pre_hist):
                    continue
                key = ("\n".join(pre_hist[-20:]), text)
                if key in seen:
                    continue
                seen.add(key)
                ep_examples.append(SuccessSFTExample(
                    source_path=path,
                    session_id=str(ep.get("session_id", ep_idx)),
                    episode_idx=int(ep_idx),
                    assistant_turn=int(turn),
                    success_turn=int(success_turn),
                    avg_rationality=float(avg_rat),
                    progress_final=float(progress),
                    history_lines=list(pre_hist[-30:]),
                    target_text=text,
                ))
            ep_examples.sort(
                key=lambda x: (
                    x.assistant_turn != x.success_turn,
                    x.assistant_turn,
                )
            )
            examples.extend(ep_examples[: max(int(max_examples_per_episode), 1)])
    return examples


def mine_dialogue_examples(
    dialogue_paths: Sequence[str],
    *,
    max_assistant_turn: int,
    include_nonfinal_asks: bool,
    max_examples_per_dialogue: int,
) -> List[SuccessSFTExample]:
    adapter = get_adapter("p4g")
    examples: List[SuccessSFTExample] = []
    seen = set()
    for path in dialogue_paths:
        sessions = adapter.load_sessions(path)
        for s_idx, session in enumerate(sessions):
            if float(session.donation or 0.0) <= 0.0:
                continue
            history: List[str] = []
            assistant_turn = 0
            dial_examples: List[SuccessSFTExample] = []
            for turn_idx, turn in enumerate(session.turns):
                if turn.speaker == "assistant":
                    assistant_turn += 1
                    text = str(turn.text or "").strip()
                    if assistant_turn <= int(max_assistant_turn):
                        if _DONATION_ASK_RE.search(text) and _GOOD_CLOSE_RE.search(text):
                            if not _BAD_TEXT_RE.search(text) and not _history_has_recent_refusal(history):
                                key = ("\n".join(history[-20:]), text)
                                if key not in seen:
                                    seen.add(key)
                                    dial_examples.append(SuccessSFTExample(
                                        source_path=path,
                                        session_id=session.session_id,
                                        episode_idx=int(s_idx),
                                        assistant_turn=int(assistant_turn),
                                        success_turn=int(assistant_turn),
                                        avg_rationality=1.0,
                                        progress_final=1.0,
                                        history_lines=list(history[-30:]),
                                        target_text=text,
                                    ))
                prefix = "Assistant" if turn.speaker == "assistant" else "User"
                history.append(f"{prefix}: {turn.text}")
            if not include_nonfinal_asks and dial_examples:
                dial_examples = [dial_examples[-1]]
            examples.extend(dial_examples[: max(int(max_examples_per_dialogue), 1)])
    return examples


def _loss_for_example(
    policy: LoRAPolicy,
    ex: SuccessSFTExample,
    max_target_len: int,
    task_name: str,
) -> Optional[torch.Tensor]:
    messages = build_chat_prompt_for_policy(
        role="assistant",
        history_lines=ex.history_lines,
        task_name=task_name,
    )
    response_ids = _tokenize_target(policy.tokenizer, ex.target_text, max_target_len)
    if response_ids.numel() == 0:
        return None
    logp, mask = policy.log_probs_of_response(messages, response_ids)
    if logp.numel() == 0:
        return None
    valid = mask.float()
    return -(logp * valid).sum() / valid.sum().clamp(min=1.0)


def train(args) -> Dict:
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    examples: List[SuccessSFTExample] = []
    if args.dialogue_path:
        examples.extend(mine_dialogue_examples(
            args.dialogue_path,
            max_assistant_turn=args.max_success_turn,
            include_nonfinal_asks=bool(args.include_nonfinal_asks),
            max_examples_per_dialogue=args.max_examples_per_episode,
        ))
    if args.episode_path:
        examples.extend(mine_examples(
            args.episode_path,
            max_success_turn=args.max_success_turn,
            min_avg_rationality=args.min_avg_rationality,
            min_progress=args.min_progress,
            include_nonfinal_asks=bool(args.include_nonfinal_asks),
            max_examples_per_episode=args.max_examples_per_episode,
        ))
    if int(args.max_examples) > 0:
        examples = examples[: int(args.max_examples)]
    if not examples:
        raise RuntimeError("no SFT examples after filtering")

    dump_json([asdict(ex) for ex in examples], os.path.join(args.out_dir, "sft_examples.json"))
    lora_kwargs = infer_lora_config_from_adapter(args.init_adapter)
    cfg = PolicyConfig(
        model_name_or_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=args.max_target_len,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        generation_use_cache=False,
        attn_implementation=args.attn_implementation,
        **lora_kwargs,
    )
    print(
        f"[success-sft] loading base={args.model_path} init={args.init_adapter} "
        f"device={args.device} examples={len(examples)}"
    )
    policy = LoRAPolicy(cfg)
    policy.load_adapter(args.init_adapter)
    policy.train_mode()
    opt = torch.optim.AdamW(
        policy.trainable_parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )
    rng = random.Random(args.seed)
    losses: List[float] = []
    t0 = time.perf_counter()
    batch_size = max(int(args.batch_size), 1)
    global_step = 0
    for epoch in range(int(args.num_epochs)):
        order = list(range(len(examples)))
        rng.shuffle(order)
        pbar = tqdm(
            range(0, len(order), batch_size),
            desc=f"[success-sft] ep {epoch + 1}/{args.num_epochs}",
        )
        for start in pbar:
            idxs = order[start: start + batch_size]
            opt.zero_grad(set_to_none=True)
            batch_loss = 0.0
            valid = 0
            for idx in idxs:
                loss = _loss_for_example(
                    policy,
                    examples[idx],
                    max_target_len=args.max_target_len,
                    task_name=args.task,
                )
                if loss is None or not torch.isfinite(loss):
                    continue
                (loss / float(len(idxs))).backward()
                batch_loss += float(loss.detach().item())
                valid += 1
            if valid == 0:
                continue
            torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), float(args.max_grad_norm))
            opt.step()
            global_step += 1
            mean_loss = batch_loss / float(valid)
            losses.append(mean_loss)
            if global_step % max(int(args.log_every), 1) == 0:
                pbar.set_postfix({
                    "loss": f"{mean_loss:.4f}",
                    "avg": f"{sum(losses[-20:]) / min(len(losses), 20):.4f}",
                })

    save_dir = os.path.join(args.out_dir, "pi_S")
    ensure_dir(save_dir)
    policy.save_adapter(save_dir)
    summary = {
        "task": args.task,
        "init_adapter": args.init_adapter,
        "save_dir": save_dir,
        "examples": len(examples),
        "epochs": int(args.num_epochs),
        "global_steps": int(global_step),
        "final_loss": float(losses[-1]) if losses else None,
        "mean_loss": float(sum(losses) / len(losses)) if losses else None,
        "elapsed_sec": float(time.perf_counter() - t0),
        "args": vars(args),
    }
    dump_json(summary, os.path.join(args.out_dir, "success_sft_log.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episode_path", action="append", default=[])
    p.add_argument("--dialogue_path", action="append", default=[],
                   help="P4G train/valid split JSON. Only dialogues with "
                        "observed donation > 0 are mined.")
    p.add_argument("--model_path", default="/path/to/base-model")
    p.add_argument("--init_adapter", default="checkpoints/phase2_closure/best/pi_S")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--task", default="p4g")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--max_prompt_len", type=int, default=2048)
    p.add_argument("--max_target_len", type=int, default=128)
    p.add_argument("--max_success_turn", type=int, default=5)
    p.add_argument("--min_avg_rationality", type=float, default=0.70)
    p.add_argument("--min_progress", type=float, default=0.0)
    p.add_argument("--include_nonfinal_asks", type=int, default=1)
    p.add_argument("--max_examples_per_episode", type=int, default=2)
    p.add_argument("--max_examples", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
