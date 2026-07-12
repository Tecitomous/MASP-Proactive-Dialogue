#!/usr/bin/env python3
"""Lightweight DPO continuation for P4G close timing.

Pairs are mined from P4G train dialogues with donation > 0. The chosen response
is a real successful donation close. The rejected response is a safe but
non-closing continuation. This tests whether preference training can teach
phase2_closure to close earlier without positive-only SFT drift.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from masp.models.policy import (
    LoRAPolicy,
    PolicyConfig,
    build_chat_prompt_for_policy,
    infer_lora_config_from_adapter,
)
from masp.utils.io import dump_json, ensure_dir
from masp.utils.seed import set_seed
from train_p4g_success_sft import (
    _BAD_TEXT_RE,
    _GOOD_CLOSE_RE,
    SuccessSFTExample,
    mine_dialogue_examples,
)


@dataclass
class ClosePreferencePair:
    session_id: str
    source_path: str
    assistant_turn: int
    history_lines: List[str]
    chosen: str
    rejected: str
    ref_chosen_logp: Optional[float] = None
    ref_rejected_logp: Optional[float] = None


def _tokenize_target(tokenizer, text: str, max_target_len: int) -> torch.Tensor:
    ids = tokenizer(
        text or "",
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


def _generic_rejected(ex: SuccessSFTExample) -> str:
    last_user = ""
    for line in reversed(ex.history_lines):
        if str(line).startswith("User:"):
            last_user = str(line)[5:].strip().lower()
            break
    if any(word in last_user for word in ("what", "how", "where", "who", "tell me", "?")):
        return (
            "Save the Children supports children through health, education, "
            "nutrition, and emergency relief programs around the world."
        )
    return (
        "Save the Children is an important charity that helps children with "
        "health, education, safety, and emergency support around the world."
    )


def build_pairs(args) -> List[ClosePreferencePair]:
    examples = mine_dialogue_examples(
        args.dialogue_path,
        max_assistant_turn=args.max_success_turn,
        include_nonfinal_asks=bool(args.include_nonfinal_asks),
        max_examples_per_dialogue=args.max_examples_per_dialogue,
    )
    pairs: List[ClosePreferencePair] = []
    for ex in examples:
        rejected = _generic_rejected(ex)
        if rejected.strip() == ex.target_text.strip():
            continue
        pairs.append(ClosePreferencePair(
            session_id=ex.session_id,
            source_path=ex.source_path,
            assistant_turn=ex.assistant_turn,
            history_lines=ex.history_lines,
            chosen=ex.target_text,
            rejected=rejected,
        ))
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]
    if not pairs:
        raise RuntimeError("no DPO pairs after filtering")
    return pairs


def _messages(pair: ClosePreferencePair, task: str) -> List[Dict[str, str]]:
    return build_chat_prompt_for_policy(
        role="assistant",
        history_lines=pair.history_lines,
        task_name=task,
    )


@torch.no_grad()
def fill_generated_rejected(
    policy: LoRAPolicy,
    pairs: List[ClosePreferencePair],
    args,
) -> List[ClosePreferencePair]:
    policy.eval_mode()
    kept: List[ClosePreferencePair] = []
    for pair in tqdm(pairs, desc="[close-dpo] generated rejected"):
        out = policy.generate(
            _messages(pair, args.task),
            max_new_tokens=args.gen_max_new_tokens,
            temperature=args.gen_temperature,
            top_p=args.gen_top_p,
            do_sample=True,
        )
        text = str(out.get("text", "")).strip()
        if not text:
            continue
        # Keep real policy failures: no clear close, or an unsafe/bad close.
        # If the current policy already produces a clean close, this context is
        # not useful for DPO.
        if _GOOD_CLOSE_RE.search(text) and not _BAD_TEXT_RE.search(text):
            continue
        if text.strip() == pair.chosen.strip():
            continue
        pair.rejected = text
        kept.append(pair)
    return kept


def _mean_logp(policy: LoRAPolicy, pair: ClosePreferencePair, text: str, max_target_len: int, task: str) -> torch.Tensor:
    ids = _tokenize_target(policy.tokenizer, text, max_target_len=max_target_len)
    logp, mask = policy.log_probs_of_response(_messages(pair, task), ids)
    valid = mask.float()
    return (logp * valid).sum() / valid.sum().clamp(min=1.0)


@torch.no_grad()
def precompute_ref_logps(policy: LoRAPolicy, pairs: List[ClosePreferencePair], args) -> None:
    policy.eval_mode()
    for pair in tqdm(pairs, desc="[close-dpo] ref logp"):
        pair.ref_chosen_logp = float(_mean_logp(
            policy, pair, pair.chosen, args.max_target_len, args.task
        ).detach().item())
        pair.ref_rejected_logp = float(_mean_logp(
            policy, pair, pair.rejected, args.max_target_len, args.task
        ).detach().item())


def train(args) -> Dict:
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    pairs = build_pairs(args)
    dump_json([asdict(p) for p in pairs], os.path.join(args.out_dir, "dpo_pairs_unscored.json"))

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
        f"[close-dpo] loading policy base={args.model_path} init={args.init_adapter} "
        f"pairs={len(pairs)} device={args.device}"
    )
    policy = LoRAPolicy(cfg)
    policy.load_adapter(args.init_adapter)
    if args.rejected_source == "generated":
        pairs = fill_generated_rejected(policy, pairs, args)
        if int(args.max_pairs) > 0:
            pairs = pairs[: int(args.max_pairs)]
        if not pairs:
            raise RuntimeError("no generated-rejected DPO pairs after filtering")
        print(f"[close-dpo] generated-rejected pairs kept={len(pairs)}")
    precompute_ref_logps(policy, pairs, args)
    dump_json([asdict(p) for p in pairs], os.path.join(args.out_dir, "dpo_pairs.json"))

    policy.train_mode()
    opt = torch.optim.AdamW(
        policy.trainable_parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )
    rng = random.Random(args.seed)
    beta = float(args.beta)
    batch_size = max(int(args.batch_size), 1)
    losses: List[float] = []
    accs: List[float] = []
    global_step = 0
    t0 = time.perf_counter()

    for epoch in range(int(args.num_epochs)):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        pbar = tqdm(
            range(0, len(order), batch_size),
            desc=f"[close-dpo] ep {epoch + 1}/{args.num_epochs}",
        )
        for start in pbar:
            idxs = order[start: start + batch_size]
            opt.zero_grad(set_to_none=True)
            batch_losses: List[torch.Tensor] = []
            batch_acc = 0.0
            for idx in idxs:
                pair = pairs[idx]
                pi_c = _mean_logp(policy, pair, pair.chosen, args.max_target_len, args.task)
                pi_r = _mean_logp(policy, pair, pair.rejected, args.max_target_len, args.task)
                ref_c = torch.tensor(float(pair.ref_chosen_logp), device=pi_c.device)
                ref_r = torch.tensor(float(pair.ref_rejected_logp), device=pi_c.device)
                logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))
                batch_losses.append(-F.logsigmoid(logits))
                batch_acc += float((pi_c > pi_r).detach().item())
            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).mean()
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), float(args.max_grad_norm))
            opt.step()
            global_step += 1
            losses.append(float(loss.detach().item()))
            accs.append(batch_acc / float(len(idxs)))
            if global_step % max(int(args.log_every), 1) == 0:
                pbar.set_postfix({
                    "loss": f"{losses[-1]:.4f}",
                    "acc": f"{sum(accs[-20:]) / min(len(accs), 20):.3f}",
                })

    save_dir = os.path.join(args.out_dir, "pi_S")
    ensure_dir(save_dir)
    policy.save_adapter(save_dir)
    summary = {
        "task": args.task,
        "init_adapter": args.init_adapter,
        "save_dir": save_dir,
        "pairs": len(pairs),
        "epochs": int(args.num_epochs),
        "global_steps": int(global_step),
        "final_loss": float(losses[-1]) if losses else None,
        "mean_loss": float(sum(losses) / len(losses)) if losses else None,
        "mean_pref_acc": float(sum(accs) / len(accs)) if accs else None,
        "elapsed_sec": float(time.perf_counter() - t0),
        "args": vars(args),
    }
    dump_json(summary, os.path.join(args.out_dir, "close_dpo_log.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dialogue_path", action="append", required=True)
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
    p.add_argument("--include_nonfinal_asks", type=int, default=1)
    p.add_argument("--max_examples_per_dialogue", type=int, default=2)
    p.add_argument("--max_pairs", type=int, default=0)
    p.add_argument("--rejected_source", default="template",
                   choices=["template", "generated"])
    p.add_argument("--gen_temperature", type=float, default=0.75)
    p.add_argument("--gen_top_p", type=float, default=0.95)
    p.add_argument("--gen_max_new_tokens", type=int, default=96)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
