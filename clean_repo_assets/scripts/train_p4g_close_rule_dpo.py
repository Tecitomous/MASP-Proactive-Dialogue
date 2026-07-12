#!/usr/bin/env python3
"""DPO distillation from the audited P4G close-inject rule.

This script builds preference pairs from train-split conversation contexts,
not from held-out eval dumps. For each assistant context it samples the current
policy response, applies the audited close-inject rule, and keeps the pair only
when the rule makes a clean change:

    chosen   = close-inject repaired response
    rejected = raw current-policy response

The goal is to move the behavior into pi_S itself without using test episodes
or hand-written train-human templates.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from evaluate_masp import (
    _BAD_CANDIDATE_RE,
    _DONATION_ASK_RE,
    _GOODBYE_CANDIDATE_RE,
    _inject_close_ask_if_needed,
    _recent_user_refusal_count,
)
from masp.data.dataset_adapter import get_adapter
from masp.models.policy import (
    LoRAPolicy,
    PolicyConfig,
    build_chat_prompt_for_policy,
    infer_lora_config_from_adapter,
)
from masp.utils.io import dump_json, ensure_dir
from masp.utils.seed import set_seed


_USER_COMMIT_RE = re.compile(
    r"("
    r"\b(?:i'?ll|i will|i can|i would|i'd|i am willing|i'?m willing|"
    r"happy to|sure|yes)\b.{0,100}\b(?:donat|contribut|give)|"
    r"\b(?:donat|contribut|give)\b.{0,60}\b(?:\$?\d|cents?|dollars?)"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_AMOUNT_RE = re.compile(r"(\$\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:cents?|dollars?)\b)", re.I)


@dataclass
class CloseRuleContext:
    session_id: str
    source_path: str
    dialogue_turn_index: int
    assistant_turn: int
    history_lines: List[str]


@dataclass
class CloseRulePreferencePair:
    session_id: str
    source_path: str
    dialogue_turn_index: int
    assistant_turn: int
    history_lines: List[str]
    chosen: str
    rejected: str
    ref_chosen_logp: Optional[float] = None
    ref_rejected_logp: Optional[float] = None


def _has_user_turn(history_lines: Sequence[str]) -> bool:
    return any(str(line).startswith("User:") for line in history_lines)


def _last_user_text(history_lines: Sequence[str]) -> str:
    for line in reversed(history_lines):
        if str(line).startswith("User:"):
            return str(line)[5:].strip()
    return ""


def _looks_like_user_already_committed(history_lines: Sequence[str]) -> bool:
    last_user = _last_user_text(history_lines)
    if not last_user:
        return False
    if "?" not in last_user and _AMOUNT_RE.search(last_user):
        return True
    return bool(_USER_COMMIT_RE.search(last_user))


def _iter_contexts(
    dialogue_paths: Sequence[str],
    *,
    min_assistant_turn: int,
    max_assistant_turn: int,
    skip_any_recent_refusal: bool,
    max_history_lines: int,
) -> Iterable[CloseRuleContext]:
    adapter = get_adapter("p4g")
    for path in dialogue_paths:
        sessions = adapter.load_sessions(path)
        for session in sessions:
            history: List[str] = []
            assistant_seen = 0
            for turn_idx, turn in enumerate(session.turns):
                if turn.speaker == "assistant":
                    next_assistant_turn = assistant_seen + 1
                    if (
                        next_assistant_turn >= int(min_assistant_turn)
                        and next_assistant_turn <= int(max_assistant_turn)
                        and _has_user_turn(history)
                    ):
                        if skip_any_recent_refusal and _recent_user_refusal_count(history) > 0:
                            pass
                        elif _looks_like_user_already_committed(history):
                            pass
                        else:
                            yield CloseRuleContext(
                                session_id=session.session_id,
                                source_path=path,
                                dialogue_turn_index=int(turn_idx),
                                assistant_turn=int(next_assistant_turn),
                                history_lines=list(history[-int(max_history_lines):]),
                            )
                    assistant_seen += 1
                prefix = "Assistant" if turn.speaker == "assistant" else "User"
                history.append(f"{prefix}: {turn.text}")


def build_contexts(args) -> List[CloseRuleContext]:
    contexts = list(_iter_contexts(
        args.dialogue_path,
        min_assistant_turn=args.min_assistant_turn,
        max_assistant_turn=args.max_assistant_turn,
        skip_any_recent_refusal=bool(args.skip_any_recent_refusal),
        max_history_lines=args.max_history_lines,
    ))
    rng = random.Random(args.seed)
    rng.shuffle(contexts)
    if int(args.max_contexts) > 0:
        contexts = contexts[: int(args.max_contexts)]
    if not contexts:
        raise RuntimeError("no train contexts after filtering")
    return contexts


def _messages(ctx: CloseRuleContext | CloseRulePreferencePair, task: str) -> List[Dict[str, str]]:
    return build_chat_prompt_for_policy(
        role="assistant",
        history_lines=ctx.history_lines,
        task_name=task,
    )


def _is_clean_pair(chosen: str, rejected: str) -> bool:
    c = (chosen or "").strip()
    r = (rejected or "").strip()
    if not c or not r or c == r:
        return False
    if not _DONATION_ASK_RE.search(c):
        return False
    if _BAD_CANDIDATE_RE.search(c):
        return False
    if _GOODBYE_CANDIDATE_RE.search(c) and not _DONATION_ASK_RE.search(c):
        return False
    return True


@torch.no_grad()
def generate_pairs(policy: LoRAPolicy, contexts: List[CloseRuleContext], args) -> List[CloseRulePreferencePair]:
    policy.eval_mode()
    pairs: List[CloseRulePreferencePair] = []
    pbar = tqdm(contexts, desc="[close-rule-dpo] pair gen")
    for ctx in pbar:
        batch = [_messages(ctx, args.task) for _ in range(max(int(args.candidates_per_context), 1))]
        outs = policy.generate_batch(
            batch,
            max_new_tokens=args.gen_max_new_tokens,
            temperature=args.gen_temperature,
            top_p=args.gen_top_p,
            do_sample=True,
        )
        kept_for_context = 0
        for out in outs:
            rejected = str(out.get("text", "")).strip()
            chosen = _inject_close_ask_if_needed(
                rejected,
                list(ctx.history_lines),
                task_name=args.task,
                min_turn=int(args.close_min_turn),
                style=args.close_style,
                refusal_stop_after=int(args.refusal_stop_after),
            ).strip()
            if not _is_clean_pair(chosen, rejected):
                continue
            pairs.append(CloseRulePreferencePair(
                session_id=ctx.session_id,
                source_path=ctx.source_path,
                dialogue_turn_index=ctx.dialogue_turn_index,
                assistant_turn=ctx.assistant_turn,
                history_lines=list(ctx.history_lines),
                chosen=chosen,
                rejected=rejected,
            ))
            kept_for_context += 1
            if int(args.max_pairs_per_context) > 0 and kept_for_context >= int(args.max_pairs_per_context):
                break
        pbar.set_postfix({"pairs": len(pairs)})
        if int(args.max_pairs) > 0 and len(pairs) >= int(args.max_pairs):
            pairs = pairs[: int(args.max_pairs)]
            break
    if not pairs:
        raise RuntimeError("no close-rule DPO pairs after generation/filtering")
    return pairs


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


def _mean_logp(
    policy: LoRAPolicy,
    pair: CloseRulePreferencePair,
    text: str,
    max_target_len: int,
    task: str,
) -> torch.Tensor:
    ids = _tokenize_target(policy.tokenizer, text, max_target_len=max_target_len)
    logp, mask = policy.log_probs_of_response(_messages(pair, task), ids)
    valid = mask.float()
    return (logp * valid).sum() / valid.sum().clamp(min=1.0)


@torch.no_grad()
def precompute_ref_logps(policy: LoRAPolicy, pairs: List[CloseRulePreferencePair], args) -> None:
    policy.eval_mode()
    for pair in tqdm(pairs, desc="[close-rule-dpo] ref logp"):
        pair.ref_chosen_logp = float(_mean_logp(
            policy, pair, pair.chosen, args.max_target_len, args.task,
        ).detach().item())
        pair.ref_rejected_logp = float(_mean_logp(
            policy, pair, pair.rejected, args.max_target_len, args.task,
        ).detach().item())


def _load_policy(args) -> LoRAPolicy:
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
    policy = LoRAPolicy(cfg)
    policy.load_adapter(args.init_adapter)
    return policy


def train(args) -> Dict:
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    contexts = build_contexts(args)
    dump_json([asdict(c) for c in contexts], os.path.join(args.out_dir, "close_rule_contexts.json"))

    print(
        f"[close-rule-dpo] loading policy base={args.model_path} "
        f"init={args.init_adapter} contexts={len(contexts)} device={args.device}"
    )
    policy = _load_policy(args)
    pairs = generate_pairs(policy, contexts, args)
    dump_json([asdict(p) for p in pairs], os.path.join(args.out_dir, "dpo_pairs_unscored.json"))
    print(f"[close-rule-dpo] kept pairs={len(pairs)}")

    if bool(args.build_only):
        summary = {
            "task": args.task,
            "init_adapter": args.init_adapter,
            "contexts": len(contexts),
            "pairs": len(pairs),
            "build_only": True,
            "args": vars(args),
        }
        dump_json(summary, os.path.join(args.out_dir, "close_rule_dpo_log.json"))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary

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
            desc=f"[close-rule-dpo] ep {epoch + 1}/{args.num_epochs}",
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
        "contexts": len(contexts),
        "pairs": len(pairs),
        "epochs": int(args.num_epochs),
        "global_steps": int(global_step),
        "final_loss": float(losses[-1]) if losses else None,
        "mean_loss": float(sum(losses) / len(losses)) if losses else None,
        "mean_pref_acc": float(sum(accs) / len(accs)) if accs else None,
        "elapsed_sec": float(time.perf_counter() - t0),
        "args": vars(args),
    }
    dump_json(summary, os.path.join(args.out_dir, "close_rule_dpo_log.json"))
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
    p.add_argument("--min_assistant_turn", type=int, default=4)
    p.add_argument("--max_assistant_turn", type=int, default=6)
    p.add_argument("--max_history_lines", type=int, default=30)
    p.add_argument("--skip_any_recent_refusal", type=int, default=1)
    p.add_argument("--close_min_turn", type=int, default=4)
    p.add_argument("--close_style", default="legacy",
                   choices=["legacy", "adaptive", "interest3", "replace_bad"])
    p.add_argument("--refusal_stop_after", type=int, default=2)
    p.add_argument("--max_contexts", type=int, default=0)
    p.add_argument("--max_pairs", type=int, default=0)
    p.add_argument("--max_pairs_per_context", type=int, default=1)
    p.add_argument("--candidates_per_context", type=int, default=2)
    p.add_argument("--gen_temperature", type=float, default=0.75)
    p.add_argument("--gen_top_p", type=float, default=0.95)
    p.add_argument("--gen_max_new_tokens", type=int, default=96)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--build_only", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
