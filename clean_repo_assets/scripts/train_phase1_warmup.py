#!/usr/bin/env python3
"""
Phase 1 — Behavior-Cloning warm-start for π_S_ref and π_U_ref.

Objectives:

    L_SFT^S = - Σ_{t,j} log π_θ_ref^S (a_{t,j} | h_t, ẑ_t, a_{t,<j})
    L_SFT^U = - Σ_{t,j} log π_ψ_ref^U (u_{t,j} | h_t, z_t, p_u, u_{t,<j})

We warm-start both LoRA adapters from real P4G utterances. For each
prefix the gold (B, D, I, ρ, c, v) silver label produced in Phase 0a is
injected as the canonical state text block so the prompt
template is identical to the one used during Phase 2 self-play.

Outputs:
    {out_dir}/pi_S/    — assistant LoRA adapter
    {out_dir}/pi_U/    — user LoRA adapter
    {out_dir}/phase1_log.json

Hardware sizing
---------------
Single LoRA a compatible causal LM in bf16 on one A800-40GB. With
gradient_checkpointing=False (default here) we comfortably fit
micro_batch=1 + grad_accum=8. To parallelize the two roles, run two
processes on different cards:

    python train_phase1_warmup.py --role assistant --device cuda:0 ...
    python train_phase1_warmup.py --role user      --device cuda:1 ...

Default `--role both` trains them sequentially on the same card.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from masp.data.bdi_dataset import BDILabelCache
from masp.data.dataset_adapter import get_adapter, DialogueSession
from masp.mind.bdi_schema import BDI, TASK_CONFIGS
from masp.models.policy import (
    LoRAPolicy,
    PolicyConfig,
    build_chat_prompt_for_policy,
)
from masp.utils.io import dump_json, ensure_dir
from masp.utils.seed import set_seed


# --------------------------------------------------------------------- BC examples

class BCExample:
    """One supervised example: (role, history_lines, target_text, pre-turn BDI)."""
    __slots__ = ("role", "history_lines", "target_text", "bdi", "session_id", "turn_idx")

    def __init__(
        self,
        role: str,
        history_lines: List[str],
        target_text: str,
        bdi: Optional[BDI],
        session_id: str,
        turn_idx: int,
    ):
        self.role = role
        self.history_lines = history_lines
        self.target_text = target_text
        self.bdi = bdi
        self.session_id = session_id
        self.turn_idx = turn_idx


def build_bc_examples(
    sessions: Sequence[DialogueSession],
    cache: BDILabelCache,
    role: str,
    max_history_lines: int = 30,
) -> List[BCExample]:
    """Build (history -> next-turn) BC examples for the requested role.

    For each turn we attach the silver BDI inferred from the history
    *before* that turn (next_speaker == speaker(turn)). For the assistant
    side this BDI plays the role of `ẑ_t` in the spec's L_SFT^S; at
    Phase-2 inference time the trained student M_φ produces the actual
    `ẑ_t` from the same prompt template.
    """
    if role not in {"assistant", "user"}:
        raise ValueError(f"role must be 'assistant' or 'user', got {role!r}")

    bdi_lookup: Dict[Tuple[str, int, str], BDI] = {}
    for e in cache.entries:
        bdi_lookup[(e.session_id, int(e.turn_idx), str(e.next_speaker))] = e.bdi

    examples: List[BCExample] = []
    for s in sessions:
        history: List[str] = []
        for idx, t in enumerate(s.turns):
            if t.speaker == role:
                bdi_pre = bdi_lookup.get((s.session_id, idx, role))
                # If a turn has no matching cache entry we skip it rather than
                # train on an unconditioned prompt — silently skipping keeps the
                # train-time prompt template identical to inference.
                if bdi_pre is not None:
                    examples.append(BCExample(
                        role=role,
                        history_lines=history[-max_history_lines:],
                        target_text=t.text,
                        bdi=bdi_pre,
                        session_id=s.session_id,
                        turn_idx=int(idx),
                    ))
            prefix = "Assistant" if t.speaker == "assistant" else "User"
            history.append(f"{prefix}: {t.text}")
    return examples


# --------------------------------------------------------------------- SFT loss

def _tokenize_target(
    tokenizer,
    target_text: str,
    max_target_len: int,
    add_eos: bool = True,
) -> torch.Tensor:
    """Tokenize a single target string into ids; append EOS if missing."""
    ids = tokenizer(
        target_text or "",
        truncation=True,
        max_length=max(max_target_len - 1, 1),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    if add_eos and tokenizer.eos_token_id is not None:
        if ids.numel() == 0 or int(ids[-1].item()) != int(tokenizer.eos_token_id):
            ids = torch.cat(
                [ids, torch.tensor([int(tokenizer.eos_token_id)], dtype=ids.dtype)],
                dim=0,
            )
    return ids.long()


def _build_messages_for_example(ex: BCExample, task_name: str) -> List[Dict[str, str]]:
    """Same prompt template as Phase 2 inference, anchored on the silver BDI."""
    bdi_text = ex.bdi.to_text(include_scalars=True) if ex.bdi else ""
    if ex.role == "assistant":
        return build_chat_prompt_for_policy(
            role="assistant",
            history_lines=ex.history_lines,
            belief_hint_text=bdi_text,
            task_name=task_name,
        )
    return build_chat_prompt_for_policy(
        role="user",
        history_lines=ex.history_lines,
        bdi_text=bdi_text,
        task_name=task_name,
    )


def sft_step_loss(
    policy: LoRAPolicy,
    ex: BCExample,
    max_target_len: int,
    task_name: str,
) -> Optional[torch.Tensor]:
    """Compute - mean(log p(target | prompt)) for a single example.

    Returns None if the target tokenizes to length 0 (skip).
    """
    messages = _build_messages_for_example(ex, task_name=task_name)
    response_ids = _tokenize_target(
        policy.tokenizer,
        ex.target_text,
        max_target_len=max_target_len,
        add_eos=True,
    )
    if response_ids.numel() == 0:
        return None
    logp, mask = policy.log_probs_of_response(messages, response_ids)
    if logp.numel() == 0:
        return None
    valid = mask.float()
    nll = -(logp * valid).sum() / valid.sum().clamp(min=1.0)
    return nll


# --------------------------------------------------------------------- training

def train_one_role(
    *,
    role: str,
    examples: List[BCExample],
    model_path: str,
    device: str,
    dtype: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: List[str],
    max_prompt_len: int,
    max_target_len: int,
    gradient_checkpointing: bool,
    attn_implementation: str,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    log_every: int,
    save_dir: str,
    seed: int,
    task_name: str,
) -> Dict:
    """Train a single LoRA policy on its role-specific BC examples."""
    if not examples:
        raise RuntimeError(f"no BC examples for role={role!r}; check the BDI cache")
    rng = random.Random(seed)

    cfg = PolicyConfig(
        model_name_or_path=model_path,
        device=device,
        dtype=dtype,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=list(lora_target_modules),
        max_prompt_len=max_prompt_len,
        max_new_tokens=max_target_len,
        gradient_checkpointing=gradient_checkpointing,
        attn_implementation=attn_implementation,
        generation_use_cache=False,  # train-time only
    )
    print(f"[phase1/{role}] loading {model_path} on {device} (lora_r={lora_r})")
    policy = LoRAPolicy(cfg)
    policy.train_mode()

    # AdamW only over the LoRA-trainable params.
    optimizer = torch.optim.AdamW(
        policy.trainable_parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
        betas=(0.9, 0.95),
    )

    history: Dict[str, List[float]] = {"loss": [], "lr": []}

    n_examples = len(examples)
    steps_per_epoch = math.ceil(n_examples / max(batch_size, 1))
    total_steps = steps_per_epoch * num_epochs
    print(
        f"[phase1/{role}] examples={n_examples} batch_size={batch_size} "
        f"epochs={num_epochs} total_steps={total_steps}"
    )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    t_start = time.perf_counter()
    for epoch in range(int(num_epochs)):
        order = list(range(n_examples))
        rng.shuffle(order)
        running = 0.0
        running_n = 0
        pbar = tqdm(
            range(0, n_examples, batch_size),
            desc=f"[phase1/{role}] ep {epoch + 1}/{num_epochs}",
        )
        for batch_start in pbar:
            batch_idxs = order[batch_start: batch_start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            valid_in_batch = 0
            batch_loss_sum = 0.0
            for i in batch_idxs:
                ex = examples[i]
                loss_i = sft_step_loss(
                    policy,
                    ex,
                    max_target_len=max_target_len,
                    task_name=task_name,
                )
                if loss_i is None or not torch.isfinite(loss_i):
                    continue
                # Scale per-example so the optimizer step matches the mean over
                # successful examples in the batch.
                (loss_i / float(max(len(batch_idxs), 1))).backward()
                batch_loss_sum += float(loss_i.detach().item())
                valid_in_batch += 1
            if valid_in_batch == 0:
                continue
            torch.nn.utils.clip_grad_norm_(
                policy.trainable_parameters(), float(max_grad_norm)
            )
            optimizer.step()
            global_step += 1
            mean_loss = batch_loss_sum / float(valid_in_batch)
            running += mean_loss
            running_n += 1
            history["loss"].append(mean_loss)
            history["lr"].append(float(optimizer.param_groups[0]["lr"]))
            if global_step % max(int(log_every), 1) == 0:
                pbar.set_postfix({
                    "loss": f"{mean_loss:.4f}",
                    "avg": f"{running / max(running_n, 1):.4f}",
                    "ex/s": f"{(global_step * batch_size) / max(time.perf_counter() - t_start, 1e-6):.1f}",
                })

    # ------- save adapter
    ensure_dir(save_dir)
    print(f"[phase1/{role}] saving LoRA adapter to {save_dir}")
    policy.save_adapter(save_dir)

    summary = {
        "role": role,
        "examples": int(n_examples),
        "epochs": int(num_epochs),
        "global_steps": int(global_step),
        "final_loss": float(history["loss"][-1]) if history["loss"] else float("nan"),
        "elapsed_sec": float(time.perf_counter() - t_start),
        "save_dir": save_dir,
    }
    return summary


# --------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    # ---- data
    p.add_argument("--train_path", type=str, required=True,
                   help="Raw P4G train split (json) — used for utterance text.")
    p.add_argument("--train_cache", type=str, required=True,
                   help="Phase 0a BDI label cache for train split.")
    p.add_argument("--task", type=str, default="p4g",
                   choices=list(TASK_CONFIGS.keys()))
    # ---- backbone
    p.add_argument("--model_path", type=str, required=True,
                   help="a compatible causal LM base checkpoint.")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Single GPU. For both roles run two processes "
                        "with different --device + --role values.")
    p.add_argument("--dtype", type=str, default="bf16")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    # ---- which role(s) to train
    p.add_argument("--role", type=str, default="both",
                   choices=["both", "assistant", "user"])
    # ---- output
    p.add_argument("--out_dir", type=str, required=True,
                   help="Adapters land at {out_dir}/pi_S and {out_dir}/pi_U.")
    # ---- LoRA
    p.add_argument("--lora_r", type=int, default=64,
                   help="LoRA rank.")
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    # ---- length
    p.add_argument("--max_prompt_len", type=int, default=2048,
                   help="Maximum prompt length in tokens.")
    p.add_argument("--max_target_len", type=int, default=128,
                   help="Maximum target length (slightly higher than generation-time max_new_tokens to absorb training samples).")
    p.add_argument("--max_history_lines", type=int, default=30)
    # ---- optim
    p.add_argument("--batch_size", type=int, default=8,
                   help="Effective batch via grad accumulation (we run 1 sample at a time).")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="SFT learning rate.")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Default off — single-policy SFT on a 40GB card has plenty of room.")
    # ---- misc
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    # ---- data
    adapter = get_adapter(args.task)
    print(f"[phase1] loading raw sessions: {args.train_path}")
    sessions = adapter.load_sessions(args.train_path)
    print(f"[phase1] loaded {len(sessions)} sessions")
    print(f"[phase1] loading BDI cache: {args.train_cache}")
    cache = BDILabelCache.load(args.train_cache)
    print(f"[phase1] cache entries={len(cache.entries)}")

    lora_targets = [s.strip() for s in args.lora_targets.split(",") if s.strip()]

    summaries: List[Dict] = []

    roles_to_train = ["assistant", "user"] if args.role == "both" else [args.role]
    for role in roles_to_train:
        examples = build_bc_examples(
            sessions, cache, role=role,
            max_history_lines=args.max_history_lines,
        )
        print(f"[phase1/{role}] BC examples: {len(examples)}")
        save_dir = os.path.join(
            args.out_dir, "pi_S" if role == "assistant" else "pi_U",
        )
        summary = train_one_role(
            role=role,
            examples=examples,
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=lora_targets,
            max_prompt_len=args.max_prompt_len,
            max_target_len=args.max_target_len,
            gradient_checkpointing=bool(args.gradient_checkpointing),
            attn_implementation=args.attn_implementation,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every,
            save_dir=save_dir,
            seed=args.seed,
            task_name=args.task,
        )
        summaries.append(summary)
        # Free the policy + cuda cache before training the next role.
        torch.cuda.empty_cache()

    log_path = os.path.join(args.out_dir, "phase1_log.json")
    dump_json({"task": args.task, "roles": summaries, "args": vars(args)}, log_path)
    print(f"[phase1] done. summary -> {log_path}")


if __name__ == "__main__":
    main()
