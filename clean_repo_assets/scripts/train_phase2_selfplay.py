#!/usr/bin/env python3
"""
Phase 2 — Mentalized Adversarial Self-Play with alternating PPO.

Implements the MASP+OGR alternating self-play loop:

    for outer_iter in range(N_outer):
        # A. freeze user, update system
        episodes = rollout(...)
        ppo_update_system(episodes)

        # B. freeze system, update user
        episodes = rollout(...)
        ppo_update_user(episodes)

        # C. refresh student M_φ on collected prefixes
        update_student(student, teacher, collected_prefixes)

Hard constraints enforced:

    1. Teacher F_ω is loaded frozen and kept frozen the whole run.
    2. System policy never sees the oracle z_t — only ẑ_t produced by
       the student M_φ (via env.infer_bdi_hint_text).
    3. λ_out defaults to 0 → OGR is OFF unless explicitly enabled with
       --outcome_dir / --lambda_out > 0.
    4. Qwen3 chat template runs with enable_thinking=False (patched in
       masp.models.policy._apply_chat).
    5. Each phase produces standalone, recoverable checkpoints (best/ +
       latest/ + per-iter snapshots when --snapshot_every > 0).

Inputs (all from earlier phases):
    --train_cache    Phase 0a BDI label cache (provides MindPrior).
    --teacher_ckpt   Phase 0c teacher F_ω checkpoint.
    --mentalization_ckpt  Phase 0d student M_φ checkpoint.
    --pi_S_adapter   Phase 1 π_S_ref LoRA adapter directory.
    --pi_U_adapter   Phase 1 π_U_ref LoRA adapter directory.

Outputs under --out_dir:
    latest/{pi_S, pi_U, mentalization.pt}    — written each iter
    best/{pi_S, pi_U, mentalization.pt}      — written when eval improves
    iter_{n}/{...}                            — per-iter snapshots if asked
    selfplay_log.json                         — train + eval metrics
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

from masp.data.bdi_dataset import BDILabelCache
from masp.data.dataset_adapter import get_adapter
from masp.env.dialogue_env import POBGDialogueEnv
from masp.eval.metrics import DialogMetrics
from masp.eval.success_judge import build_success_judge
from masp.mind.bdi_extractor import BDIExtractor
from masp.mind.bdi_schema import TASK_CONFIGS
from masp.mind.mind_prior import MindPrior, MindPriorEntry
from masp.models.mentalization import (
    MentalizationConfig,
    MentalizationModule,
    TeacherMentalizationModule,
)
from masp.models.policy import LoRAPolicy, PolicyConfig, infer_lora_config_from_adapter
from masp.models.sentence_encoder import SentenceEncoder, SentenceEncoderConfig
from masp.rl.buffer import TrajectoryBuffer
from masp.rl.ppo import PPOConfig, PPOTrainer
from masp.rl.grpo import GRPOConfig, GRPOTrainer
from masp.rl.rewards import RationalityJudge, RewardConfig
from masp.rl.rollout import RolloutConfig, SelfPlayRollout
from masp.utils.io import dump_json, ensure_dir
from masp.utils.llm_client import LLMClient, LLMConfig
from masp.utils.seed import set_seed


# ----------------------------------------------------------- helpers

def _build_mind_prior_from_cache(cache: BDILabelCache, seed: int = 0) -> MindPrior:
    entries: List[MindPriorEntry] = []
    for sid, bdi in cache.initial_bdi.items():
        entries.append(MindPriorEntry(
            session_id=sid,
            bdi=bdi,
            profile_text=cache.profile_text.get(sid, ""),
        ))
    return MindPrior(entries=entries, seed=seed)


def _make_llm(args, role: str) -> LLMClient:
    """Build an LLMClient for the OBU or Judge from CLI flags."""
    if role == "judge":
        backend = args.judge_backend
        model = args.judge_model
        api_base = args.judge_api_base
        api_key_env = args.judge_api_key_env
        workers = args.judge_parallel_workers
    elif role == "obu":
        backend = args.obu_backend
        model = args.obu_model
        api_base = args.obu_api_base
        api_key_env = args.obu_api_key_env
        workers = args.obu_parallel_workers
    else:
        raise ValueError(f"unknown llm role: {role}")
    return LLMClient(LLMConfig(
        backend=backend,
        model=model,
        api_base=api_base,
        api_key_env=api_key_env,
        azure_endpoint=args.azure_endpoint,
        azure_api_version=args.azure_api_version,
        azure_thinking_budget=args.azure_thinking_budget,
        parallel_workers=workers,
        verbose=bool(args.llm_verbose),
        name=role,
    ))


def _build_mentalization_config(encoder, task_cfg, proj_hidden: int, dropout: float) -> MentalizationConfig:
    return MentalizationConfig(
        hidden_size=encoder.hidden_size,
        proj_hidden=int(proj_hidden),
        dropout=float(dropout),
        alpha_rho=task_cfg.alpha_rho,
        alpha_c=task_cfg.alpha_c,
        alpha_v=task_cfg.alpha_v,
    )


def _save_checkpoint_dir(
    out_dir: str,
    tag: str,
    pi_S: LoRAPolicy,
    pi_U: LoRAPolicy,
    student: MentalizationModule,
) -> str:
    """Write {out_dir}/{tag}/{pi_S, pi_U, mentalization.pt}. Returns dir path."""
    target = os.path.join(out_dir, tag)
    ensure_dir(target)
    pi_S.save_adapter(os.path.join(target, "pi_S"))
    pi_U.save_adapter(os.path.join(target, "pi_U"))
    student.save(os.path.join(target, "mentalization.pt"))
    return target


def _dist_is_on() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_rank() -> int:
    return dist.get_rank() if _dist_is_on() else 0


def _dist_world_size() -> int:
    return dist.get_world_size() if _dist_is_on() else 1


def _is_main_process() -> bool:
    return _dist_rank() == 0


def _dist_barrier() -> None:
    if not _dist_is_on():
        return
    backend = str(dist.get_backend()).lower()
    if backend == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def _init_distributed_if_needed(args) -> Tuple[bool, int, int, int]:
    """Initialize torch.distributed for phase2 DDP launched by torchrun.

    In DDP mode each process owns one visible GPU and loads the full phase2
    stack on that device. Rollouts are sharded across ranks, then trajectory
    objects are gathered so policy updates can run with synchronized gradients.
    """
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    requested = bool(int(getattr(args, "ddp", 0))) or env_world > 1
    if not requested:
        return False, 0, 1, 0
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requested, but torch.cuda.is_available() is false.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        timeout_minutes = float(os.environ.get("PHASE2_DDP_TIMEOUT_MINUTES", "60"))
        dist.init_process_group(
            backend=str(args.ddp_backend),
            timeout=timedelta(minutes=timeout_minutes),
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if bool(int(args.ddp_single_device_per_rank)):
        device = f"cuda:{local_rank}"
        args.pi_S_device = device
        args.pi_S_ref_device = device
        args.pi_U_device = device
        args.pi_U_ref_device = device
        args.mentalization_device = device
        if args.actor_replica_devices and rank == 0:
            print("[ddp] ignoring actor_replica_devices; torchrun ranks are the rollout actors.")
        args.actor_replica_devices = ""
    return True, rank, world_size, local_rank


def _local_episode_count(total: int) -> int:
    world_size = _dist_world_size()
    rank = _dist_rank()
    total = int(total)
    if world_size <= 1:
        return total
    base = total // world_size
    rem = total % world_size
    return base + (1 if rank < rem else 0)


def _gather_rollout_outputs(
    buf: TrajectoryBuffer,
    metrics: DialogMetrics,
    ment_samples: List[Dict],
) -> Tuple[TrajectoryBuffer, DialogMetrics, List[Dict]]:
    if not _dist_is_on():
        return buf, metrics, ment_samples

    payload = {
        "steps": buf.steps,
        "successes": list(metrics.successes),
        "turns": list(metrics.turns),
        "rewards": list(metrics.rewards),
        "traces": list(metrics.traces),
        "ment_samples": ment_samples,
    }
    gathered: List[Optional[Dict]] = [None for _ in range(_dist_world_size())]
    dist.all_gather_object(gathered, payload)

    merged_buf = TrajectoryBuffer()
    merged_metrics = DialogMetrics()
    merged_ment: List[Dict] = []
    for item in gathered:
        if not item:
            continue
        for step in item["steps"]:
            merged_buf.add(step)
        merged_metrics.successes.extend(item["successes"])
        merged_metrics.turns.extend(item["turns"])
        merged_metrics.rewards.extend(item["rewards"])
        merged_metrics.traces.extend(item.get("traces", []))
        merged_ment.extend(item["ment_samples"])
    return merged_buf, merged_metrics, merged_ment


def _eval_score_from_summary(summary: Dict, max_turns: int) -> float:
    return float(summary.get("SR", 0.0)) - 1e-3 * float(
        summary.get("AT", float(max_turns))
    )


def _best_score_from_log(log: Dict, max_turns: int) -> float:
    loaded_best = log.get("best_score", float("-inf"))
    if isinstance(loaded_best, (int, float)) and math.isfinite(float(loaded_best)):
        return float(loaded_best)

    best = float("-inf")
    for item in log.get("iters", []):
        if not isinstance(item, dict) or not isinstance(item.get("eval"), dict):
            continue
        best = max(best, _eval_score_from_summary(item["eval"], max_turns))
    return best


def _write_eval_traces(
    out_dir: str,
    outer_iter: int,
    eval_summary: Dict,
    metrics: DialogMetrics,
) -> str:
    trace_path = os.path.join(out_dir, "eval_traces", f"iter_{outer_iter:04d}.json")
    episodes: List[Dict] = []
    for episode_idx, trace in enumerate(metrics.traces):
        item = dict(trace)
        item["episode_index"] = int(episode_idx)
        episodes.append(item)
    dump_json({
        "iter": int(outer_iter),
        "summary": eval_summary,
        "n_traces": int(len(episodes)),
        "episodes": episodes,
    }, trace_path)
    return trace_path


# ----------------------------------------------------------- student refresh

def _bdi_loss(z_pred: torch.Tensor, z_target: torch.Tensor, scalar_loss_weight: float) -> torch.Tensor:
    if z_pred.shape != z_target.shape:
        raise ValueError(f"shape mismatch: {z_pred.shape} vs {z_target.shape}")
    if float(scalar_loss_weight) < 0:
        return F.mse_loss(z_pred, z_target)
    text_loss = F.mse_loss(z_pred[:, :-3], z_target[:, :-3])
    scalar_loss = F.mse_loss(z_pred[:, -3:], z_target[:, -3:])
    return text_loss + float(scalar_loss_weight) * scalar_loss

def _student_refresh_step(
    student: MentalizationModule,
    optimizer: torch.optim.Optimizer,
    histories: List[str],
    z_targets: torch.Tensor,
    real_histories: List[str],
    real_z_targets: torch.Tensor,
    real_weight: float,
    scalar_loss_weight: float,
    max_grad_norm: float,
) -> Dict:
    """One mini-batch student refresh against the frozen teacher.

    The composite loss follows the student co-training step (step C above)
    with an optional weighted
    blend of real-data targets (silver labels from the BDI cache) so the
    student doesn't drift purely on self-play distribution.
    """
    student.train()
    student.encoder.eval()  # encoder stays frozen
    losses: List[torch.Tensor] = []
    if histories:
        z_pred = student.forward(histories)  # (B, 3d+3) on student.device
        z_targets = z_targets.to(z_pred.device, dtype=z_pred.dtype)
        loss_sp = _bdi_loss(z_pred, z_targets, scalar_loss_weight)
        losses.append(loss_sp)
    real_loss_value = 0.0
    if real_histories and real_weight > 0.0:
        z_pred_real = student.forward(real_histories)
        real_z_targets = real_z_targets.to(z_pred_real.device, dtype=z_pred_real.dtype)
        loss_real = _bdi_loss(z_pred_real, real_z_targets, scalar_loss_weight)
        losses.append(real_weight * loss_real)
        real_loss_value = float(loss_real.detach().item())
    if not losses:
        return {"student_loss": 0.0, "self_play_loss": 0.0, "real_loss": 0.0, "did_step": False}
    total = sum(losses)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        list(student.trainable_parameters()), float(max_grad_norm)
    )
    optimizer.step()
    return {
        "student_loss": float(total.detach().item()),
        "self_play_loss": float(losses[0].detach().item()) if histories else 0.0,
        "real_loss": real_loss_value,
        "did_step": True,
    }


# ----------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()

    # ============ data
    p.add_argument("--train_cache", type=str, required=True,
                   help="Phase 0a BDI label cache for the train split.")
    p.add_argument("--task", type=str, default="p4g",
                   choices=list(TASK_CONFIGS.keys()))

    # ============ backbones / checkpoints
    p.add_argument("--model_path", type=str, required=True,
                   help="a compatible causal LM base — used by all four LoRA policies.")
    p.add_argument("--encoder_model", type=str, required=True,
                   help="Frozen sentence encoder backbone (typically a compatible causal LM).")
    p.add_argument("--teacher_ckpt", type=str, required=True,
                   help="Phase 0c teacher F_ω checkpoint (will be FROZEN).")
    p.add_argument("--mentalization_ckpt", type=str, required=True,
                   help="Phase 0d student M_φ checkpoint (co-trained at step C).")
    p.add_argument("--pi_S_adapter", type=str, required=True,
                   help="Phase 1 π_S_ref LoRA adapter directory.")
    p.add_argument("--pi_U_adapter", type=str, required=True,
                   help="Phase 1 π_U_ref LoRA adapter directory.")
    p.add_argument("--proj_hidden", type=int, default=512,
                   help="Mentalizer head hidden size (must match Phase 0).")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Mentalizer head dropout (must match Phase 0).")
    p.add_argument("--encoder_max_len", type=int, default=384,
                   help="Sentence encoder truncation length.")

    # ============ output
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--snapshot_every", type=int, default=0,
                   help="If > 0, also dump iter_{n}/ snapshots every N iters.")
    p.add_argument("--resume_log", type=str, default="",
                   help="Existing selfplay_log.json to append; starts from len(iters)+1.")

    # ============ device layout (8 × A800 40GB)
    p.add_argument("--pi_S_device", type=str, default="cuda:0")
    p.add_argument("--pi_S_ref_device", type=str, default="cuda:1")
    p.add_argument("--pi_U_device", type=str, default="cuda:2")
    p.add_argument("--pi_U_ref_device", type=str, default="cuda:3")
    p.add_argument("--mentalization_device", type=str, default="cuda:4",
                   help="Encoder + teacher F_ω + student M_φ all share this card.")

    # ============ precision
    p.add_argument("--dtype", type=str, default="bf16")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    p.add_argument("--generation_use_cache", type=int, default=1)
    p.add_argument("--policy_gradient_checkpointing", type=int, default=0,
                   help="0 lets generation use KV cache; set 1 if phase2 PPO OOMs.")
    p.add_argument("--lora_r", type=int, default=None,
                   help="Optional LoRA rank override. Defaults to adapter_config.json.")
    p.add_argument("--lora_alpha", type=int, default=None,
                   help="Optional LoRA alpha override. Defaults to adapter_config.json.")
    p.add_argument("--lora_dropout", type=float, default=None,
                   help="Optional LoRA dropout override. Defaults to adapter_config.json.")

    # ============ outer loop
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--episodes_per_cycle", type=int, default=24,
                   help="Episodes per rollout. Each outer iter does TWO rollouts "
                        "(one for π_S update, one for π_U update).")
    p.add_argument("--rollout_batch_size", type=int, default=12)
    p.add_argument("--max_turns", type=int, default=8)

    # ============ eval
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--eval_only", type=int, default=0,
                   help="1 = run evaluation once and exit without PPO updates.")

    # ============ reward weights (paper §3.4 starting points)
    p.add_argument("--alpha_shape", type=float, default=1.0,
                   help="w_prog in spec eq 23/25.")
    p.add_argument("--alpha_ment",  type=float, default=0.25,
                   help="w_inf in spec eq 23.")
    p.add_argument("--alpha_rat",   type=float, default=0.5,
                   help="w_rat in spec eq 25.")
    p.add_argument("--alpha_task",  type=float, default=0.0,
                   help="non-paper success bonus, default off.")
    p.add_argument("--alpha_term",  type=float, default=0.0,
                   help="non-paper terminal zero-sum, default off.")
    p.add_argument("--alpha_fid",   type=float, default=0.0,
                   help="non-paper voice-fidelity, default off.")
    p.add_argument("--step_penalty", type=float, default=0.0)
    p.add_argument("--alpha_safety", type=float, default=0.0,
                   help="non-paper anti-coercion penalty, default off.")
    p.add_argument("--alpha_early_success", type=float, default=0.0,
                   help="non-paper bonus for earlier success, gated by a "
                        "positive rationality signal; default off.")
    p.add_argument("--alpha_close_quality", type=float, default=0.0,
                   help="non-paper bounded reward for safe small-dollar close "
                        "timing and against high-dollar/goodbye failures.")
    p.add_argument("--success_threshold", type=float, default=0.6)

    # ============ PPO
    p.add_argument("--ppo_lr", type=float, default=1e-5)
    p.add_argument("--ppo_clip", type=float, default=0.2)
    p.add_argument("--ppo_epochs", type=int, default=2)
    p.add_argument("--ppo_minibatch", type=int, default=4)
    p.add_argument("--beta_kl_S", type=float, default=0.05,
                   help="π_S KL-to-ref coefficient (paper §3.4 w_kl^S).")
    p.add_argument("--beta_kl_U", type=float, default=0.05,
                   help="π_U KL-to-ref coefficient (paper §3.4 w_kl^U).")
    p.add_argument("--ppo_target_kl", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--ppo_log_every", type=int, default=10)
    p.add_argument("--ppo_gradient_checkpointing", type=int, default=1,
                   help="Enable gradient checkpointing only during PPO backward; keeps rollout generation cache fast.")
    p.add_argument("--rl_alg", type=str, default="ppo", choices=["ppo", "grpo"],
                   help="Policy-gradient branch for Phase 2: original PPO or GRPO-style group-relative update.")
    p.add_argument("--grpo_group_size", type=int, default=8,
                   help="Number of rollout steps per relative-reward group when --rl_alg grpo.")
    p.add_argument("--grpo_use_returns", type=int, default=1,
                   help="1 = normalize discounted returns in GRPO; 0 = normalize immediate rewards.")

    # ============ rollout sampling
    p.add_argument("--temperature_system", type=float, default=0.9)
    p.add_argument("--temperature_user", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens_system", type=int, default=96)
    p.add_argument("--max_new_tokens_user", type=int, default=96)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--rollout_verbose", type=int, default=1)
    p.add_argument("--rollout_log_every", type=int, default=1)

    # ============ student co-train (step C)
    p.add_argument("--ment_lr", type=float, default=3e-4)
    p.add_argument("--ment_updates_per_cycle", type=int, default=12)
    p.add_argument("--ment_batch_size", type=int, default=128)
    p.add_argument("--ment_real_weight", type=float, default=1.0,
                   help="Blend weight for silver-label examples drawn from "
                        "the BDI cache; 0.0 = self-play only.")
    p.add_argument("--ment_scalar_loss_weight", type=float, default=-1.0,
                   help="If >= 0, student refresh uses text_mse + weight * "
                        "scalar_mse instead of plain mean MSE over 3d+3 dims.")
    p.add_argument("--ment_max_history_lines", type=int, default=30)
    p.add_argument("--ment_grad_norm", type=float, default=1.0)

    # ============ judges
    p.add_argument("--judge_backend", type=str, default="openai",
                   choices=["openai", "azure", "local", "heuristic"])
    p.add_argument("--judge_model", type=str, default="gpt-3.5-turbo")
    p.add_argument("--judge_api_base", type=str, default="")
    p.add_argument("--judge_api_key_env", type=str, default="")
    p.add_argument("--judge_num_samples", type=int, default=5)
    p.add_argument("--judge_temperature", type=float, default=1.0)
    p.add_argument("--judge_max_tokens", type=int, default=16)
    p.add_argument("--judge_parallel_workers", type=int, default=16)

    p.add_argument("--obu_backend", type=str, default="openai",
                   choices=["openai", "azure", "local", "heuristic"])
    p.add_argument("--obu_model", type=str, default="gpt-3.5-turbo")
    p.add_argument("--obu_api_base", type=str, default="")
    p.add_argument("--obu_api_key_env", type=str, default="")
    p.add_argument("--obu_parallel_workers", type=int, default=8)

    p.add_argument("--azure_endpoint", type=str, default="")
    p.add_argument("--azure_api_version", type=str, default="2024-03-01-preview")
    p.add_argument("--azure_thinking_budget", type=int, default=0)

    # ============ env-call parallelism
    p.add_argument("--parallel_env_calls", type=int, default=1)
    p.add_argument("--env_call_workers", type=int, default=6)
    p.add_argument("--local_obu_feedback", type=int, default=0,
                   help="1 = skip rollout OBU calls and keep the current BDI text "
                        "for user conditioning; z* still comes from the frozen teacher.")
    p.add_argument("--local_rationality_feedback", type=int, default=0,
                   help="1 = skip rationality-judge calls and use q_t=+1.")
    p.add_argument("--actor_replica_devices", type=str, default="",
                   help="Comma-separated GPU devices for extra rollout actor replicas. Each replica loads pi_S+pi_U on one H20.")
    p.add_argument("--actor_replica_include_main", type=int, default=1,
                   help="1 = central train policies also collect rollout shards; 0 = replicas only if provided.")
    p.add_argument("--actor_sync_dir", type=str, default="",
                   help="Temporary directory for syncing updated LoRA adapters to rollout replicas.")

    p.add_argument("--llm_verbose", type=int, default=1)

    # ============ distributed phase2
    p.add_argument("--ddp", type=int, default=0,
                   help="Enable torchrun/DDP mode. Also auto-enables when WORLD_SIZE > 1.")
    p.add_argument("--ddp_backend", type=str, default="nccl")
    p.add_argument("--ddp_single_device_per_rank", type=int, default=1,
                   help="1 = load all phase2 modules on cuda:LOCAL_RANK in each rank.")

    # ============ OGR (wired but default OFF)
    p.add_argument("--outcome_dir", type=str, default="",
                   help="Directory of OutcomeEnsemble head_*.pt; if empty OGR is OFF.")
    p.add_argument("--lambda_out", type=float, default=0.0,
                   help="OGR weight; 0.0 = OFF (default for P4G v1).")
    p.add_argument("--w_p", type=float, default=0.5)
    p.add_argument("--w_a", type=float, default=1.0)
    p.add_argument("--w_sigma", type=float, default=0.2)

    # ============ misc
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    ddp_enabled, rank, world_size, local_rank = _init_distributed_if_needed(args)
    set_seed(int(args.seed) + (rank * 1009 if ddp_enabled else 0))
    if _is_main_process():
        ensure_dir(args.out_dir)
    if ddp_enabled:
        _dist_barrier()

    if ddp_enabled and _is_main_process():
        print(
            f"[ddp] enabled world_size={world_size} local_rank={local_rank} "
            f"single_device_per_rank={int(bool(args.ddp_single_device_per_rank))}",
            flush=True,
        )
    print(f"[phase2][rank{rank}] task={args.task}")
    task_cfg = TASK_CONFIGS[args.task]

    # ---------------- Encoder + teacher (FROZEN) + student
    print(f"[phase2] loading sentence encoder on {args.mentalization_device}")
    encoder = SentenceEncoder(SentenceEncoderConfig(
        model_name_or_path=args.encoder_model,
        device=args.mentalization_device,
        dtype=args.dtype,
        max_len=int(args.encoder_max_len),
    ))
    mcfg = _build_mentalization_config(encoder, task_cfg, args.proj_hidden, args.dropout)

    print(f"[phase2] loading FROZEN teacher F_ω from {args.teacher_ckpt}")
    teacher = TeacherMentalizationModule(encoder, mcfg).to(encoder.device)
    teacher.load(args.teacher_ckpt, map_location=str(encoder.device))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False  # teacher is frozen in RL

    print(f"[phase2] loading student M_φ from {args.mentalization_ckpt}")
    student = MentalizationModule(encoder, mcfg).to(encoder.device)
    student.load(args.mentalization_ckpt, map_location=str(encoder.device))

    student_optimizer = torch.optim.AdamW(
        list(student.trainable_parameters()),
        lr=float(args.ment_lr),
        weight_decay=0.0,
    )

    # ---------------- LLM judges
    judge_llm = _make_llm(args, role="judge")
    obu_llm = _make_llm(args, role="obu")
    bdi_extractor = BDIExtractor(llm=obu_llm, task_cfg=task_cfg)
    p4g_judge = build_success_judge(
        task_name=args.task,
        llm=judge_llm,
        success_threshold=args.success_threshold,
        num_samples=args.judge_num_samples,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
    )
    rationality_judge = RationalityJudge(
        llm=judge_llm,
        task_description=task_cfg.task_description,
    )

    # ---------------- Mind prior
    print(f"[phase2] loading BDI cache: {args.train_cache}")
    cache = BDILabelCache.load(args.train_cache)
    mind_prior = _build_mind_prior_from_cache(cache, seed=args.seed)
    if len(mind_prior) == 0:
        raise RuntimeError("MindPrior is empty — re-run extract_bdi_labels.py.")
    print(f"[phase2] mind prior size = {len(mind_prior)}")

    # ---------------- Reward + env
    reward_cfg = RewardConfig(
        alpha_shape=args.alpha_shape,
        alpha_ment=args.alpha_ment,
        alpha_rat=args.alpha_rat,
        alpha_task=args.alpha_task,
        alpha_term=args.alpha_term,
        alpha_fid=args.alpha_fid,
        step_penalty=args.step_penalty,
        alpha_safety=args.alpha_safety,
        alpha_early_success=args.alpha_early_success,
        alpha_close_quality=args.alpha_close_quality,
        success_threshold=args.success_threshold,
    )

    # Optional OGR: default off.
    outcome_ensemble = None
    outcome_reward_cfg = None
    if args.outcome_dir and float(args.lambda_out) > 0.0:
        from masp.models.outcome import (
            OutcomeEnsemble, OutcomeHeadConfig, OutcomeRewardConfig,
        )
        head_cfg = OutcomeHeadConfig(hidden_size=encoder.hidden_size)
        outcome_ensemble = OutcomeEnsemble.from_dir(
            args.outcome_dir, encoder, head_cfg,
        )
        outcome_reward_cfg = OutcomeRewardConfig(
            lambda_out=float(args.lambda_out),
            w_p=float(args.w_p),
            w_a=float(args.w_a),
            w_sigma=float(args.w_sigma),
        )
        print(
            f"[phase2] OGR enabled: K={len(outcome_ensemble.heads)} heads "
            f"λ_out={args.lambda_out} w_p={args.w_p} w_a={args.w_a} w_σ={args.w_sigma}"
        )
    else:
        print("[phase2] OGR disabled (λ_out = 0).")

    env = POBGDialogueEnv(
        task_cfg=task_cfg,
        mind_prior=mind_prior,
        sentence_encoder=encoder,
        mentalization=student,
        bdi_extractor=bdi_extractor,
        p4g_judge=p4g_judge,
        rationality_judge=rationality_judge,
        reward_cfg=reward_cfg,
        max_turns=args.max_turns,
        parallel_env_calls=bool(args.parallel_env_calls),
        env_call_workers=args.env_call_workers,
        teacher_mentalization=teacher,           # frozen teacher drives z*
        outcome_ensemble=outcome_ensemble,
        outcome_reward_cfg=outcome_reward_cfg,
        local_obu_feedback=bool(args.local_obu_feedback),
        local_rationality_feedback=bool(args.local_rationality_feedback),
    )

    # ---------------- Policies (4 LoRA a compatible causal LM copies)
    pi_S_lora_cfg = infer_lora_config_from_adapter(
        args.pi_S_adapter,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    pi_U_lora_cfg = infer_lora_config_from_adapter(
        args.pi_U_adapter,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    print(
        "[phase2] pi_S LoRA: "
        f"r={pi_S_lora_cfg.get('lora_r')} "
        f"alpha={pi_S_lora_cfg.get('lora_alpha')} "
        f"dropout={pi_S_lora_cfg.get('lora_dropout')}"
    )
    print(
        "[phase2] pi_U LoRA: "
        f"r={pi_U_lora_cfg.get('lora_r')} "
        f"alpha={pi_U_lora_cfg.get('lora_alpha')} "
        f"dropout={pi_U_lora_cfg.get('lora_dropout')}"
    )

    def _build_policy(
        device: str,
        label: str,
        max_new_tokens: int,
        lora_kwargs: Dict[str, object],
    ) -> LoRAPolicy:
        cfg = PolicyConfig(
            model_name_or_path=args.model_path,
            device=device,
            dtype=args.dtype,
            max_new_tokens=int(max_new_tokens),
            attn_implementation=args.attn_implementation,
            generation_use_cache=bool(int(args.generation_use_cache)),
            gradient_checkpointing=bool(int(args.policy_gradient_checkpointing)),
            **lora_kwargs,
        )
        print(f"[phase2] building {label} on {device}")
        return LoRAPolicy(cfg)

    pi_S = _build_policy(args.pi_S_device, "π_S", args.max_new_tokens_system, pi_S_lora_cfg)
    pi_S.load_adapter(args.pi_S_adapter)
    pi_S_ref = _build_policy(args.pi_S_ref_device, "π_S_ref", args.max_new_tokens_system, pi_S_lora_cfg)
    pi_S_ref.load_adapter(args.pi_S_adapter)
    for param in pi_S_ref.model.parameters():
        param.requires_grad = False
    pi_S_ref.eval_mode()

    pi_U = _build_policy(args.pi_U_device, "π_U", args.max_new_tokens_user, pi_U_lora_cfg)
    pi_U.load_adapter(args.pi_U_adapter)
    pi_U_ref = _build_policy(args.pi_U_ref_device, "π_U_ref", args.max_new_tokens_user, pi_U_lora_cfg)
    pi_U_ref.load_adapter(args.pi_U_adapter)
    for param in pi_U_ref.model.parameters():
        param.requires_grad = False
    pi_U_ref.eval_mode()

    # ---------------- RL trainers (PPO or GRPO)
    rl_alg = str(args.rl_alg).lower()
    if rl_alg == "ppo":
        trainer_S_cfg = PPOConfig(
            lr=args.ppo_lr,
            clip_ratio=args.ppo_clip,
            kl_coeff=args.beta_kl_S,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.ppo_minibatch,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.ppo_target_kl,
            log_every=args.ppo_log_every,
            gradient_checkpointing=bool(int(args.ppo_gradient_checkpointing)),
        )
        trainer_U_cfg = PPOConfig(
            lr=args.ppo_lr,
            clip_ratio=args.ppo_clip,
            kl_coeff=args.beta_kl_U,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.ppo_minibatch,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.ppo_target_kl,
            log_every=args.ppo_log_every,
            gradient_checkpointing=bool(int(args.ppo_gradient_checkpointing)),
        )
        trainer_S = PPOTrainer(pi_S, pi_S_ref, trainer_S_cfg)
        trainer_U = PPOTrainer(pi_U, pi_U_ref, trainer_U_cfg)
        rl_loss_key = "ppo_loss"
    elif rl_alg == "grpo":
        trainer_S_cfg = GRPOConfig(
            lr=args.ppo_lr,
            clip_ratio=args.ppo_clip,
            kl_coeff=args.beta_kl_S,
            grpo_epochs=args.ppo_epochs,
            minibatch_size=args.ppo_minibatch,
            group_size=args.grpo_group_size,
            use_returns=bool(int(args.grpo_use_returns)),
            max_grad_norm=args.max_grad_norm,
            target_kl=args.ppo_target_kl,
            log_every=args.ppo_log_every,
            gradient_checkpointing=bool(int(args.ppo_gradient_checkpointing)),
        )
        trainer_U_cfg = GRPOConfig(
            lr=args.ppo_lr,
            clip_ratio=args.ppo_clip,
            kl_coeff=args.beta_kl_U,
            grpo_epochs=args.ppo_epochs,
            minibatch_size=args.ppo_minibatch,
            group_size=args.grpo_group_size,
            use_returns=bool(int(args.grpo_use_returns)),
            max_grad_norm=args.max_grad_norm,
            target_kl=args.ppo_target_kl,
            log_every=args.ppo_log_every,
            gradient_checkpointing=bool(int(args.ppo_gradient_checkpointing)),
        )
        trainer_S = GRPOTrainer(pi_S, pi_S_ref, trainer_S_cfg)
        trainer_U = GRPOTrainer(pi_U, pi_U_ref, trainer_U_cfg)
        rl_loss_key = "grpo_loss"
    else:
        raise ValueError(f"unknown rl_alg: {args.rl_alg}")
    print(
        f"[phase2] rl_alg={rl_alg} lr={args.ppo_lr} epochs={args.ppo_epochs} "
        f"minibatch={args.ppo_minibatch} clip={args.ppo_clip} "
        f"grpo_group_size={args.grpo_group_size}",
        flush=True,
    )

    # ---------------- Rollout actors
    rollout_cfg = RolloutConfig(
        num_episodes=args.episodes_per_cycle,
        rollout_batch_size=args.rollout_batch_size,
        max_turns=args.max_turns,
        gamma=args.gamma,
        advantage_norm=True,
        verbose=bool(args.rollout_verbose),
        log_every=args.rollout_log_every,
        max_new_tokens_system=args.max_new_tokens_system,
        max_new_tokens_user=args.max_new_tokens_user,
        temperature_system=args.temperature_system,
        temperature_user=args.temperature_user,
        top_p=args.top_p,
        task_name=args.task,
    )
    rollout = SelfPlayRollout(env=env, pi_S=pi_S, pi_U=pi_U, cfg=rollout_cfg)

    actor_replica_devices = [
        x.strip() for x in str(args.actor_replica_devices or "").split(",") if x.strip()
    ]
    actor_rollouts: List[SelfPlayRollout] = []
    actor_policies: List[Tuple[LoRAPolicy, LoRAPolicy]] = []
    for idx, device in enumerate(actor_replica_devices, start=1):
        print(f"[phase2] building rollout actor replica {idx} on {device} (pi_S + pi_U)", flush=True)
        actor_pi_S = _build_policy(device, f"pi_S_actor{idx}", args.max_new_tokens_system, pi_S_lora_cfg)
        actor_pi_S.load_adapter(args.pi_S_adapter)
        actor_pi_S.eval_mode()
        actor_pi_U = _build_policy(device, f"pi_U_actor{idx}", args.max_new_tokens_user, pi_U_lora_cfg)
        actor_pi_U.load_adapter(args.pi_U_adapter)
        actor_pi_U.eval_mode()
        actor_policies.append((actor_pi_S, actor_pi_U))
        actor_rollouts.append(SelfPlayRollout(env=env, pi_S=actor_pi_S, pi_U=actor_pi_U, cfg=rollout_cfg))

    def _merge_rollouts(results: List[Tuple[TrajectoryBuffer, DialogMetrics, List[Dict]]]):
        merged_buf = TrajectoryBuffer()
        merged_metrics = DialogMetrics()
        merged_ment: List[Dict] = []
        for shard_buf, shard_metrics, shard_ment in results:
            for step in shard_buf.steps:
                merged_buf.add(step)
            merged_metrics.successes.extend(shard_metrics.successes)
            merged_metrics.turns.extend(shard_metrics.turns)
            merged_metrics.rewards.extend(shard_metrics.rewards)
            merged_metrics.traces.extend(shard_metrics.traces)
            merged_ment.extend(shard_ment)
        return merged_buf, merged_metrics, merged_ment

    def _run_rollout_pool(
        n_episodes: int,
        collect_S: bool,
        collect_U: bool,
        collect_ment: bool,
        phase_name: str,
    ) -> Tuple[TrajectoryBuffer, DialogMetrics, List[Dict]]:
        shards: List[SelfPlayRollout] = []
        if bool(int(args.actor_replica_include_main)) or not actor_rollouts:
            shards.append(rollout)
        shards.extend(actor_rollouts)
        if not shards:
            shards = [rollout]
        n_episodes = int(n_episodes)
        if n_episodes <= 0:
            return TrajectoryBuffer(), DialogMetrics(), []
        base = n_episodes // len(shards)
        rem = n_episodes % len(shards)
        counts = [base + (1 if i < rem else 0) for i in range(len(shards))]
        tasks = [(i, shard, counts[i]) for i, shard in enumerate(shards) if counts[i] > 0]
        if len(tasks) <= 1:
            _, shard, count = tasks[0]
            return shard.run_batch(count, collect_S, collect_U, collect_ment, phase_name=phase_name)
        print(
            f"[rollout_pool:{phase_name}] shards={len(tasks)} counts={[c for _, _, c in tasks]} "
            f"include_main={int(bool(int(args.actor_replica_include_main)))}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [
                pool.submit(
                    shard.run_batch,
                    count,
                    collect_S,
                    collect_U,
                    collect_ment,
                    f"{phase_name}:actor{i}",
                )
                for i, shard, count in tasks
            ]
            results = [f.result() for f in futures]
        return _merge_rollouts(results)

    def _sync_actor_replicas(role: str) -> None:
        if not actor_policies:
            return
        sync_root = args.actor_sync_dir or os.path.join(args.out_dir, "_actor_sync")
        ensure_dir(sync_root)
        if role == "system":
            path = os.path.join(sync_root, "pi_S")
            pi_S.save_adapter(path)
            for actor_pi_S, _actor_pi_U in actor_policies:
                actor_pi_S.load_adapter(path)
                actor_pi_S.eval_mode()
        elif role == "user":
            path = os.path.join(sync_root, "pi_U")
            pi_U.save_adapter(path)
            for _actor_pi_S, actor_pi_U in actor_policies:
                actor_pi_U.load_adapter(path)
                actor_pi_U.eval_mode()
        else:
            raise ValueError(f"unknown actor sync role: {role}")
        print(f"[phase2] synced {role} adapter to {len(actor_policies)} actor replica(s)", flush=True)

    # Pre-stack a real-data pool for student co-training (step C blend). We snapshot histories + teacher-projected z_targets from the
    # cache so the student never drifts purely on self-play distribution.
    print("[phase2] preparing real-data pool for student co-training ...")
    real_history_strings: List[str] = []
    for e in cache.entries:
        if not e.history_upto:
            continue
        h = "\n".join(e.history_upto[-int(args.ment_max_history_lines):])
        real_history_strings.append(h or "(no turns yet)")
    print(f"[phase2] real-data pool: {len(real_history_strings)} histories")
    real_rng = random.Random(args.seed)

    @torch.no_grad()
    def _teacher_z_for_real_batch(histories: List[str]) -> torch.Tensor:
        """Compute z* targets for the real-data co-training mini-batch
        through the FROZEN teacher (step C)."""
        # No profile_text by default for P4G v1 — profile injection stays out
        # of the canonical state text block at this stage.
        return teacher.forward_with_context(
            history_texts=histories,
            profile_texts=[""] * len(histories),
            goal_text=task_cfg.goal_bdi.to_text(include_scalars=True),
        ).detach()

    # ---------------- Outer loop
    log: Dict = {
        "task": args.task,
        "args": vars(args),
        "iters": [],
    }
    best_score = float("-inf")
    start_iter = 1
    if args.resume_log and os.path.exists(args.resume_log):
        with open(args.resume_log, "r", encoding="utf-8") as f:
            loaded_log = json.load(f)
        if isinstance(loaded_log, dict) and isinstance(loaded_log.get("iters"), list):
            log = loaded_log
            log.setdefault("task", args.task)
            log.setdefault("args", vars(args))
            log.setdefault("resume_events", []).append({
                "time": time.strftime("%F %T"),
                "from_iter": len(log.get("iters", [])) + 1,
                "target_iterations": int(args.iterations),
                "pi_S_adapter": args.pi_S_adapter,
                "pi_U_adapter": args.pi_U_adapter,
                "mentalization_ckpt": args.mentalization_ckpt,
                "rollout_batch_size": int(args.rollout_batch_size),
                "ppo_minibatch": int(args.ppo_minibatch),
                "attn_implementation": str(args.attn_implementation),
                "ddp_world_size": int(world_size),
            })
            start_iter = len(log.get("iters", [])) + 1
            best_score = _best_score_from_log(log, args.max_turns)
            if _is_main_process():
                print(
                    f"[phase2] resuming from {args.resume_log}: "
                    f"start_iter={start_iter} target={args.iterations} "
                    f"best={best_score:.4f}",
                    flush=True,
                )
        else:
            raise ValueError(f"invalid resume_log format: {args.resume_log}")
    snapshot_every = max(int(args.snapshot_every), 0)
    eval_every = max(int(args.eval_every), 1)

    if int(args.eval_only):
        if _is_main_process():
            print(f"[phase2/eval_only] running {args.eval_episodes} eval episodes ...")
        pi_S.eval_mode()
        pi_U.eval_mode()
        local_eval_eps = _local_episode_count(args.eval_episodes)
        buf_e_local, metrics_e_local, ment_e_local = _run_rollout_pool(
            n_episodes=local_eval_eps,
            collect_S=False,
            collect_U=False,
            collect_ment=False,
            phase_name="eval",
        )
        _buf_e, metrics_e, _ment_e = _gather_rollout_outputs(
            buf_e_local, metrics_e_local, ment_e_local
        )
        eval_summary = metrics_e.summary(args.max_turns)
        score = _eval_score_from_summary(eval_summary, args.max_turns)
        trace_path = os.path.join(args.out_dir, "eval_traces", "iter_0000.json")
        log["eval_only"] = {
            "eval": eval_summary,
            "eval_trace_path": trace_path,
            "score": float(score),
        }
        log["best_score"] = float(score)
        if _is_main_process():
            _write_eval_traces(args.out_dir, 0, eval_summary, metrics_e)
            _save_checkpoint_dir(args.out_dir, "latest", pi_S, pi_U, student)
            _save_checkpoint_dir(args.out_dir, "best", pi_S, pi_U, student)
            dump_json(log, os.path.join(args.out_dir, "selfplay_log.json"))
            print(
                f"[phase2/eval_only] SR={eval_summary.get('SR', 0.0):.3f} "
                f"AT={eval_summary.get('AT', 0.0):.2f} "
                f"score={score:.4f} traces={len(metrics_e.traces)} -> {trace_path}"
            )
        _dist_barrier()
        if _dist_is_on():
            dist.destroy_process_group()
        return

    for outer_iter in range(start_iter, int(args.iterations) + 1):
        iter_start = time.perf_counter()
        iter_log: Dict = {"iter": outer_iter}
        print(
            f"\n[phase2] ===== outer_iter {outer_iter}/{args.iterations} "
            f"===== ({time.strftime('%F %T')})"
        )

        # =========== A. freeze user, update system =====================
        pi_U.eval_mode()
        pi_S.train_mode()
        local_eps_A = _local_episode_count(args.episodes_per_cycle)
        bufA_local, metricsA_local, ment_samplesA_local = _run_rollout_pool(
            n_episodes=local_eps_A,
            collect_S=True,
            collect_U=False,
            collect_ment=True,
            phase_name="A",
        )
        bufA, metricsA, ment_samplesA = _gather_rollout_outputs(
            bufA_local, metricsA_local, ment_samplesA_local
        )
        sys_steps = bufA.by_role("system")
        update_metrics_S = trainer_S.update(sys_steps, tag="S")
        _sync_actor_replicas("system")
        sumA = metricsA.summary(args.max_turns)
        iter_log["phaseA"] = {
            "sys_steps": int(len(sys_steps)),
            "rl_alg": rl_alg,
            "policy_update_S": update_metrics_S,
            "metrics": sumA,
        }
        if _is_main_process():
            print(
                f"[phase2/A] sys_steps={len(sys_steps)} "
                f"{rl_alg}_loss={update_metrics_S.get(rl_loss_key, 0.0):.4f} "
                f"SR={sumA.get('SR', 0.0):.3f} AT={sumA.get('AT', 0.0):.2f}"
            )

        # =========== B. freeze system, update user =====================
        pi_S.eval_mode()
        pi_U.train_mode()
        local_eps_B = _local_episode_count(args.episodes_per_cycle)
        bufB_local, metricsB_local, ment_samplesB_local = _run_rollout_pool(
            n_episodes=local_eps_B,
            collect_S=False,
            collect_U=True,
            collect_ment=True,
            phase_name="B",
        )
        bufB, metricsB, ment_samplesB = _gather_rollout_outputs(
            bufB_local, metricsB_local, ment_samplesB_local
        )
        usr_steps = bufB.by_role("user")
        update_metrics_U = trainer_U.update(usr_steps, tag="U")
        _sync_actor_replicas("user")
        sumB = metricsB.summary(args.max_turns)
        iter_log["phaseB"] = {
            "usr_steps": int(len(usr_steps)),
            "rl_alg": rl_alg,
            "policy_update_U": update_metrics_U,
            "metrics": sumB,
        }
        if _is_main_process():
            print(
                f"[phase2/B] usr_steps={len(usr_steps)} "
                f"{rl_alg}_loss={update_metrics_U.get(rl_loss_key, 0.0):.4f} "
                f"SR={sumB.get('SR', 0.0):.3f} AT={sumB.get('AT', 0.0):.2f}"
            )

        # =========== C. refresh student M_φ on collected prefixes =====
        ment_pool = ment_samplesA + ment_samplesB
        student_logs: List[Dict] = []
        if ment_pool and args.ment_updates_per_cycle > 0:
            for upd in range(args.ment_updates_per_cycle):
                # Self-play sample
                idx = [
                    real_rng.randrange(len(ment_pool))
                    for _ in range(min(args.ment_batch_size, len(ment_pool)))
                ]
                hist_batch = [ment_pool[i]["history_text"] for i in idx]
                z_batch = torch.stack([ment_pool[i]["z_target"] for i in idx], dim=0)
                # Real-data sample (silver-label targets via teacher)
                if real_history_strings and args.ment_real_weight > 0.0:
                    real_idx = [
                        real_rng.randrange(len(real_history_strings))
                        for _ in range(min(args.ment_batch_size, len(real_history_strings)))
                    ]
                    real_hists = [real_history_strings[i] for i in real_idx]
                    real_z = _teacher_z_for_real_batch(real_hists).cpu()
                else:
                    real_hists, real_z = [], torch.zeros(0)
                step_log = _student_refresh_step(
                    student=student,
                    optimizer=student_optimizer,
                    histories=hist_batch,
                    z_targets=z_batch,
                    real_histories=real_hists,
                    real_z_targets=real_z,
                    real_weight=float(args.ment_real_weight),
                    scalar_loss_weight=float(args.ment_scalar_loss_weight),
                    max_grad_norm=float(args.ment_grad_norm),
                )
                student_logs.append(step_log)
        iter_log["phaseC_student"] = {
            "updates": int(len(student_logs)),
            "mean_loss": float(
                sum(s["student_loss"] for s in student_logs)
                / max(len(student_logs), 1)
            ) if student_logs else 0.0,
        }
        if _is_main_process():
            print(
                f"[phase2/C] student_updates={len(student_logs)} "
                f"mean_loss={iter_log['phaseC_student']['mean_loss']:.4f}"
            )

        # =========== save latest + maybe snapshot =====================
        if _is_main_process():
            _save_checkpoint_dir(args.out_dir, "latest", pi_S, pi_U, student)
            if snapshot_every > 0 and outer_iter % snapshot_every == 0:
                _save_checkpoint_dir(args.out_dir, f"iter_{outer_iter}", pi_S, pi_U, student)
        _dist_barrier()

        # =========== eval + best tracking =============================
        if outer_iter % eval_every == 0:
            if _is_main_process():
                print(f"[phase2/eval] running {args.eval_episodes} eval episodes ...")
            pi_S.eval_mode()
            pi_U.eval_mode()
            local_eval_eps = _local_episode_count(args.eval_episodes)
            buf_e_local, metrics_e_local, ment_e_local = _run_rollout_pool(
                n_episodes=local_eval_eps,
                collect_S=False,
                collect_U=False,
                collect_ment=False,
                phase_name="eval",
            )
            _buf_e, metrics_e, _ment_e = _gather_rollout_outputs(
                buf_e_local, metrics_e_local, ment_e_local
            )
            eval_summary = metrics_e.summary(args.max_turns)
            iter_log["eval"] = eval_summary
            trace_path = os.path.join(
                args.out_dir, "eval_traces", f"iter_{outer_iter:04d}.json"
            )
            iter_log["eval_trace_path"] = trace_path
            score = _eval_score_from_summary(eval_summary, args.max_turns)
            if _is_main_process():
                _write_eval_traces(args.out_dir, outer_iter, eval_summary, metrics_e)
                print(
                    f"[phase2/eval] SR={eval_summary.get('SR', 0.0):.3f} "
                    f"AT={eval_summary.get('AT', 0.0):.2f} "
                    f"score={score:.4f} (best={best_score:.4f}) "
                    f"traces={len(metrics_e.traces)} -> {trace_path}"
                )
            if score > best_score:
                best_score = score
                if _is_main_process():
                    _save_checkpoint_dir(args.out_dir, "best", pi_S, pi_U, student)
                    print(f"[phase2/eval] NEW BEST score={score:.4f} saved to best/")
                _dist_barrier()

        iter_log["wall_sec"] = float(time.perf_counter() - iter_start)
        log["iters"].append(iter_log)
        log["best_score"] = float(best_score) if math.isfinite(best_score) else None
        if _is_main_process():
            dump_json(log, os.path.join(args.out_dir, "selfplay_log.json"))

    # ---------------- final save
    if _is_main_process():
        _save_checkpoint_dir(args.out_dir, "final", pi_S, pi_U, student)
    log["best_score"] = float(best_score) if math.isfinite(best_score) else None
    if _is_main_process():
        dump_json(log, os.path.join(args.out_dir, "selfplay_log.json"))
    _dist_barrier()
    if _is_main_process():
        print(f"\n[phase2] done. best score = {best_score:.4f}")
        print(f"[phase2] checkpoints under: {args.out_dir}")
    if _dist_is_on():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
