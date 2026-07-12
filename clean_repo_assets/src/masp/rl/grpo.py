"""
GRPO-style trainer for MASP self-play.

This is a value-free, group-relative policy-gradient update for dialogue
self-play. Instead of fitting a critic, each rollout batch is partitioned into
small groups and the reward/return inside each group is normalized to produce
advantages. The policy loss then uses the same clipped ratio and reference KL
regularization used by PPO.

For canonical GRPO the group contains several samples from the same prompt. In
MASP self-play the prompts evolve through the dialogue game, so we use rollout
mini-groups as the practical analogue. This keeps the branch simple, stable,
and directly comparable to the existing PPO path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import random
import time

import torch
import torch.distributed as dist

from ..models.policy import LoRAPolicy
from .buffer import TrajectoryStep


def _dist_is_on() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _dist_rank() -> int:
    return dist.get_rank() if _dist_is_on() else 0


def _dist_world_size() -> int:
    return dist.get_world_size() if _dist_is_on() else 1


def _zero_loss_for_params(params: List[torch.nn.Parameter]) -> torch.Tensor:
    for p in params:
        if p.requires_grad:
            return p.sum() * 0.0
    return torch.tensor(0.0)


def _average_gradients(params: List[torch.nn.Parameter]) -> None:
    if not _dist_is_on():
        return
    world_size = float(_dist_world_size())
    for p in params:
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def _masked_approx_kl(cur_logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Non-negative per-token KL approximation against the frozen reference.

    The previous direct sample term ``cur_logp - ref_logp`` can be negative on
    sampled minibatches and, when minimized as a loss, gives the optimizer an
    easy way to reduce the objective by moving away from the reference. The
    Schulman k3 approximation stays >= 0 and is zero when both policies agree.
    """
    log_ratio = (ref_logp - cur_logp).float().clamp(min=-20.0, max=20.0)
    approx_kl = torch.exp(log_ratio) - log_ratio - 1.0
    denom = mask.float().sum().clamp(min=1.0)
    return (approx_kl * mask.float()).sum() / denom


@dataclass
class GRPOConfig:
    lr: float = 1e-5
    weight_decay: float = 0.0
    clip_ratio: float = 0.2
    kl_coeff: float = 0.05
    max_grad_norm: float = 1.0
    grpo_epochs: int = 2
    minibatch_size: int = 4
    group_size: int = 8
    use_returns: bool = True
    target_kl: float = 0.05
    log_every: int = 10
    gradient_checkpointing: bool = True
    adv_eps: float = 1e-6


class GRPOTrainer:
    def __init__(
        self,
        policy: LoRAPolicy,
        ref_policy: LoRAPolicy,
        cfg: GRPOConfig,
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(
            self.policy.trainable_parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    def _score(self, step: TrajectoryStep) -> float:
        return float(step.ret if self.cfg.use_returns else step.reward)

    def _assign_group_advantages(self, steps: List[TrajectoryStep]) -> None:
        if not steps:
            return
        group_size = max(int(self.cfg.group_size), 2)
        fallback_scores = torch.tensor([self._score(s) for s in steps], dtype=torch.float32)
        fallback_mean = float(fallback_scores.mean().item())
        fallback_std = float(fallback_scores.std(unbiased=False).item())
        if fallback_std < self.cfg.adv_eps:
            fallback_std = 1.0

        for start in range(0, len(steps), group_size):
            group = steps[start:start + group_size]
            if len(group) < 2:
                for step in group:
                    step.advantage = (self._score(step) - fallback_mean) / fallback_std
                continue
            scores = torch.tensor([self._score(s) for s in group], dtype=torch.float32)
            mean = float(scores.mean().item())
            std = float(scores.std(unbiased=False).item())
            if std < self.cfg.adv_eps:
                std = 1.0
            for step, score in zip(group, scores.tolist()):
                step.advantage = (float(score) - mean) / std

    def _batch_loss(self, steps: List[TrajectoryStep]) -> Tuple[Optional[torch.Tensor], float]:
        self.policy.train_mode()
        messages = [s.messages for s in steps]
        response_ids = [s.response_ids for s in steps]

        cur_pairs = self.policy.log_probs_of_responses_batch_train(messages, response_ids)
        with torch.no_grad():
            ref_logps = self.ref_policy.log_probs_of_responses_batch(messages, response_ids)

        losses: List[torch.Tensor] = []
        kls: List[float] = []
        for step, (cur_logp, mask), ref_logp in zip(steps, cur_pairs, ref_logps):
            old_logp = step.old_logp.to(cur_logp.device).detach()
            ref_logp = ref_logp.to(cur_logp.device).detach()
            if old_logp.shape != cur_logp.shape or ref_logp.shape != cur_logp.shape:
                L = min(old_logp.shape[0], ref_logp.shape[0], cur_logp.shape[0])
                cur_logp = cur_logp[:L]
                old_logp = old_logp[:L]
                ref_logp = ref_logp[:L]
                mask = mask[:L]

            adv = torch.tensor(
                float(step.advantage), dtype=cur_logp.dtype, device=cur_logp.device
            )
            log_ratio = cur_logp - old_logp
            ratio = torch.exp(log_ratio)
            unclipped = ratio * adv
            clipped = torch.clamp(
                ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
            ) * adv
            denom = mask.float().sum().clamp(min=1.0)
            pg_loss = (-torch.min(unclipped, clipped) * mask.float()).sum() / denom

            kl_loss = _masked_approx_kl(cur_logp, ref_logp, mask)
            loss = pg_loss + self.cfg.kl_coeff * kl_loss
            if torch.isfinite(loss):
                losses.append(loss)
                kls.append(float(kl_loss.detach().item()))

        if not losses:
            return None, 0.0
        return torch.stack(losses).mean(), float(sum(kls) / max(len(kls), 1))

    def update(self, steps: List[TrajectoryStep], tag: str = "") -> dict:
        if not steps:
            return {"grpo_loss": 0.0, "grpo_kl": 0.0, "num_steps": 0, "num_updates": 0}

        work_steps = list(steps)
        self._assign_group_advantages(work_steps)

        losses: List[float] = []
        kl_values: List[float] = []
        n_updates = 0
        start_time = time.perf_counter()
        mb = max(int(self.cfg.minibatch_size), 1)
        epochs = max(int(self.cfg.grpo_epochs), 1)
        ddp = _dist_is_on()
        world_size = _dist_world_size()
        rank = _dist_rank()
        if ddp:
            total_mb = ((len(work_steps) + (mb * world_size) - 1) // (mb * world_size)) * epochs
        else:
            total_mb = ((len(work_steps) + mb - 1) // mb) * epochs
        label = f":{tag}" if tag else ""
        if rank == 0:
            print(
                f"[grpo{label}] start steps={len(work_steps)} epochs={epochs} "
                f"minibatch={mb} ddp_world={world_size if ddp else 1} "
                f"group_size={max(int(self.cfg.group_size), 2)} "
                f"use_returns={int(bool(self.cfg.use_returns))} total_mb={total_mb} "
                f"gc={int(bool(self.cfg.gradient_checkpointing))}",
                flush=True,
            )

        old_gc = bool(getattr(self.policy.cfg, "gradient_checkpointing", False))
        if bool(self.cfg.gradient_checkpointing) and hasattr(self.policy, "set_gradient_checkpointing"):
            self.policy.set_gradient_checkpointing(True)

        try:
            for _epoch in range(epochs):
                order = list(range(len(work_steps)))
                if ddp:
                    random.Random(1729 + _epoch).shuffle(order)
                    global_mb = mb * world_size
                    starts = range(0, len(order), global_mb)
                else:
                    random.shuffle(order)
                    starts = range(0, len(order), mb)

                for start in starts:
                    if ddp:
                        global_idxs = order[start:start + (mb * world_size)]
                        idxs = global_idxs[rank * mb: (rank + 1) * mb]
                    else:
                        idxs = order[start:start + mb]
                    mb_steps = [work_steps[i] for i in idxs]
                    params = self.policy.trainable_parameters()
                    self.optimizer.zero_grad(set_to_none=True)
                    mb_loss, mb_kl = self._batch_loss(mb_steps) if mb_steps else (None, 0.0)
                    if mb_loss is None or not torch.isfinite(mb_loss):
                        mb_loss = _zero_loss_for_params(params)
                    mb_loss.backward()
                    _average_gradients(params)
                    torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)
                    self.optimizer.step()
                    n_updates += 1
                    losses.append(float(mb_loss.detach().item()))
                    kl_values.append(float(mb_kl))

                    log_every = max(int(self.cfg.log_every), 0)
                    if rank == 0 and log_every and (n_updates % log_every == 0 or n_updates == total_mb):
                        print(
                            f"[grpo{label}] update={n_updates}/{total_mb} "
                            f"loss={losses[-1]:.4f} kl={mb_kl:.4f} "
                            f"elapsed={time.perf_counter() - start_time:.1f}s",
                            flush=True,
                        )
        finally:
            if hasattr(self.policy, "set_gradient_checkpointing"):
                self.policy.set_gradient_checkpointing(old_gc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_loss = float(sum(losses) / max(len(losses), 1))
        mean_kl = float(sum(kl_values) / max(len(kl_values), 1))
        if rank == 0:
            print(
                f"[grpo{label}] done updates={n_updates} loss={mean_loss:.4f} "
                f"kl={mean_kl:.4f} elapsed={time.perf_counter() - start_time:.1f}s",
                flush=True,
            )
        return {
            "grpo_loss": mean_loss,
            "grpo_kl": mean_kl,
            "num_steps": int(len(work_steps)),
            "num_updates": int(n_updates),
            "group_size": int(max(int(self.cfg.group_size), 2)),
            "use_returns": bool(self.cfg.use_returns),
        }
