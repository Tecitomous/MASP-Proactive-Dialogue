"""
Lightweight PPO-clip trainer for a LoRA-wrapped causal LM policy.

We use a value-function-free variant: advantages come from Monte-Carlo
returns normalized per batch, and we clip the PPO ratio in the standard way:

    L_PPO(θ) = -E[min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]

where r_t(θ) = exp(log π_θ(a_t|s_t) - log π_θ_old(a_t|s_t)).

A KL penalty against a frozen reference policy is added on top:

    L_KL = β · KL(π_θ || π_ref)

The reference KL is estimated token-wise from the difference of log-probs
(standard RLHF estimate).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

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
    """Non-negative per-token KL approximation against the frozen reference."""
    log_ratio = (ref_logp - cur_logp).float().clamp(min=-20.0, max=20.0)
    approx_kl = torch.exp(log_ratio) - log_ratio - 1.0
    denom = mask.float().sum().clamp(min=1.0)
    return (approx_kl * mask.float()).sum() / denom


@dataclass
class PPOConfig:
    lr: float = 1e-5
    weight_decay: float = 0.0
    clip_ratio: float = 0.2
    kl_coeff: float = 0.05
    entropy_coeff: float = 0.0
    max_grad_norm: float = 1.0
    ppo_epochs: int = 2
    minibatch_size: int = 4
    target_kl: float = 0.05  # diagnostic target KL
    log_every: int = 10
    gradient_checkpointing: bool = True


class PPOTrainer:
    def __init__(
        self,
        policy: LoRAPolicy,
        ref_policy: LoRAPolicy,
        cfg: PPOConfig,
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(
            self.policy.trainable_parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    # ------------------------------------------------- single step update
    def _step_loss(
        self,
        step: TrajectoryStep,
    ) -> torch.Tensor:
        """
        Compute the per-step loss
            L = -min(r·A, clip(r, 1±ε)·A)  +  β · KL
        where the ratio is the exponential of the difference of summed log
        probs over the response segment. (Token-level averages are used.)
        """
        self.policy.train_mode()

        # Current log-probs (with grad)
        cur_logp, mask = self.policy.log_probs_of_response(step.messages, step.response_ids)
        old_logp = step.old_logp.to(cur_logp.device).detach()
        if old_logp.shape != cur_logp.shape:
            # Numerical oddity — truncate or pad to min length
            L = min(old_logp.shape[0], cur_logp.shape[0])
            cur_logp = cur_logp[:L]
            old_logp = old_logp[:L]
            mask = mask[:L]

        # Reference log-probs (no grad)
        with torch.no_grad():
            ref_logp = self.ref_policy.log_probs_ref(step.messages, step.response_ids)
            ref_logp = ref_logp.to(cur_logp.device)
            if ref_logp.shape != cur_logp.shape:
                L = min(ref_logp.shape[0], cur_logp.shape[0])
                ref_logp = ref_logp[:L]
                cur_logp = cur_logp[:L]
                old_logp = old_logp[:L]
                mask = mask[:L]

        # Advantage (scalar, broadcast over response tokens)
        adv = torch.tensor(float(step.advantage), dtype=cur_logp.dtype, device=cur_logp.device)

        # Token-level ratio
        log_ratio = cur_logp - old_logp
        ratio = torch.exp(log_ratio)

        clip_ratio = self.cfg.clip_ratio
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
        pg_per_token = -torch.min(unclipped, clipped)
        pg_loss = (pg_per_token * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

        # KL against ref. Use a non-negative approximation; the raw sampled
        # cur_logp-ref_logp term can become negative and destabilize training.
        kl_loss = _masked_approx_kl(cur_logp, ref_logp, mask)

        loss = pg_loss + self.cfg.kl_coeff * kl_loss
        return loss

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
            pg_per_token = -torch.min(unclipped, clipped)
            denom = mask.float().sum().clamp(min=1.0)
            pg_loss = (pg_per_token * mask.float()).sum() / denom

            kl_loss = _masked_approx_kl(cur_logp, ref_logp, mask)
            loss = pg_loss + self.cfg.kl_coeff * kl_loss
            if torch.isfinite(loss):
                losses.append(loss)
                kls.append(float(kl_loss.detach().item()))

        if not losses:
            return None, 0.0
        return torch.stack(losses).mean(), float(sum(kls) / max(len(kls), 1))

    # ---------------------------------------------------------- update
    def update(
        self,
        steps: List[TrajectoryStep],
        tag: str = "",
    ) -> dict:
        if not steps:
            return {"ppo_loss": 0.0, "num_steps": 0}

        losses: List[float] = []
        kl_values: List[float] = []
        n_updates = 0
        start_time = time.perf_counter()

        import random
        mb = max(int(self.cfg.minibatch_size), 1)
        ddp = _dist_is_on()
        world_size = _dist_world_size()
        rank = _dist_rank()
        if ddp:
            total_mb = ((len(steps) + (mb * world_size) - 1) // (mb * world_size)) * max(int(self.cfg.ppo_epochs), 1)
        else:
            total_mb = ((len(steps) + mb - 1) // mb) * max(int(self.cfg.ppo_epochs), 1)
        label = f":{tag}" if tag else ""
        if rank == 0:
            print(
                f"[ppo{label}] start steps={len(steps)} epochs={self.cfg.ppo_epochs} "
                f"minibatch={mb} ddp_world={world_size if ddp else 1} "
                f"total_mb={total_mb} gc={int(bool(self.cfg.gradient_checkpointing))}",
                flush=True,
            )

        old_gc = bool(getattr(self.policy.cfg, "gradient_checkpointing", False))
        if bool(self.cfg.gradient_checkpointing) and hasattr(self.policy, "set_gradient_checkpointing"):
            self.policy.set_gradient_checkpointing(True)

        try:
            for epoch in range(int(self.cfg.ppo_epochs)):
                order = list(range(len(steps)))
                if ddp:
                    random.Random(1729 + epoch).shuffle(order)
                    global_mb = mb * world_size
                    starts = range(0, len(order), global_mb)
                else:
                    random.shuffle(order)
                    starts = range(0, len(order), mb)

                for start in starts:
                    if ddp:
                        global_idxs = order[start: start + (mb * world_size)]
                        idxs = global_idxs[rank * mb: (rank + 1) * mb]
                    else:
                        idxs = order[start: start + mb]
                    mb_steps = [steps[i] for i in idxs]
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
                            f"[ppo{label}] update={n_updates}/{total_mb} "
                            f"loss={losses[-1]:.4f} kl={mb_kl:.4f} "
                            f"elapsed={time.perf_counter() - start_time:.1f}s",
                            flush=True,
                        )

        finally:
            if hasattr(self.policy, "set_gradient_checkpointing"):
                self.policy.set_gradient_checkpointing(old_gc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if rank == 0:
            print(
                f"[ppo{label}] done updates={n_updates} "
                f"loss={float(sum(losses) / max(len(losses), 1)):.4f} "
                f"kl={float(sum(kl_values) / max(len(kl_values), 1)):.4f} "
                f"elapsed={time.perf_counter() - start_time:.1f}s",
                flush=True,
            )
        return {
            "ppo_loss": float(sum(losses) / max(len(losses), 1)),
            "ppo_kl": float(sum(kl_values) / max(len(kl_values), 1)),
            "num_updates": int(n_updates),
        }
