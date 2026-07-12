"""
Trajectory buffer used by the PPO trainer.

A trajectory step stores everything needed for an off-graph PPO update:
  * the chat messages (so we can re-tokenize and run fresh forward passes)
  * the sampled response token ids
  * the "old" log-prob (collected at rollout time)
  * the scalar per-token advantage (broadcast over response tokens)
  * a label telling us which side's policy produced this step

We keep the buffer role-separated — system vs user — so each PPO update
only sees steps produced by its own policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class TrajectoryStep:
    role: str  # 'system' | 'user'
    messages: List[Dict[str, str]]
    response_ids: torch.Tensor  # (T,)
    old_logp: torch.Tensor      # (T,) per-token log-prob at collection time
    advantage: float            # scalar advantage for the step
    ret: float                  # discounted return used as value target
    reward: float               # raw reward, diagnostic only


class TrajectoryBuffer:
    def __init__(self):
        self.steps: List[TrajectoryStep] = []

    def add(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    def by_role(self, role: str) -> List[TrajectoryStep]:
        return [s for s in self.steps if s.role == role]

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)

    # -------------------------------------------------- GAE-lite utilities

    @staticmethod
    def compute_monte_carlo_returns(
        rewards: List[float],
        gamma: float,
    ) -> List[float]:
        """Plain discounted Monte-Carlo returns (no value network)."""
        returns: List[float] = [0.0] * len(rewards)
        running = 0.0
        for i in reversed(range(len(rewards))):
            running = float(rewards[i]) + gamma * running
            returns[i] = running
        return returns

    @staticmethod
    def normalize(xs: List[float], eps: float = 1e-8) -> List[float]:
        if not xs:
            return xs
        t = torch.tensor(xs, dtype=torch.float32)
        mean = float(t.mean().item())
        std = float(t.std().item()) if t.numel() > 1 else 1.0
        return [(float(v) - mean) / (std + eps) for v in xs]
