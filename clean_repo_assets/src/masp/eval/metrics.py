"""DialogXpert-style metrics: SR, AT, AT_success."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DialogMetrics:
    successes: List[int] = field(default_factory=list)
    turns: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    traces: List[Dict] = field(default_factory=list)

    def add(
        self,
        success: bool,
        turns: int,
        reward: float = 0.0,
        trace: Optional[Dict] = None,
    ) -> None:
        self.successes.append(1 if success else 0)
        self.turns.append(int(turns))
        self.rewards.append(float(reward))
        if trace is not None:
            self.traces.append(trace)

    def summary(self, max_turns: int) -> Dict[str, float]:
        n = max(len(self.successes), 1)
        sr = sum(self.successes) / n
        at = sum(self.turns) / n
        succ_turns = [t for s, t in zip(self.successes, self.turns) if s == 1]
        at_succ = (sum(succ_turns) / len(succ_turns)) if succ_turns else float(max_turns)
        mean_reward = sum(self.rewards) / n
        return {
            "SR": float(sr),
            "AT": float(at),
            "AT_success": float(at_succ),
            "mean_reward": float(mean_reward),
            "n_episodes": int(n),
        }
