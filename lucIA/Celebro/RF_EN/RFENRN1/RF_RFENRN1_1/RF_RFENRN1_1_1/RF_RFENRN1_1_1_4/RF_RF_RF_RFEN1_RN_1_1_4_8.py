from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import random


@dataclass
class PPOAgent:
    """Minimal PPO-like skeleton without external dependencies.

    This is not a full PPO but provides compatible interfaces:
    - select_action(state) -> action, logprob
    - remember(trajectory)
    - update() to simulate policy/value improvement
    """

    action_space: List[str] = field(default_factory=lambda: ["move", "say", "create_prim", "noop"])
    policy_table: Dict[str, float] = field(default_factory=dict)
    value_table: Dict[str, float] = field(default_factory=dict)
    gamma: float = 0.99
    clip_eps: float = 0.2
    lr: float = 0.05
    _buffer: List[Tuple[Dict[str, float], str, float]] = field(default_factory=list)

    def _state_key(self, state: Dict[str, float]) -> str:
        x = round(state.get("x", 0.0), 1)
        y = round(state.get("y", 0.0), 1)
        z = round(state.get("z", 0.0), 1)
        e = int(state.get("errors", 0.0))
        return f"{x},{y},{z},{e}"

    def select_action(self, state: Dict[str, float]) -> Tuple[str, float]:
        key = self._state_key(state)
        scores = []
        for a in self.action_space:
            scores.append(self.policy_table.get(f"{key}|{a}", 0.0))
        # softmax
        mx = max(scores) if scores else 0.0
        exps = [math.exp(s - mx) for s in scores]
        ssum = sum(exps) or 1.0
        probs = [e / ssum for e in exps]
        # sample
        r = random.random()
        cum = 0.0
        idx = 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                idx = i
                break
        action = self.action_space[idx]
        logprob = math.log(max(1e-8, probs[idx]))
        return action, logprob

    def remember(self, state: Dict[str, float], action: str, reward: float) -> None:
        self._buffer.append((dict(state), action, float(reward)))

    def update(self) -> None:
        # Compute returns (simple discounted sum)
        G = 0.0
        returns: List[float] = []
        for _, _, r in reversed(self._buffer):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        # Normalize returns
        mean = sum(returns) / max(1, len(returns))
        var = sum((g - mean) ** 2 for g in returns) / max(1, len(returns))
        std = math.sqrt(max(1e-6, var))
        norm_returns = [(g - mean) / std for g in returns]

        # Policy/value table update
        for (state, action, _), adv in zip(self._buffer, norm_returns):
            key = self._state_key(state)
            pkey = f"{key}|{action}"
            # clipped policy update
            old = self.policy_table.get(pkey, 0.0)
            self.policy_table[pkey] = old + max(-self.clip_eps, min(self.clip_eps, self.lr * adv))
            # value update towards return
            vkey = key
            vold = self.value_table.get(vkey, 0.0)
            self.value_table[vkey] = vold + self.lr * adv

        self._buffer.clear()
