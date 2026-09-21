from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import re


@dataclass
class RF_ENV:
    """Reinforcement Learning environment neuron for OpenSim logs.

    Responsibilities:
    - parse_log_line: extract key state from a single log line
    - reduce_state: accumulate into a compact state dictionary
    - compute_reward: simple reward shaping for navigation tasks
    """

    last_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    last_error_count: int = 0
    _pos_re = re.compile(r"pos=\(([-+\d\.]+),([-+\d\.]+),([-+\d\.]+)\)")

    def parse_log_line(self, line: str) -> Dict[str, float]:
        line = line.strip()
        state: Dict[str, float] = {}
        m = self._pos_re.search(line)
        if m:
            try:
                x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
                state.update({"x": x, "y": y, "z": z})
                self.last_position = (x, y, z)
            except Exception:
                pass
        if "ERROR" in line or "Exception" in line:
            self.last_error_count += 1
        state["errors"] = float(self.last_error_count)
        return state

    def reduce_state(self, acc: Dict[str, float], update: Dict[str, float]) -> Dict[str, float]:
        out = dict(acc)
        for k, v in update.items():
            out[k] = v
        return out

    def compute_reward(self, state: Dict[str, float], goal: Tuple[float, float, float]) -> float:
        x = state.get("x", self.last_position[0])
        y = state.get("y", self.last_position[1])
        z = state.get("z", self.last_position[2])
        dx = goal[0] - x
        dy = goal[1] - y
        dz = goal[2] - z
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        penalty = state.get("errors", 0.0) * 0.5
        return max(0.0, 10.0 - dist) - penalty

    # Gym-like API (minimal, no dependency)
    def reset(self) -> Dict[str, float]:
        self.last_position = (0.0, 0.0, 0.0)
        self.last_error_count = 0
        return {"x": 0.0, "y": 0.0, "z": 0.0, "errors": 0.0}

    def step(self, action: str, goal: Tuple[float, float, float]) -> Tuple[Dict[str, float], float, bool, Dict[str, float]]:
        # Placeholder state transition; in real system, action affects OpenSim
        state = {"x": self.last_position[0], "y": self.last_position[1], "z": self.last_position[2], "errors": float(self.last_error_count)}
        reward = self.compute_reward(state, goal)
        done = reward > 9.5
        info = {}
        return state, reward, done, info
