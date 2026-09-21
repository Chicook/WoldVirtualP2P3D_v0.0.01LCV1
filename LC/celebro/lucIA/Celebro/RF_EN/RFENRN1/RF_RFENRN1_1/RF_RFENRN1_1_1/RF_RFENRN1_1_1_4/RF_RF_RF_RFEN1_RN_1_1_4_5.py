from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class RF_CMND:
    """Command neuron to build and schedule OpenSim commands.

    Methods:
    - build_command: create CLI strings for common actions
    - schedule_delay: compute timing offsets
    - apply_policy: simple action selection from a state dict
    """

    default_delay_ms: int = 50
    _history: List[str] = None  # type: ignore

    def build_command(self, action: str, params: Optional[Dict[str, str]] = None) -> str:
        params = params or {}
        if action == "move":
            x = params.get("x", "0")
            y = params.get("y", "0")
            z = params.get("z", "0")
            return f"agent move {x} {y} {z}"
        if action == "create_prim":
            shape = params.get("shape", "box")
            return f"create prim {shape}"
        if action == "say":
            msg = params.get("text", "hello")
            return f"say {msg}"
        return f"noop {action}"

    def schedule_delay(self, step_idx: int) -> int:
        return int(self.default_delay_ms + (step_idx % 5) * 10)

    def apply_policy(self, state: Dict[str, float]) -> str:
        # minimal heuristic policy: if errors increase, say diagnostic; else move towards origin
        if state.get("errors", 0.0) > 0:
            return self.build_command("say", {"text": "diagnosing"})
        return self.build_command("move", {"x": "0", "y": "0", "z": "0"})

    # Simple finite-state sequencing without external libs
    def start_session(self) -> None:
        self._history = []

    def record(self, cmd: str) -> None:
        if self._history is None:
            self._history = []
        self._history.append(cmd)

    def last_commands(self, n: int = 5) -> List[str]:
        if not self._history:
            return []
        return self._history[-n:]
