from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class OpenSimTerminal:
    """Abstracción mínima de terminal OpenSim.

    Esta clase no abre procesos; sólo simula un buffer de logs y comandos
    para pruebas sin dependencias externas.
    """

    _log_buffer: List[str] = field(default_factory=list)
    _cmd_buffer: List[str] = field(default_factory=list)

    def send_command(self, cmd: str) -> None:
        self._cmd_buffer.append(cmd)
        # Simular efecto mínimo
        now = time.time()
        self._log_buffer.append(f"{now:.3f} CMD {cmd}")
        if cmd.startswith("agent move"):
            self._log_buffer.append(f"{now:.3f} pos=(0.0,0.0,0.0)")

    def read_log(self, max_lines: int = 20) -> List[str]:
        if not self._log_buffer:
            return []
        lines = self._log_buffer[-max_lines:]
        return list(lines)

    def last_commands(self, n: int = 10) -> List[str]:
        return self._cmd_buffer[-n:]
