from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RFENRN_OPT:
    """Resource optimizer neuron to adapt pruning and P2P bandwidth.

    Methods:
    - read_resources: get CPU/RAM usage via os and placeholders
    - adjust_pruning: return pruning ratio [0,1]
    - adjust_p2p_bandwidth: return KB/s target based on load
    """

    target_cpu_util: float = 0.6
    target_ram_util: float = 0.7

    def read_resources(self) -> Dict[str, float]:
        # Placeholders since psutil is not guaranteed available
        cpu_cores = os.cpu_count() or 1
        cpu_util = min(0.95, 0.25 + (cpu_cores % 4) * 0.1)
        ram_util = 0.5  # static placeholder
        return {"cpu_util": cpu_util, "ram_util": ram_util}

    def adjust_pruning(self, cpu_util: float, ram_util: float) -> float:
        cpu_gap = max(0.0, cpu_util - self.target_cpu_util)
        ram_gap = max(0.0, ram_util - self.target_ram_util)
        pruning = min(1.0, 0.2 + cpu_gap * 0.5 + ram_gap * 0.3)
        return pruning

    def adjust_p2p_bandwidth(self, cpu_util: float, ram_util: float) -> int:
        base_kbps = 64_000  # 64 Mbps baseline
        penalty = int(base_kbps * (max(cpu_util - self.target_cpu_util, 0.0) + max(ram_util - self.target_ram_util, 0.0)) * 0.5)
        return max(2_000, base_kbps - penalty)

    # Optional pruning/quantization hooks (no hard deps)
    def try_prune_torch(self, module: object, amount: float = 0.2) -> bool:
        try:
            import torch
            import torch.nn.utils.prune as prune  # type: ignore
        except Exception:
            return False
        try:
            for name, p in getattr(module, 'named_modules', lambda: [])():
                if hasattr(p, 'weight'):
                    prune.l1_unstructured(p, name='weight', amount=amount)
        except Exception:
            return False
        return True

    def try_export_onnx(self, module: object, dummy_input: Optional[object], path: str) -> bool:
        try:
            import torch  # type: ignore
        except Exception:
            return False
        try:
            torch.onnx.export(module, dummy_input, path, opset_version=17, input_names=['input'], output_names=['output'])
            return True
        except Exception:
            return False
