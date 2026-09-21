from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ENRN_TRNS:
    """Lightweight transformer-like stub for sequence processing.

    Provides minimal interfaces:
    - configure: set hidden size and heads
    - generate: produce tokens based on a prompt
    - load_weights/save_weights: no-op stubs for compatibility
    """

    hidden_size: int = 128
    num_heads: int = 4
    use_flash_attention: bool = False
    quantization_bits: Optional[int] = None  # 8 or 4

    def configure(self, hidden_size: int, num_heads: int) -> None:
        if hidden_size <= 0 or num_heads <= 0:
            raise ValueError("hidden_size and num_heads must be > 0")
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def generate(self, prompt_ids: List[int], max_new_tokens: int = 16, temperature: float = 0.8) -> List[int]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        # deterministic, simple echo with jitter to avoid external deps
        out: List[int] = list(prompt_ids)
        base = (self.hidden_size // (self.num_heads or 1)) % 997
        for i in range(max_new_tokens):
            prev = out[-1] if out else base
            # very cheap pseudo randomness influenced by temperature
            next_id = (prev * 31 + int(temperature * 13) + i) % 1024
            out.append(next_id)
        return out

    def load_weights(self, path: Optional[str] = None) -> None:
        return None

    def save_weights(self, path: Optional[str] = None) -> None:
        return None

    # Optional acceleration backends (no hard deps)
    def configure_backend(self, use_flash_attention: bool = False) -> None:
        self.use_flash_attention = use_flash_attention

    def try_quantize(self, bits: int = 8) -> bool:
        if bits not in (4, 8):
            return False
        try:
            import bitsandbytes as bnb  # type: ignore
        except Exception:
            return False
        self.quantization_bits = bits
        return True

    # Simulate KV-cache return for downstream callers
    def generate_with_kv_cache(self, prompt_ids: List[int], max_new_tokens: int = 16) -> Tuple[List[int], dict]:
        tokens = self.generate(prompt_ids, max_new_tokens=max_new_tokens)
        kv_cache = {"size": len(tokens) // 2, "heads": self.num_heads}
        return tokens, kv_cache
