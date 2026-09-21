"""Fallback offline sin alucinaciones."""
from typing import Dict, List, Optional
from .base import LLMProvider

class EchoProvider(LLMProvider):
    name = "echo-offline"
    def list_models(self) -> List[str]:
        return ["lucia-core-offline"]
    def chat(self, messages: List[Dict], model: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 450, timeout: int = 20) -> str:
        return "Pues mira, estoy en modo local sin modelo externo. Mi red en Celebro sigue activa y te respondo con lo asimilado."
