"""Proveedor base OpenAI-compatible."""
from typing import Dict, List, Optional

class LLMProvider:
    name = "base"
    def list_models(self) -> List[str]:
        return []
    def chat(self, messages: List[Dict], model: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 450, timeout: int = 20) -> str:
        raise NotImplementedError
    def health(self) -> bool:
        try:
            self.list_models()
            return True
        except Exception:
            return False
