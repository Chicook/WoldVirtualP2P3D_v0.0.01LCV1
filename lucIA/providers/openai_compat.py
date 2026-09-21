"""Cliente OpenAI-compatible generico: sirve para LM Studio, Ollama y llama.cpp server."""
import json
import urllib.request
from typing import Dict, List, Optional
from .base import LLMProvider

class OpenAICompatProvider(LLMProvider):
    def __init__(self, base_url: str, name: str = "openai-compat", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.api_key = api_key or ""

    def list_models(self) -> List[str]:
        req = urllib.request.Request(f"{self.base_url}/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("id") for m in data.get("data", []) if m.get("id")]

    def chat(self, messages: List[Dict], model: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 450, timeout: int = 20) -> str:
        payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["HTTP-Referer"] = "https://woldvirtual.org"
            headers["X-Title"] = "LucIA"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            headers=headers,
            data=json.dumps(payload).encode("utf-8"),
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        return msg.get("content") or msg.get("reasoning_content") or choice.get("text") or ""
