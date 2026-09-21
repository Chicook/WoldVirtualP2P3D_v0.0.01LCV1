"""Factoria de proveedores: lmstudio, ollama, llamacpp, offline."""
import os
from .base import LLMProvider
from .echo import EchoProvider
from .openai_compat import OpenAICompatProvider

def get_provider(kind: str = "auto") -> LLMProvider:
    kind = (kind or "auto").lower()
    lm_url = os.environ.get("LUCIA_LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
    ol_url = os.environ.get("LUCIA_OLLAMA_URL", "http://127.0.0.1:11434/v1")
    lc_url = os.environ.get("LUCIA_LLAMACPP_URL", "http://127.0.0.1:8080/v1")
    or_url = os.environ.get("LUCIA_OPENROUTER_URL", "https://openrouter.ai/api/v1")
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    candidates = []
    if kind in ("auto", "lmstudio", "lm"):
        candidates.append(OpenAICompatProvider(lm_url, "lmstudio"))
    if kind in ("auto", "ollama", "ol"):
        candidates.append(OpenAICompatProvider(ol_url, "ollama"))
    if kind in ("auto", "llamacpp"):
        candidates.append(OpenAICompatProvider(lc_url, "llamacpp"))
    if kind in ("auto", "openrouter", "cloud", "cl"):
        if or_key:
            candidates.append(OpenAICompatProvider(or_url, "openrouter", api_key=or_key))
    if kind in ("offline",):
        return EchoProvider()
    for p in candidates:
        if p.health():
            return p
    return EchoProvider()
