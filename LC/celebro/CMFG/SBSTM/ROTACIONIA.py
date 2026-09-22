"""ROTACIONIA - Orquestador de inferencia para LucIA.

Orden configurable y resiliente:
    Ollama local -> LM Studio local -> OpenRouter gratuito.

La respuesta externa se devuelve sin ejecutar herramientas. La asimilación
neuronal se realiza en TRNLUC, después de recibir la respuesta.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[4]
CONFIG_PATH = ROOT_DIR / "LC" / "modelosIAlocal" / "IAlocal.json"


class RotadorIA:
    """Prueba backends en el orden declarado y devuelve el primero válido."""

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.config_path = config_path
        self.config = self._cargar_config()

    def _cargar_config(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def recargar(self) -> None:
        self.config = self._cargar_config()

    def estado(self) -> Dict[str, Any]:
        return {
            "orden": [b.get("id") for b in self._backends()],
            "backends": [{"id": b.get("id"), "enabled": bool(b.get("enabled")),
                          "type": b.get("type"), "base_url": b.get("base_url")}
                         for b in self._backends()],
        }

    def _backends(self) -> List[Dict[str, Any]]:
        backends = [b for b in self.config.get("backends", [])
                    if isinstance(b, dict) and b.get("enabled", False)]
        return sorted(backends, key=lambda b: int(b.get("priority", 999)))

    @staticmethod
    def _prompt_sistema(contexto: Dict[str, Any]) -> str:
        tono = contexto.get("tono_cognitivo", "analítico")
        valencia = contexto.get("estado_emocional", 0.0)
        return (
            "Eres un backend de LucIA. Responde en español, con claridad y sin "
            "llamadas de herramientas. No generes JSON de funciones, no escribas "
            "web_search y no afirmes que tienes acceso a Internet. La respuesta "
            "será procesada por la red neuronal de LucIA. "
            f"Tono cognitivo: {tono}. Valencia: {valencia:+.2f}."
        )

    def _ollama(self, prompt: str, contexto: Dict[str, Any], backend: Dict[str, Any]) -> Optional[Tuple[str, str, float]]:
        base = str(backend.get("base_url", "http://127.0.0.1:11434")).rstrip("/")
        tags = [str(self.config.get("default_model", "qwen2.5:3b"))]
        tags.extend(str(x) for x in self.config.get("models_available", []) if str(x) not in tags)
        try:
            req_tags = urllib.request.Request(f"{base}/api/tags")
            with urllib.request.urlopen(req_tags, timeout=3) as resp:
                disponibles = {str(x.get("name")) for x in json.loads(resp.read().decode()).get("models", [])}
            tags = [tag for tag in tags if tag in disponibles]
        except Exception:
            return None
        for tag in tags:
            inicio = time.perf_counter()
            payload = json.dumps({
                "model": tag,
                "system": self._prompt_sistema(contexto),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": float(self.config.get("temperature", 0.2)),
                    "num_predict": int(self.config.get("max_tokens", 1024)),
                },
            }).encode("utf-8")
            try:
                req = urllib.request.Request(f"{base}/api/generate", data=payload,
                                             headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=120) as resp:
                    text = str(json.loads(resp.read().decode()).get("response", "")).strip()
                if text:
                    return text, f"ollama:{tag}", (time.perf_counter() - inicio) * 1000.0
            except Exception:
                continue
        return None

    def _lmstudio(self, prompt: str, contexto: Dict[str, Any], backend: Dict[str, Any]) -> Optional[Tuple[str, str, float]]:
        base = str(backend.get("base_url", "http://127.0.0.1:1234")).rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        model = str(self.config.get("lmstudio_model", os.getenv("LMSTUDIO_MODEL", "qwen2.5-3b-instruct")))
        inicio = time.perf_counter()
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": self._prompt_sistema(contexto)},
                         {"role": "user", "content": prompt}],
            "temperature": float(self.config.get("temperature", 0.2)),
            "max_tokens": int(self.config.get("max_tokens", 1024)),
            "stream": False,
        }).encode("utf-8")
        try:
            req = urllib.request.Request(f"{base}/chat/completions", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
            text = str(data["choices"][0]["message"].get("content", "")).strip()
            return (text, f"lmstudio:{model}", (time.perf_counter() - inicio) * 1000.0) if text else None
        except Exception:
            return None

    def consultar(self, prompt: str, contexto: Optional[Dict[str, Any]] = None,
                  cliente_openrouter: Any = None) -> Dict[str, Any]:
        contexto = contexto or {}
        errores: List[str] = []
        for backend in self._backends():
            tipo = str(backend.get("type", "")).lower()
            try:
                if tipo == "ollama":
                    resultado = self._ollama(prompt, contexto, backend)
                elif tipo == "lmstudio":
                    resultado = self._lmstudio(prompt, contexto, backend)
                elif tipo == "openrouter" and cliente_openrouter is not None:
                    if not cliente_openrouter.esta_autenticado():
                        resultado = None
                        errores.append("openrouter:sin_api_key")
                    else:
                        texto, modelo, latencia = cliente_openrouter.generar_respuesta(
                            prompt, contexto_neuronal=contexto, stream_en_vivo=False)
                        resultado = None if not texto or texto.startswith("[IAFREE]") else (texto, modelo, latencia)
                else:
                    resultado = None
                if resultado:
                    texto, modelo, latencia = resultado
                    return {"texto": texto, "modelo": modelo, "latencia_ms": latencia,
                            "fuente": tipo, "errores": errores}
            except Exception as exc:
                errores.append(f"{backend.get('id', tipo)}:{type(exc).__name__}")
        return {"texto": "", "modelo": "sin-backend", "latencia_ms": 0.0,
                "fuente": "none", "errores": errores}


_ROTADOR: Optional[RotadorIA] = None


def get_rotador_ia() -> RotadorIA:
    global _ROTADOR
    if _ROTADOR is None:
        _ROTADOR = RotadorIA()
    return _ROTADOR

