"""NEUROSINTESIS - segunda pasada lingüística de LucIA.

La IA externa aporta una respuesta fuente. Esta capa recibe el estado
neuronal después de PSNRCV, construye un contexto cognitivo verificable y
genera una respuesta nueva en voz de LucIA.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[4]
CONFIG_PATH = ROOT_DIR / "LC" / "modelosIAlocal" / "IAlocal.json"
MEMORIA_PATH = ROOT_DIR / "LC" / "celebro" / "PSNRL" / "memoria_neural.jsonl"


class SintetizadorNeuronalLucIA:
    """Genera la respuesta final después de la asimilación neuronal."""

    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.config_path = config_path
        self.config = self._cargar_config()

    def _cargar_config(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _limpiar_respuesta(texto: str) -> Optional[str]:
        """Rechaza llamadas de herramientas y salidas vacías del modelo."""
        limpio = (texto or "").strip()
        if not limpio:
            return None
        patrones_herramienta = (
            r'"name"\s*:\s*"(?:web_search|browser|search|function)"',
            r"\b(?:web_search|tool_call|function_call)\b",
        )
        if any(re.search(pat, limpio, flags=re.IGNORECASE) for pat in patrones_herramienta):
            return None
        return limpio

    def construir_contexto(self, pregunta: str, respuesta_fuente: str,
                           estado_neuronal: Dict[str, Any], modelo_origen: str) -> Dict[str, Any]:
        campos = (
            "tono_cognitivo", "estado_emocional", "total_neuronas", "norma_delta_aplicada",
            "deriva_acumulada", "estado_control", "guia_sintesis", "vector_semantico",
        )
        estado = {k: estado_neuronal.get(k) for k in campos if k in estado_neuronal}
        return {
            "modelo_origen": modelo_origen,
            "estado_neuronal": estado,
            "pregunta": pregunta[:3000],
            "respuesta_fuente": respuesta_fuente[:6000],
            "regla": "La fuente informa; LucIA razona, integra y responde con palabras nuevas.",
        }

    @staticmethod
    def _sistema() -> str:
        return (
            "Eres la capa lingüística propia de LucIA. La respuesta fuente no es tu voz: "
            "es solo material de análisis. Usa el estado neuronal para responder de nuevo "
            "con tus propias palabras, en español, de forma clara y útil. No menciones el "
            "modelo de origen. No inventes hechos. No llames herramientas, no generes JSON "
            "de funciones y no escribas web_search. Devuelve únicamente la respuesta final "
            "para la persona que hizo la pregunta."
        )

    def _prompt(self, contexto: Dict[str, Any]) -> str:
        return (
            "Procesa este registro cognitivo. La pregunta es la entrada original, la "
            "respuesta fuente es información provisional y el estado neuronal contiene "
            "la síntesis distribuida por la red. Redacta ahora la respuesta final de LucIA.\n\n"
            + json.dumps(contexto, ensure_ascii=False, indent=2)
        )

    def _ollama(self, prompt: str, contexto: Dict[str, Any], backend: Dict[str, Any]) -> Optional[Tuple[str, str, float]]:
        base = str(backend.get("base_url", "http://127.0.0.1:11434")).rstrip("/")
        cfg = self.config.get("synthesis", {})
        model = str(cfg.get("model", self.config.get("default_model", "qwen2.5:3b")))
        inicio = time.perf_counter()
        payload = json.dumps({
            "model": model,
            "system": self._sistema(),
            "prompt": self._prompt(contexto),
            "stream": False,
            "options": {
                "temperature": float(cfg.get("temperature", 0.25)),
                "num_predict": int(cfg.get("max_tokens", 768)),
            },
        }).encode("utf-8")
        try:
            req = urllib.request.Request(f"{base}/api/generate", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=float(cfg.get("timeout_seconds", 120))) as resp:
                texto = self._limpiar_respuesta(str(json.loads(resp.read().decode()).get("response", "")))
            if texto:
                return texto, f"LucIA-sintesis:ollama:{model}", (time.perf_counter() - inicio) * 1000.0
        except Exception:
            return None
        return None

    def _lmstudio(self, prompt: str, contexto: Dict[str, Any], backend: Dict[str, Any]) -> Optional[Tuple[str, str, float]]:
        base = str(backend.get("base_url", "http://127.0.0.1:1234")).rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        cfg = self.config.get("synthesis", {})
        model = str(cfg.get("lmstudio_model", self.config.get("lmstudio_model", "qwen2.5-3b-instruct")))
        inicio = time.perf_counter()
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": self._sistema()},
                         {"role": "user", "content": self._prompt(contexto)}],
            "temperature": float(cfg.get("temperature", 0.25)),
            "max_tokens": int(cfg.get("max_tokens", 768)),
            "stream": False,
        }).encode("utf-8")
        try:
            req = urllib.request.Request(f"{base}/chat/completions", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=float(cfg.get("timeout_seconds", 120))) as resp:
                data = json.loads(resp.read().decode())
            texto = self._limpiar_respuesta(str(data["choices"][0]["message"].get("content", "")))
            if texto:
                return texto, f"LucIA-sintesis:lmstudio:{model}", (time.perf_counter() - inicio) * 1000.0
        except Exception:
            return None
        return None

    def _openrouter(self, contexto: Dict[str, Any], cliente_openrouter: Any) -> Optional[Tuple[str, str, float]]:
        if cliente_openrouter is None or not cliente_openrouter.esta_autenticado():
            return None
        try:
            texto, modelo, latencia = cliente_openrouter.generar_respuesta(
                prompt=self._prompt(contexto),
                contexto_neuronal=contexto.get("estado_neuronal", {}),
                stream_en_vivo=False,
            )
            texto_limpio = self._limpiar_respuesta(texto)
            return (texto_limpio, f"LucIA-sintesis:{modelo}", latencia) if texto_limpio else None
        except Exception:
            return None

    def sintetizar(self, pregunta: str, respuesta_fuente: str,
                   estado_neuronal: Dict[str, Any], modelo_origen: str,
                   cliente_openrouter: Any = None) -> Dict[str, Any]:
        contexto = self.construir_contexto(pregunta, respuesta_fuente, estado_neuronal, modelo_origen)
        cfg = self.config.get("synthesis", {})
        backends = cfg.get("order", ["ollama-local", "lmstudio-local", "openrouter-free"])
        definidos = {str(b.get("id")): b for b in self.config.get("backends", []) if isinstance(b, dict)}
        for backend_id in backends:
            backend = definidos.get(str(backend_id), {})
            if not backend.get("enabled", False):
                continue
            tipo = str(backend.get("type", "")).lower()
            resultado = self._ollama(self._prompt(contexto), contexto, backend) if tipo == "ollama" else None
            if tipo == "lmstudio":
                resultado = self._lmstudio(self._prompt(contexto), contexto, backend)
            if tipo == "openrouter":
                resultado = self._openrouter(contexto, cliente_openrouter)
            if resultado:
                texto, modelo, latencia = resultado
                self.persistir_memoria(pregunta, respuesta_fuente, estado_neuronal, modelo_origen, texto)
                return {"texto": texto, "modelo": modelo, "latencia_ms": latencia,
                        "fuente": "sintesis_neuronal", "contexto": contexto}
        self.persistir_memoria(pregunta, respuesta_fuente, estado_neuronal, modelo_origen, "")
        return {"texto": "", "modelo": "sintesis-no-disponible", "latencia_ms": 0.0,
                "fuente": "none", "contexto": contexto}

    @staticmethod
    def persistir_memoria(pregunta: str, fuente: str, estado: Dict[str, Any],
                          modelo_origen: str, respuesta_final: str) -> None:
        try:
            MEMORIA_PATH.parent.mkdir(parents=True, exist_ok=True)
            registro = {
                "timestamp": time.time(),
                "pregunta": pregunta[:3000],
                "respuesta_fuente_sha256": hashlib.sha256(fuente.encode("utf-8")).hexdigest(),
                "respuesta_final": respuesta_final[:6000],
                "modelo_origen": modelo_origen,
                "estado_neuronal": {k: estado.get(k) for k in (
                    "tono_cognitivo", "estado_emocional", "norma_delta_aplicada",
                    "deriva_acumulada", "total_neuronas", "guia_sintesis") if k in estado},
            }
            with MEMORIA_PATH.open("a", encoding="utf-8") as archivo:
                archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
        except Exception:
            pass


_SINTETIZADOR: Optional[SintetizadorNeuronalLucIA] = None


def get_sintetizador_lucia() -> SintetizadorNeuronalLucIA:
    global _SINTETIZADOR
    if _SINTETIZADOR is None:
        _SINTETIZADOR = SintetizadorNeuronalLucIA()
    return _SINTETIZADOR

