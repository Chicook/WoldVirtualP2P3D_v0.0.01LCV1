"""
RF_RF_RF_RFEN1_RN_1_1_10_1
Neurona de Refuerzo LLM-Orquestador para Sincronización Cognitiva en Tiempo Real
===============================================================================

Esta neurona se apoya en modelos LLM locales gestionados por Ollama para:
- Recibir el estado de sincronización en tiempo real (por ejemplo `RealTimeSync`)
- Consultar información adicional en Internet con duckduckgo-search
- Cruzar los datos locales + web + otros LLMs
- Emitir una decisión/diagnóstico reforzado listo para integrarse con LucIA Core

NOTA: Esta neurona es autocontenida. Puedes conectarla desde `main.py` o desde
otras neuronas RF_EN creando un wrapper que le pase:
- métricas de sincronización
- estado de red / latencias
- descripción de la escena 3D
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import requests
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None  # type: ignore

import sys
import os
# Añadir la raíz de lucIA al path para importar llm_connector
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# Subir 7 niveles hasta lucIA
lucia_root = os.path.abspath(os.path.join(current_file_dir, "../../../../../../.."))
if lucia_root not in sys.path:
    sys.path.insert(0, lucia_root)

try:
    from llm_connector import LLMConnector
except ImportError:
    LLMConnector = None


# ──────────────────────────────────────────────────────────────
# DATA CLASSES DE CONTEXTO
# ──────────────────────────────────────────────────────────────

@dataclass
class SyncContext:
    """
    Estado mínimo necesario para que la neurona pueda razonar:
    - métricas de sincronización tiempo real
    - info de red / P2P
    - descripción sintética de la escena 3D
    """

    sync_metrics: Dict[str, Any] = field(default_factory=dict)
    network_status: Dict[str, Any] = field(default_factory=dict)
    scene_summary: str = ""

    def to_prompt_block(self) -> str:
        return (
            "MÉTRICAS_DE_SINCRONIZACIÓN:\n"
            f"{json.dumps(self.sync_metrics, ensure_ascii=False, indent=2)}\n\n"
            "ESTADO_DE_RED:\n"
            f"{json.dumps(self.network_status, ensure_ascii=False, indent=2)}\n\n"
            "RESUMEN_ESCENA_3D:\n"
            f"{self.scene_summary}\n"
        )


@dataclass
class ReinforcedDecision:
    """Salida estructurada de la neurona."""

    text: str
    confidence: float
    latency_ms: float
    issues_detected: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# POOL LOCAL DE MODELO (LIGERO, ESPECÍFICO PARA ESTA NEURONA)
# ──────────────────────────────────────────────────────────────

class _LocalModelPool:
    """
    Versión reducida del pool de modelos.
    Esta neurona usa típicamente un solo modelo pequeño (qwen2.5:0.5b o gemma:270m).
    """

    def __init__(self, model_name: str = "openrouter/free") -> None:
        self.model_name = model_name
        self.llm = LLMConnector() if LLMConnector else None

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.llm is None:
            # Fallback textual si el conector no está disponible (modo offline / test)
            joined = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return f"[MODO_OFFLINE_SIMULADO]\n{joined[:1500]}"

        # Extraer el último mensaje del usuario, los anteriores van al historial
        user_input = ""
        history = []
        for m in messages:
            if m["role"] == "user":
                user_input = m["content"]
            elif m["role"] == "system":
                # Hack temporal para inyectar system prompt si es necesario,
                # aunque LLMConnector ya tiene uno, lo sobreescribimos aquí para este caso.
                original_prompt = self.llm.system_prompt
                self.llm.system_prompt = m["content"]
            else:
                history.append(m)

        resp = self.llm.generate_response(user_input, model_key=self.model_name, conversation_history=history)
        return resp


# ──────────────────────────────────────────────────────────────
# NEURONA PRINCIPAL
# ──────────────────────────────────────────────────────────────

class RF_RF_RF_RFEN1_RN_1_1_10_1:
    """
    Neurona RF_EN especializada en:
    - Tomar decisiones de sincronización a partir de métricas y contexto
    - Consultar Internet (duckduckgo-search) para detectar patrones/problemas conocidos
    - Combinar resultados locales + web + otros LLMs (stub) en una decisión reforzada
    """

    def __init__(self, model_name: str = "openrouter/free") -> None:
        self.model_pool = _LocalModelPool(model_name=model_name)
        self.ddg = DDGS() if DDGS is not None else None

    # ── Búsqueda web ───────────────────────────────────────────
    def _search_web(self, query: str, max_results: int = 4) -> str:
        if self.ddg is None:
            return "[WEB_OFFLINE]"
        results = self.ddg.text(query, max_results=max_results)
        snippets = []
        for r in results:
            body = r.get("body") or ""
            title = r.get("title") or ""
            url = r.get("href") or ""
            snippets.append(f"Título: {title}\nResumen: {body}\nURL: {url}")
        return "\n\n".join(snippets) if snippets else "[SIN_RESULTADOS_WEB]"

    # ── Stub de otros LLMs gratuitos (extensible) ──────────────
    def _other_llm_validation(self, ctx: SyncContext) -> str:
        """
        Punto de extensión para integrar otros LLMs gratuitos con acceso web.
        Por ahora devuelve un stub descriptivo para no depender de APIs externas.
        """
        return (
            "[OTRO_LLM_STUB]\n"
            "Validación heurística de sincronización basada en patrones conocidos.\n"
            f"FPS: {ctx.sync_metrics.get('fps', 'N/A')} | "
            f"latencia_media_ms: {ctx.network_status.get('avg_latency_ms', 'N/A')}"
        )

    # ── Forward principal ──────────────────────────────────────
    def forward(self, ctx: SyncContext) -> ReinforcedDecision:
        """
        Ejecuta el pipeline de refuerzo:
        1) Construye un resumen textual del contexto.
        2) Lanza búsqueda web en segundo plano (sin bloquear la lógica core).
        3) Combina:
           - estado local
           - resultados de web
           - validación de otros LLMs (stub)
        4) Genera una decisión/diagnóstico reforzado.
        """
        t0 = time.time()

        context_block = ctx.to_prompt_block()
        query = f"OpenSim metaverse real-time sync fps {ctx.sync_metrics.get('fps', '')}"
        web_info = self._search_web(query)
        other_llm = self._other_llm_validation(ctx)

        system_msg = (
            "Eres una neurona de refuerzo para sincronización en tiempo real "
            "en un metaverso 3D. Debes:\n"
            "- Analizar métricas de sincronización y red.\n"
            "- Detectar posibles cuellos de botella y riesgos.\n"
            "- Usar información de Internet como referencia, sin copiarla ciegamente.\n"
            "- Proponer acciones concretas y priorizadas.\n"
        )

        user_msg = (
            "CONTEXTO LOCAL:\n"
            f"{context_block}\n\n"
            "REFERENCIAS_WEB:\n"
            f"{web_info}\n\n"
            "OTRO_LLM_VALIDACIÓN:\n"
            f"{other_llm}\n\n"
            "Devuelve una respuesta estructurada en JSON con claves:\n"
            "  - decision_text (string)\n"
            "  - confidence (0-1)\n"
            "  - issues_detected (lista de strings)\n"
            "  - recommended_actions (lista de strings)"
        )

        raw = self.model_pool.chat(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        )

        decision = self._parse_decision(raw)
        decision.latency_ms = (time.time() - t0) * 1000.0
        return decision

    # ── Parser robusto de la salida LLM ────────────────────────
    def _parse_decision(self, raw: str) -> ReinforcedDecision:
        """
        Intenta extraer un JSON válido de la respuesta del modelo.
        Si falla, encapsula todo en `decision_text` con confianza baja.
        """
        try:
            # Buscar primer bloque JSON en la respuesta
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(raw[start: end + 1])
            else:
                data = {}
        except Exception:
            data = {}

        decision_text = data.get("decision_text", raw)
        confidence = float(data.get("confidence", 0.4))
        issues = list(data.get("issues_detected", []))
        actions = list(data.get("recommended_actions", []))

        return ReinforcedDecision(
            text=decision_text,
            confidence=max(0.0, min(1.0, confidence)),
            latency_ms=0.0,
            issues_detected=issues,
            recommended_actions=actions,
        )


# ──────────────────────────────────────────────────────────────
# PEQUEÑO TEST MANUAL
# ──────────────────────────────────────────────────────────────

def _demo() -> None:
    ctx = SyncContext(
        sync_metrics={"fps": 42.7, "target_fps": 45, "avg_frame_time_ms": 23.5},
        network_status={"avg_latency_ms": 85, "packet_loss": 0.01},
        scene_summary="Región urbana con 25 avatares, 40 objetos físicos activos.",
    )
    neuron = RF_RF_RF_RFEN1_RN_1_1_10_1()
    decision = neuron.forward(ctx)
    print("DECISIÓN REFORZADA:")
    print("Texto:", decision.text[:400], "...")
    print("Confianza:", decision.confidence)
    print("Latencia (ms):", decision.latency_ms)
    print("Issues:", decision.issues_detected)
    print("Acciones:", decision.recommended_actions)


if __name__ == "__main__":
    _demo()
