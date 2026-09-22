"""Aprendizaje seguro sobre los pesos propios de LucIA.

La fuente puede ser Ollama, LM Studio u OpenRouter. Este módulo nunca intenta
editar los pesos del proveedor externo: solo protege y actualiza PSNRCV.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[4]
PSNRL_DIR = ROOT_DIR / "LC" / "celebro" / "PSNRL"
SNAPSHOT = PSNRL_DIR / "lucia_pesos_previos.npz"
SNAPSHOT_META = PSNRL_DIR / "lucia_pesos_previos.json"
REPLAY = PSNRL_DIR / "experiencias_lucia.jsonl"
WEIGHT_ATTRS = (
    "pesos", "sesgo", "q_online", "q_objetivo", "pesos_actor",
    "pesos_critic", "q_table", "pesos_actor_global", "pesos_q_principal",
)

SPECIALTY_KEYWORDS = {
    "EN": {"pregunta", "entrada", "texto", "percepcion", "sensacion", "contexto"},
    "RFSL": {"explica", "razon", "codigo", "estructura", "analisis", "aprendizaje"},
    "RFEN": {"decide", "accion", "objetivo", "plan", "prioridad", "recompensa"},
    "RNP": {"memoria", "peso", "neural", "optimizacion", "sinapsis", "patron"},
    "SLRN": {"responde", "sintesis", "generaliza", "regla", "solucion", "lenguaje"},
}


class AprendizajeSeguroLucIA:
    """Convierte una respuesta externa en aprendizaje controlado de LucIA."""

    def __init__(self, ewc_lambda: float = 0.02) -> None:
        self.ewc_lambda = max(0.0, min(1.0, float(ewc_lambda)))
        self.ultima_validacion: Dict[str, Any] = {}

    @staticmethod
    def _respuesta_valida(texto: str) -> bool:
        if not texto or not texto.strip() or len(texto) > 12000:
            return False
        return not bool(re.search(
            r'"name"\s*:\s*"(?:web_search|browser|search|function)"|\b(?:tool_call|function_call)\b',
            texto, re.IGNORECASE,
        ))

    def validar_alineacion(self, pregunta: str, respuesta_fuente: str) -> Dict[str, Any]:
        aprobado = bool((pregunta or "").strip()) and len(pregunta) <= 12000 and self._respuesta_valida(respuesta_fuente)
        resultado = {
            "aprobado": aprobado,
            "pregunta_valida": bool((pregunta or "").strip()),
            "respuesta_valida": self._respuesta_valida(respuesta_fuente),
            "motivo": "ok" if aprobado else "respuesta_no_apta_para_aprender",
        }
        self.ultima_validacion = resultado
        return resultado

    @staticmethod
    def _capturar(conversor: Any) -> Dict[str, Any]:
        arrays: Dict[str, np.ndarray] = {}
        mapa: Dict[str, Dict[str, str]] = {}
        for nombre, neurona in getattr(conversor, "neuronas", {}).items():
            for attr in WEIGHT_ATTRS:
                valor = getattr(neurona, attr, None)
                if isinstance(valor, np.ndarray):
                    clave = f"n::{nombre}::{attr}"
                    arrays[clave] = valor.copy()
                    mapa[clave] = {"tipo": "neurona", "nombre": nombre, "attr": attr}
        for attr in ("_soap_L", "_soap_R", "_grad_sum", "_grad_sq_sum", "_ema_vector"):
            valor = getattr(conversor, attr, None)
            if isinstance(valor, np.ndarray):
                clave = f"c::{attr}"
                arrays[clave] = valor.copy()
                mapa[clave] = {"tipo": "conversor", "attr": attr}
        return {
            "arrays": arrays,
            "mapa": mapa,
            "deriva": float(getattr(conversor, "deriva_acumulada", 0.0)),
            "pasos": int(getattr(conversor, "pasos_sesion", 0)),
            "historial": len(getattr(conversor, "historial_actualizaciones", [])),
        }

    @staticmethod
    def _restaurar(conversor: Any, estado: Dict[str, Any]) -> None:
        for clave, valor in estado["arrays"].items():
            meta = estado["mapa"][clave]
            if meta["tipo"] == "neurona":
                objetivo = conversor.neuronas.get(meta["nombre"])
            else:
                objetivo = conversor
            if objetivo is not None:
                setattr(objetivo, meta["attr"], valor.copy())
        conversor.deriva_acumulada = estado["deriva"]
        conversor.pasos_sesion = estado["pasos"]
        historial = getattr(conversor, "historial_actualizaciones", None)
        if isinstance(historial, list):
            del historial[estado["historial"]:]

    def _guardar_snapshot(self, estado: Dict[str, Any]) -> None:
        PSNRL_DIR.mkdir(parents=True, exist_ok=True)
        temporal = PSNRL_DIR / "lucia_pesos_previos.tmp.npz"
        np.savez_compressed(temporal, **estado["arrays"])
        os.replace(temporal, SNAPSHOT)
        meta = {k: v for k, v in estado.items() if k != "arrays"}
        temporal_meta = PSNRL_DIR / "lucia_pesos_previos.tmp.json"
        temporal_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        os.replace(temporal_meta, SNAPSHOT_META)

    def _regularizar_y_limitar(self, antes: Dict[str, Any], conversor: Any) -> float:
        """Limita el paso y conserva una fracción del estado anterior (EWC ligero)."""
        delta_total = 0.0
        tensores = 0
        for clave, previo in antes["arrays"].items():
            meta = antes["mapa"][clave]
            objetivo = conversor.neuronas.get(meta["nombre"]) if meta["tipo"] == "neurona" else conversor
            actual = getattr(objetivo, meta["attr"], None) if objetivo is not None else None
            if not isinstance(actual, np.ndarray) or actual.shape != previo.shape:
                continue
            delta = actual.astype(np.float32) - previo.astype(np.float32)
            norma = float(np.linalg.norm(delta))
            limite = max(0.05, float(np.linalg.norm(previo)) * 0.10 + 0.05)
            if norma > limite:
                delta *= limite / max(norma, 1e-8)
            importancia = np.minimum(1.0, np.abs(previo).astype(np.float32) / 0.05)
            delta *= 1.0 - self.ewc_lambda * importancia
            nuevo = np.clip(previo.astype(np.float32) + delta, -10.0, 10.0).astype(actual.dtype)
            setattr(objetivo, meta["attr"], nuevo)
            delta_total += float(np.linalg.norm(nuevo - previo))
            tensores += 1
        return delta_total / max(1, tensores)

    @staticmethod
    def _calcular_relevancias(pregunta: str, respuesta: str, nombres: List[str]) -> Dict[str, float]:
        """Calcula una intensidad determinista por especialidad, sin dejar neuronas fuera."""
        tokens = set(re.findall(r"[a-záéíóúñ0-9_]+", f"{pregunta} {respuesta}".lower()))
        relevancias: Dict[str, float] = {}
        for nombre in nombres:
            grupo = "SLRN"
            for candidato in SPECIALTY_KEYWORDS:
                if nombre.upper().startswith(candidato):
                    grupo = candidato
                    break
            aciertos = len(tokens.intersection(SPECIALTY_KEYWORDS[grupo]))
            # Variación pequeña y reproducible entre neuronas de la misma capa.
            dispersor = (sum(ord(c) for c in nombre) % 11) / 100.0
            relevancias[nombre] = round(min(1.0, 0.35 + 0.14 * aciertos + dispersor), 6)
        return relevancias

    def _aplicar_reparto_semantico(self, antes: Dict[str, Any], conversor: Any,
                                   pregunta: str, respuesta: str) -> Dict[str, float]:
        """Reescala el delta por relevancia y mantiene activas las 50 neuronas."""
        nombres = list(getattr(conversor, "neuronas", {}).keys())
        relevancias = self._calcular_relevancias(pregunta, respuesta, nombres)
        for clave, previo in antes["arrays"].items():
            meta = antes["mapa"][clave]
            if meta["tipo"] != "neurona":
                continue
            neurona = conversor.neuronas.get(meta["nombre"])
            actual = getattr(neurona, meta["attr"], None) if neurona is not None else None
            if not isinstance(actual, np.ndarray) or actual.shape != previo.shape:
                continue
            delta = actual.astype(np.float32) - previo.astype(np.float32)
            nuevo = previo.astype(np.float32) + delta * relevancias.get(meta["nombre"], 0.35)
            setattr(neurona, meta["attr"], nuevo.astype(actual.dtype))
        return relevancias

    @staticmethod
    def _pesos_validos(conversor: Any) -> bool:
        for neurona in getattr(conversor, "neuronas", {}).values():
            for attr in WEIGHT_ATTRS:
                valor = getattr(neurona, attr, None)
                if isinstance(valor, np.ndarray) and not np.all(np.isfinite(valor)):
                    return False
        return True

    def _guardar_experiencia(self, pregunta: str, respuesta: str, modelo: str, info: Dict[str, Any]) -> None:
        PSNRL_DIR.mkdir(parents=True, exist_ok=True)
        registro = {
            "timestamp": time.time(),
            "pregunta": pregunta[:3000],
            "respuesta_fuente": respuesta[:6000],
            "respuesta_sha256": hashlib.sha256(respuesta.encode("utf-8")).hexdigest(),
            "modelo_origen": modelo,
            "total_neuronas": info.get("total_neuronas", 0),
            "norma_delta_aplicada": info.get("norma_delta_aplicada", 0.0),
            "tono_cognitivo": info.get("tono_cognitivo", ""),
        }
        with REPLAY.open("a", encoding="utf-8") as archivo:
            archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")

    def asimilar(self, conversor: Any, pregunta: str, respuesta_fuente: str,
                 modelo_origen: str) -> Dict[str, Any]:
        """Aplica la respuesta fuente a LucIA y revierte si el gate falla."""
        gate = self.validar_alineacion(pregunta, respuesta_fuente)
        if not gate["aprobado"]:
            return {"aceptado": False, "rollback": False, "gate": gate}
        antes = self._capturar(conversor)
        self._guardar_snapshot(antes)
        try:
            info = conversor.asimilar_respuestas_y_calcular_sintesis(
                pregunta, respuesta_fuente, modelo_origen)
            relevancias = self._aplicar_reparto_semantico(
                antes, conversor, pregunta, respuesta_fuente)
            delta_medio = self._regularizar_y_limitar(antes, conversor)
            if not self._pesos_validos(conversor):
                raise ValueError("pesos no finitos")
            info = dict(info)
            info.update({"aceptado": True, "rollback": False, "gate": gate,
                         "delta_medio_seguro": round(delta_medio, 6),
                         "reparto_neuronal": {
                             "total_neuronas": len(relevancias),
                             "neuronas_actualizadas": sum(1 for x in relevancias.values() if x > 0),
                             "relevancia_min": min(relevancias.values(), default=0.0),
                             "relevancia_max": max(relevancias.values(), default=0.0),
                             "relevancias": relevancias,
                         }})
            self._guardar_experiencia(pregunta, respuesta_fuente, modelo_origen, info)
            self.ultima_validacion = info
            conversor.persistir_pesos_en_psnrl()
            return info
        except Exception as exc:
            self._restaurar(conversor, antes)
            try:
                conversor.persistir_pesos_en_psnrl(etiqueta="rollback_seguro")
            except Exception:
                pass
            resultado = {"aceptado": False, "rollback": True, "gate": gate,
                         "motivo": f"rollback: {type(exc).__name__}: {exc}"}
            self.ultima_validacion = resultado
            return resultado

    def restaurar_snapshot(self, conversor: Any) -> Dict[str, Any]:
        """Restaura el último snapshot persistido, incluso tras reiniciar LucIA."""
        if not SNAPSHOT.exists() or not SNAPSHOT_META.exists():
            return {"restaurado": False, "motivo": "snapshot_no_disponible"}
        try:
            meta = json.loads(SNAPSHOT_META.read_text(encoding="utf-8"))
            with np.load(SNAPSHOT, allow_pickle=False) as datos:
                estado = {
                    "arrays": {clave: datos[clave].copy() for clave in datos.files},
                    "mapa": meta["mapa"],
                    "deriva": meta["deriva"],
                    "pasos": meta["pasos"],
                    "historial": meta["historial"],
                }
            self._restaurar(conversor, estado)
            conversor.persistir_pesos_en_psnrl(etiqueta="restaurado_snapshot")
            return {"restaurado": True, "snapshot": str(SNAPSHOT)}
        except Exception as exc:
            return {"restaurado": False, "motivo": f"{type(exc).__name__}: {exc}"}

    def replayar(self, conversor: Any, limite: int = 2) -> Dict[str, Any]:
        """Reaplica las últimas experiencias aceptadas con el mismo gate seguro."""
        registros = [r for r in self.leer_experiencias(limite) if r.get("respuesta_fuente")]
        resultados: List[Dict[str, Any]] = []
        for registro in registros:
            resultados.append(self.asimilar(
                conversor=conversor,
                pregunta=str(registro.get("pregunta", "")),
                respuesta_fuente=str(registro.get("respuesta_fuente", "")),
                modelo_origen=f"replay:{registro.get('modelo_origen', 'desconocido')}",
            ))
        return {"replay": True, "solicitadas": len(registros),
                "aceptadas": sum(1 for r in resultados if r.get("aceptado")),
                "resultados": resultados}

    def leer_experiencias(self, limite: int = 10) -> List[Dict[str, Any]]:
        if not REPLAY.exists():
            return []
        salida: List[Dict[str, Any]] = []
        for linea in REPLAY.read_text(encoding="utf-8").splitlines()[-max(1, limite):]:
            try:
                salida.append(json.loads(linea))
            except json.JSONDecodeError:
                pass
        return salida


_APRENDIZAJE: Optional[AprendizajeSeguroLucIA] = None


def get_aprendizaje_seguro() -> AprendizajeSeguroLucIA:
    global _APRENDIZAJE
    if _APRENDIZAJE is None:
        _APRENDIZAJE = AprendizajeSeguroLucIA()
    return _APRENDIZAJE
