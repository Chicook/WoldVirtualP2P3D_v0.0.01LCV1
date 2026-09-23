# ─── STMRFMN_2: Orquestador Sistema LucIA ──────────────────────────
# Ciclo de vida: iniciar / turno / estado / detener
# Subsistemas: BKS (cadena), IA Free, PSN (neuronas), HRCTRC (constructor)

from __future__ import annotations

import time
import json
import random
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("STMRFMN_2")


# ═══════════════════════════════════════════════════════════════════
# CLASS 1: Configuracion - rutas, parametros y banderas
# ═══════════════════════════════════════════════════════════════════
class ClienteIPFS:
    def __init__(self, ruta_base: Path = Path(".")):
        self.ruta_base = ruta_base / "ipfs"
        self.ruta_base.mkdir(parents=True, exist_ok=True)
        self._pin_count: int = 0

    def pin(self, contenido: str, nombre: str) -> str:
        destino = self.ruta_base / f"{nombre}.json"
        destino.write_text(contenido, encoding="utf-8")
        self._pin_count += 1
        return str(destino)

    def obtener(self, nombre: str) -> Optional[str]:
        ruta = self.ruta_base / f"{nombre}.json"
        return ruta.read_text(encoding="utf-8") if ruta.exists() else None

    def estadisticas(self) -> Dict[str, Any]:
        return {"pins": self._pin_count, "ruta": str(self.ruta_base)}

    def listar_pins(self) -> List[str]:
        return sorted([p.stem for p in self.ruta_base.glob("*.json")])

    def limpiar(self) -> None:
        for p in self.ruta_base.glob("*.json"):
            p.unlink()
        self._pin_count = 0


# ═══════════════════════════════════════════════════════════════════
# CLASS 7: Telemetria - monitoreo de rendimiento y metricas
# ═══════════════════════════════════════════════════════════════════
class TelemetriaLucia:
    def __init__(self):
        self.metricas: Dict[str, List[float]] = {"latencia_turno": [], "tokens_entrada": [], "tokens_salida": [], "memoria_neuronas": []}
        self._inicio_sesion = time.time()

    def registrar_latencia(self, segundos: float) -> None:
        self.metricas["latencia_turno"].append(segundos)

    def snapshot_neuronas(self, cantidad: int) -> None:
        self.metricas["memoria_neuronas"].append(cantidad)

    def _prom(self, clave: str) -> float:
        v = self.metricas.get(clave, [])
        return sum(v) / len(v) if v else 0.0

    def reporte(self) -> Dict[str, Any]:
        return {"uptime_seg": round(time.time() - self._inicio_sesion, 2), "total_turnos": len(self.metricas["latencia_turno"]),
                "latencia_prom": round(self._prom("latencia_turno"), 4), "latencia_max": round(max(self.metricas["latencia_turno"] or [0]), 4),
                "tokens_entrada_prom": round(self._prom("tokens_entrada"), 1), "neuronas_prom": round(self._prom("memoria_neuronas"), 1)}


# ═══════════════════════════════════════════════════════════════════
# CLASS 8: Refactor + Memoria .md - persistencia estructurada
# ═══════════════════════════════════════════════════════════════════
def iniciar_lucia(ruta_base: str = ".") -> OrquestadorSistemaLucIA:
    orq = OrquestadorSistemaLucIA(ruta_base=ruta_base)
    if not orq.inicializar_subsistemas():
        raise RuntimeError("[mainLCSTM] Fallo inicializando subsistemas.")
    return orq


def detener_lucia(orq: OrquestadorSistemaLucIA) -> None:
    orq.cerrar_sistema()


def turno(orq: OrquestadorSistemaLucIA, prompt: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    resultado = orq.procesar_turno_dialogo(prompt)
    resultado["segundos"] = round(time.perf_counter() - t0, 3)
    return resultado


def estado(orq: OrquestadorSistemaLucIA) -> Dict[str, Any]:
    try:
        valida, _ = orq.servidor_bks.validar_cadena()
    except Exception:
        valida = False
    try:
        mod = orq.cliente_iafree.gestor.obtener_modelo_activo()["id"] if orq.cliente_iafree else "N/A"
    except Exception:
        mod = "N/A"
    return {"sesion": orq.config.sesion_id, "activa": orq.activa, "turnos": orq.turno_actual, "cadena_valida": bool(valida),
            "bloques": len(orq.servidor_bks.cadena), "neuronas": orq.conversor_psn.cantidad(), "modelo_activo": mod,
            "ia_local_lista": orq.ia_local_lista, "constructor_activo": bool(orq.gestor_hrctrc and orq.gestor_hrctrc.esta_activa())}


# ═══════════════════════════════════════════════════════════════════
# CLASS 10: Demo - ejecucion de ejemplo del ciclo de vida
# ═══════════════════════════════════════════════════════════════════
class DemoLucia:
    """Ejecuta un demo automatico del sistema LucIA."""

    def __init__(self):
        self.orq = iniciar_lucia()
        self.reglas_demo = [("estado", "RESPUESTA_ESTADO"), ("ayuda", "RESPUESTA_AYUDA")]
        self.prompts_demo = ["Hola Lucia", "Como estas", "Cuentame un dato", "Cual es tu estado", "Necesito ayuda"]

    def configurar(self) -> None:
        for cond, acc in self.reglas_demo:
            self.orq.gestor_hrctrc.agregar_regla(cond, acc)

    def ejecutar(self) -> None:
        print("=== STMRFMN_2 - Demo Ciclo de Vida LucIA ===")
        self.configurar()
        for p in self.prompts_demo:
            r = turno(self.orq, p)
            print(f"  Turno {r['turno']}: {r['respuesta'][:50]}... ({r['segundos']}s)")
        print(f"  Estado: {estado(self.orq)}")
        print(f"  Telemetria: {self.orq.telemetria.reporte()}")

    def cerrar(self) -> None:
        detener_lucia(self.orq)
        print("=== Sistema detenido correctamente ===")


# ═══════════════════════════════════════════════════════════════════
# Punto de entrada - ejecucion del demo
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo = DemoLucia()
    demo.ejecutar()
    demo.cerrar()
from LC.celebro.CMFG.SBSTM.STMRFMN.STMRFMN_2_ConfiguracionLucia import ConfiguracionLucia  # CLASSPACK
from LC.celebro.CMFG.SBSTM.STMRFMN.STMRFMN_2_ServidorBKS import ServidorBKS  # CLASSPACK
from LC.celebro.CMFG.SBSTM.STMRFMN.STMRFMN_2_ConversorPSN import ConversorPSN  # CLASSPACK
from LC.celebro.CMFG.SBSTM.STMRFMN.STMRFMN_2_GestorHRCTRC import GestorHRCTRC  # CLASSPACK
from LC.celebro.CMFG.SBSTM.STMRFMN.STMRFMN_2_ClienteIAFree import ClienteIAFree  # CLASSPACK
from LC.celebro.CMFG.SBSTM.STMRFMN.STMRFMN_2_GestorMemoria import GestorMemoria  # CLASSPACK
from LC.celebro.CMFG.SBSTM.STMRFMN.STMRFMN_2_OrquestadorSistemaLucIA import OrquestadorSistemaLucIA  # CLASSPACK
