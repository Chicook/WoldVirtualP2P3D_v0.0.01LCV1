# ─── STMRFMN_2: Orquestador Sistema LucIA ──────────────────────────
# Ciclo de vida: iniciar / turno / estado / detener
# Subsistemas: BKS (cadena), IA Free, PSN (neuronas), HRCTRC (constructor)

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
class ConfiguracionLucia:
    def __init__(self, ruta_base: str = ".", sesion_id: Optional[str] = None):
        self.ruta_base = Path(ruta_base)
        self.sesion_id = sesion_id or hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
        self.max_turnos: int = 1000
        self.max_neuronas: int = 5000
        self.ruta_pesos: Path = self.ruta_base / "pesos"
        self.ruta_md: Path = self.ruta_base / "memoria.md"
        self.ruta_ipfs: Path = self.ruta_base / "ipfs"
        self.ia_local_lista: bool = True
        self.debug: bool = False

    def validar_rutas(self) -> bool:
        for ruta in (self.ruta_pesos, self.ruta_ipfs):
            ruta.mkdir(parents=True, exist_ok=True)
        return True

    def como_diccionario(self) -> Dict[str, Any]:
        return {"sesion_id": self.sesion_id, "max_turnos": self.max_turnos,
                "max_neuronas": self.max_neuronas, "ia_local_lista": self.ia_local_lista, "debug": self.debug}

    def __repr__(self) -> str:
        return f"ConfiguracionLucia(sesion={self.sesion_id}, rutas={self.ruta_base})"

    def reset(self) -> None:
        """Resetea los valores configurables a sus defaults."""
        self.max_turnos = 1000
        self.max_neuronas = 5000
        self.ia_local_lista = True
        self.debug = False


# ═══════════════════════════════════════════════════════════════════
# CLASS 2: Servidor BKS - cadena de bloques ligera para trazabilidad
# ═══════════════════════════════════════════════════════════════════
class ServidorBKS:
    def __init__(self):
        self.cadena: List[Dict[str, Any]] = []
        self._ultimo_hash: str = "0" * 64

    def _hash_bloque(self, bloque: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(bloque, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def agregar(self, datos: Dict[str, Any]) -> str:
        bloque = {"index": len(self.cadena), "timestamp": time.time(), "datos": datos, "prev_hash": self._ultimo_hash}
        bloque["hash"] = self._hash_bloque(bloque)
        self._ultimo_hash = bloque["hash"]
        self.cadena.append(bloque)
        return bloque["hash"]

    def obtener_bloque(self, index: int) -> Optional[Dict[str, Any]]:
        if 0 <= index < len(self.cadena):
            return self.cadena[index]
        return None

    def profundidad(self) -> int:
        return len(self.cadena)

    def validar_cadena(self) -> Tuple[bool, str]:
        for i in range(1, len(self.cadena)):
            a, b = self.cadena[i - 1], self.cadena[i]
            if b["prev_hash"] != a["hash"] or b["hash"] != self._hash_bloque(b):
                return False, f"Incoherencia en bloque {i}"
        return True, "OK"

    def limpiar(self) -> None:
        self.cadena.clear()
        self._ultimo_hash = "0" * 64


# ═══════════════════════════════════════════════════════════════════
# CLASS 3: Conversor PSN - neuronas / embedding vectorial basico
# ═══════════════════════════════════════════════════════════════════
class ConversorPSN:
    def __init__(self, max_neuronas: int = 5000):
        self.neuronas: Dict[str, List[float]] = {}
        self.max_neuronas: int = max_neuronas
        self._dim: int = 128

    def _embedding_aleatorio(self, texto: str) -> List[float]:
        semilla = int(hashlib.sha256(texto.encode()).hexdigest(), 16) % (2**31)
        return [random.Random(semilla).uniform(-1, 1) for _ in range(self._dim)]

    def almacenar(self, clave: str, texto: str) -> None:
        if len(self.neuronas) >= self.max_neuronas:
            del self.neuronas[next(iter(self.neuronas))]
        self.neuronas[clave] = self._embedding_aleatorio(texto)

    def obtener(self, clave: str) -> Optional[List[float]]:
        return self.neuronas.get(clave)

    def cantidad(self) -> int:
        return len(self.neuronas)

    def listar_claves(self) -> List[str]:
        return list(self.neuronas.keys())

    def limpiar(self) -> None:
        self.neuronas.clear()


# ═══════════════════════════════════════════════════════════════════
# CLASS 4: Gestor HRCTRC - constructor de jerarquia de respuesta
# ═══════════════════════════════════════════════════════════════════
class GestorHRCTRC:
    def __init__(self):
        self.reglas: List[Dict[str, Any]] = []
        self._activa: bool = False

    def activar(self) -> None:
        self._activa = True
        logger.info("[HRCTRC] Constructor activado.")

    def desactivar(self) -> None:
        self._activa = False

    def esta_activa(self) -> bool:
        return self._activa

    def agregar_regla(self, condicion: str, accion: str) -> None:
        self.reglas.append({"condicion": condicion, "accion": accion})

    def listar_reglas(self) -> List[Dict[str, str]]:
        return list(self.reglas)

    def eliminar_regla(self, condicion: str) -> bool:
        for i, r in enumerate(self.reglas):
            if r["condicion"] == condicion:
                del self.reglas[i]
                return True
        return False

    def evaluar(self, prompt: str) -> Optional[str]:
        for regla in self.reglas:
            if regla["condicion"] in prompt:
                return regla["accion"]
        return None


# ═══════════════════════════════════════════════════════════════════
# CLASS 5: Cliente IA Free - interfaz al modelo de IA local/remoto
# ═══════════════════════════════════════════════════════════════════
class ClienteIAFree:
    def __init__(self):
        self.gestor = self._GestorModelos()
        self._modelo_activo_id: str = "lucia-default-v1"

    class _GestorModelos:
        def __init__(self):
            self._modelos = [
                {"id": "lucia-default-v1", "nombre": "LucIA Default", "parametros": "7B"},
                {"id": "lucia-flash-v2", "nombre": "LucIA Flash", "parametros": "3B"},
            ]

        def obtener_modelo_activo(self) -> Dict[str, str]:
            return next((m for m in self._modelos if m["id"] == "lucia-default-v1"), self._modelos[0])

        def listar_modelos(self) -> List[Dict[str, str]]:
            return list(self._modelos)

    def consultar(self, prompt: str, max_tokens: int = 256) -> str:
        logger.info(f"[IA Free] Consulta: {prompt[:60]}...")
        return f"[Respuesta simulada a: '{prompt[:40]}...']"

    def cambiar_modelo(self, modelo_id: str) -> bool:
        modelos = self.gestor.listar_modelos()
        if any(m["id"] == modelo_id for m in modelos):
            self._modelo_activo_id = modelo_id
            return True
        return False


# ═══════════════════════════════════════════════════════════════════
# CLASS 6: IPFS - cliente ligero para almacenamiento descentralizado
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
class GestorMemoria:
    def __init__(self, ruta_md: Path):
        self.ruta_md = ruta_md
        self.historial: deque = deque(maxlen=500)

    def registrar(self, turno: int, prompt: str, respuesta: str) -> None:
        self.historial.append({"turno": turno, "timestamp": datetime.now().isoformat(), "prompt": prompt, "respuesta": respuesta})

    def refactor(self) -> None:
        if len(self.historial) < 2:
            return
        compacto = [self.historial[0]]
        for e in list(self.historial)[1:]:
            if e["prompt"] != compacto[-1]["prompt"]:
                compacto.append(e)
        self.historial.clear()
        self.historial.extend(compacto)
        logger.info(f"[Memoria] Refactor: {len(self.historial)} entradas.")

    def persistir(self) -> None:
        lineas = ["# Memoria Sistema LucIA", f"_Generado: {datetime.now().isoformat()}_", ""]
        for e in self.historial:
            lineas += [f"## Turno {e['turno']} ({e['timestamp']})", f"**P:** {e['prompt']}", f"**R:** {e['respuesta']}", ""]
        self.ruta_md.write_text("\n".join(lineas), encoding="utf-8")
        logger.info(f"[Memoria] Guardado en {self.ruta_md}")

    def exportar_historial(self, ruta: Path) -> None:
        datos = [{"turno": e["turno"], "timestamp": e["timestamp"], "prompt": e["prompt"], "respuesta": e["respuesta"]} for e in self.historial]
        ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"[Memoria] Historial exportado a {ruta}")


# ═══════════════════════════════════════════════════════════════════
# CLASS 9: Orquestador principal - coordina todos los subsistemas
# ═══════════════════════════════════════════════════════════════════
class OrquestadorSistemaLucIA:
    def __init__(self, ruta_base: str = "."):
        self.config = ConfiguracionLucia(ruta_base=ruta_base)
        self.servidor_bks = ServidorBKS()
        self.conversor_psn = ConversorPSN(max_neuronas=self.config.max_neuronas)
        self.gestor_hrctrc = GestorHRCTRC()
        self.cliente_iafree = ClienteIAFree()
        self.ipfs = ClienteIPFS(Path(ruta_base))
        self.memoria = GestorMemoria(self.config.ruta_md)
        self.telemetria = TelemetriaLucia()
        self.activa: bool = False
        self.turno_actual: int = 0
        self.ia_local_lista: bool = True
        self._lock = threading.Lock()

    def inicializar_subsistemas(self) -> bool:
        try:
            self.config.validar_rutas()
            self.servidor_bks.agregar({"evento": "inicio", "sesion": self.config.sesion_id})
            self.gestor_hrctrc.activar()
            self.activa = True
            self.turno_actual = 0
            logger.info(f"[Orquestador] Sistema iniciado - sesion {self.config.sesion_id}")
            return True
        except Exception as e:
            logger.error(f"[Orquestador] Error en init: {e}")
            return False

    def procesar_turno_dialogo(self, prompt: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        with self._lock:
            self.turno_actual += 1
            hash_bloque = self.servidor_bks.agregar({"turno": self.turno_actual, "prompt": prompt})
            self.conversor_psn.almacenar(f"turno_{self.turno_actual}", prompt)
            accion = self.gestor_hrctrc.evaluar(prompt)
            respuesta = f"[HRCTRC->{accion}]" if accion else self.cliente_iafree.consultar(prompt)
            self.memoria.registrar(self.turno_actual, prompt, respuesta)
            latencia = time.perf_counter() - t0
            self.telemetria.registrar_latencia(latencia)
            self.telemetria.snapshot_neuronas(self.conversor_psn.cantidad())
            return {"turno": self.turno_actual, "hash": hash_bloque, "respuesta": respuesta, "neuronas": self.conversor_psn.cantidad(), "latencia": round(latencia, 4)}

    def obtener_telemetria(self) -> Dict[str, Any]:
        return self.telemetria.reporte()

    def exportar_metricas(self, ruta: Optional[str] = None) -> None:
        destino = Path(ruta) if ruta else self.config.ruta_base / "telemetria.json"
        datos = {"metricas": self.telemetria.metricas, "resumen": self.telemetria.reporte(), "exportado": datetime.now().isoformat()}
        destino.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"[Telemetria] Exportado a {destino}")

    def estado_interno(self) -> Dict[str, Any]:
        base = estado(self)
        base["telemetria"] = self.telemetria.reporte()
        base["ipfs"] = self.ipfs.estadisticas()
        base["config"] = self.config.como_diccionario()
        return base

    def reinicializar(self) -> bool:
        self.cerrar_sistema()
        self.servidor_bks.limpiar()
        self.conversor_psn.limpiar()
        self.gestor_hrctrc.reglas.clear()
        self.ipfs.limpiar()
        self.memoria.historial.clear()
        self.telemetria = TelemetriaLucia()
        self.turno_actual = 0
        return self.inicializar_subsistemas()

    def pausar(self) -> None:
        self.activa = False
        logger.info("[Orquestador] Sistema pausado.")

    def reanudar(self) -> None:
        self.activa = True
        logger.info("[Orquestador] Sistema reanudado.")

    def cerrar_sistema(self) -> None:
        self.activa = False
        self.memoria.refactor()
        self.memoria.persistir()
        self.servidor_bks.agregar({"evento": "cierre", "turno_final": self.turno_actual})
        valida, _ = self.servidor_bks.validar_cadena()
        logger.info(f"[Orquestador] Cerrado. Turnos={self.turno_actual}, ChainValid={valida}")


# ═══════════════════════════════════════════════════════════════════
# Funciones de utilidad publicas (API del ciclo de vida)
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