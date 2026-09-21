"""
lucIA.CORE.memory_manager - Memoria Persistente de Conversacion
================================================================

Ventana deslizante de historial de chat persistida en disco (JSON)
para que LucIA recuerde conversaciones entre sesiones.

Caracteristicas:
  - Persiste en Celebro/memoria_conversacion.json.
  - Ventana deslizante: mantiene los ultimos 30 turnos.
  - Comprime contexto antiguo automaticamente.
  - Exporta mensajes compatibles con OpenAI API.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("lucIA.MemoryManager")

_LUCIA_DIR = Path(__file__).parent.parent.resolve()
MEMORIA_PATH = _LUCIA_DIR / "Celebro" / "memoria_conversacion.json"

MAX_TURNOS_VENTANA = 30
MAX_CHARS_POR_TURNO = 800
MAX_TURNOS_EN_CONTEXTO_API = 10


class MemoryManager:
    """Gestiona la memoria conversacional persistente de LucIA."""

    def __init__(self, ruta: Optional[Path] = None):
        self.ruta = ruta or MEMORIA_PATH
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._datos: Dict[str, Any] = self._cargar()

    def _cargar(self) -> Dict[str, Any]:
        """Carga el historial desde disco."""
        if self.ruta.exists():
            try:
                with open(self.ruta, "r", encoding="utf-8") as f:
                    datos = json.load(f)
                    logger.info(
                        f"Memoria cargada: {len(datos.get('turnos', []))} turnos previos."
                    )
                    return datos
            except Exception as e:
                logger.warning(f"Error al cargar memoria, iniciando nueva: {e}")
        return {
            "version": "1.0",
            "turnos": [],
            "resumen_contexto": "",
            "total_sesiones": 0,
            "ultima_sesion": None,
        }

    def guardar(self) -> None:
        """Persiste el historial en disco."""
        try:
            with open(self.ruta, "w", encoding="utf-8") as f:
                json.dump(self._datos, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error al guardar memoria: {e}")

    def registrar_turno(
        self,
        pregunta: str,
        respuesta: str,
        modelo: str = "LucIA_Core",
        estado_emocional: float = 0.0,
    ) -> None:
        """Registra un turno completo (usuario -> LucIA) en la memoria."""
        turno = {
            "ts": time.time(),
            "user": pregunta[:MAX_CHARS_POR_TURNO],
            "lucia": respuesta[:MAX_CHARS_POR_TURNO],
            "modelo": modelo,
            "emocion": round(estado_emocional, 3),
        }
        self._datos["turnos"].append(turno)
        self._datos["ultima_sesion"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if len(self._datos["turnos"]) > MAX_TURNOS_VENTANA:
            excedente = self._datos["turnos"][:-MAX_TURNOS_VENTANA]
            self._datos["turnos"] = self._datos["turnos"][-MAX_TURNOS_VENTANA:]
            self._comprimir_excedente(excedente)
        self.guardar()

    def _comprimir_excedente(self, turnos_antiguos: List[Dict]) -> None:
        """Comprime turnos que salen de la ventana en un resumen de texto."""
        if not turnos_antiguos:
            return
        lineas = []
        for t in turnos_antiguos[-5:]:
            lineas.append(f"Usuario: {t['user'][:120]}")
            lineas.append(f"LucIA: {t['lucia'][:150]}")
        nuevo_resumen = "\n".join(lineas)
        resumen_previo = self._datos.get("resumen_contexto", "")
        combinado = (resumen_previo + "\n" + nuevo_resumen).strip()
        self._datos["resumen_contexto"] = combinado[-1500:]

    def obtener_mensajes_para_api(self) -> List[Dict[str, str]]:
        """
        Devuelve los ultimos turnos formateados como lista de mensajes
        compatibles con la API de OpenAI/LM Studio.
        """
        mensajes: List[Dict[str, str]] = []
        resumen = self._datos.get("resumen_contexto", "").strip()
        if resumen:
            mensajes.append({
                "role": "system",
                "content": (
                    "[CONTEXTO DE CONVERSACIONES ANTERIORES - usa esto para dar continuidad]:\n"
                    + resumen
                ),
            })
        ventana = self._datos["turnos"][-MAX_TURNOS_EN_CONTEXTO_API:]
        for turno in ventana:
            mensajes.append({"role": "user", "content": turno["user"]})
            mensajes.append({"role": "assistant", "content": turno["lucia"]})
        return mensajes

    def obtener_resumen_sesion(self) -> str:
        """Resumen legible del estado de la memoria para mostrar al usuario."""
        total = len(self._datos["turnos"])
        ultima = self._datos.get("ultima_sesion", "nunca")
        tiene_resumen = bool(self._datos.get("resumen_contexto", "").strip())
        return (
            f"Turnos en memoria activa: {total} | "
            f"Ultima sesion: {ultima} | "
            f"Contexto historico comprimido: {'si' if tiene_resumen else 'no'}"
        )

    def iniciar_nueva_sesion(self) -> None:
        """Marca el inicio de una nueva sesion."""
        self._datos["total_sesiones"] = self._datos.get("total_sesiones", 0) + 1
        self._datos["ultima_sesion"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.guardar()
        logger.info(
            f"Memoria: sesion #{self._datos['total_sesiones']} iniciada. "
            f"{len(self._datos['turnos'])} turnos previos cargados."
        )

    def limpiar_memoria(self) -> None:
        """Borra completamente la memoria conversacional."""
        self._datos = {
            "version": "1.0",
            "turnos": [],
            "resumen_contexto": "",
            "total_sesiones": 0,
            "ultima_sesion": None,
        }
        self.guardar()
        logger.info("Memoria conversacional borrada completamente.")

    @property
    def num_turnos(self) -> int:
        return len(self._datos["turnos"])


_memoria_global: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Devuelve la instancia singleton del MemoryManager."""
    global _memoria_global
    if _memoria_global is None:
        _memoria_global = MemoryManager()
    return _memoria_global
