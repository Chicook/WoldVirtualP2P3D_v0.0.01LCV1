"""
pesos_vivos.py - Monitor y Gestor de Dinamica de Pesos Vivos en Terminal (2026)
================================================================================
Orquesta la inyeccion, telemetria sinaptica en tiempo real y visualizacion por terminal
de las 50 neuronas de la arquitectura WoldVirtualP2P3D (ENRN, RF_SL, RF_EN, RNP, SLRN)
y su persistencia automatica en PSNRL / IPFS.

Tecnologias 2026 integradas:
- Gradient Signal-to-Noise Ratio (GSNR) adaptativo por subsistema.
- Muon Newton-Schulz 5 espectral drift tracker.
- Entropia de Shannon y actividad sinaptica normalizada.
- Dashboard ANSI de alto impacto visual para terminal interactiva.
- Registro y checkpoint seguro en LC/celebro/PSNRL.
- Diagnostico autonomo de estabilidad y alertas de saturacion de gradiente.
"""
from __future__ import annotations
import os, sys, time, math, json, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

CURRENT_FILE = Path(__file__).resolve()
CMFG_DIR = CURRENT_FILE.parent
CELEBRO_DIR = CMFG_DIR.parent
ROOT_DIR = CELEBRO_DIR.parent.parent
PSNRL_DIR = CELEBRO_DIR / "PSNRL"
PSNRL_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Configurar stdout y stderr para codificacion segura en terminal Windows
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

logger = logging.getLogger("WoldVirtualP2P3D.PesosVivos")


# ===========================================================================
# CONSTANTES DE ESTILIZACION ANSI PARA TERMINAL 2026
# ===========================================================================
class ANSI:
    """Secuencias de escape ANSI para interfaz enriquecida en terminal."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    MAGENTA = "\033[95m"
    RED     = "\033[91m"
    WHITE   = "\033[97m"
    BG_DARK = "\033[48;5;234m"
    CLEAR   = "\033[2K"


# ===========================================================================
# METRICAS DE DINAMICA SINAPTICA 2026 (GSNR, Muon Drift, Entropia)
# ===========================================================================
class MetricasDinamicasPesos:
    """Calcula metricas avanzadas de flujo de pesos neuronales en vivo."""
    EPS: float = 1e-12

    @staticmethod
    def entropia_shannon(vector: np.ndarray) -> float:
        """Calcula la entropia de distribucion de pesos normalizada [0, 1]."""
        v = np.abs(vector.ravel())
        s = float(np.sum(v))
        if s < MetricasDinamicasPesos.EPS:
            return 0.0
        p = v / s
        p = p[p > 0]
        h = -float(np.sum(p * np.log2(p + 1e-15)))
        max_h = math.log2(len(v)) if len(v) > 1 else 1.0
        return float(np.clip(h / max(max_h, 1e-5), 0.0, 1.0))

    @staticmethod
    def deriva_espectral_muon(W: np.ndarray, delta: np.ndarray) -> float:
        """Estima la alineacion ortogonal Muon (variacion espectral) entre pesos y delta."""
        if W.ndim != 2 or delta.ndim != 2 or W.shape != delta.shape:
            return float(np.linalg.norm(delta) / (np.linalg.norm(W) + 1e-8))
        norm_w = np.linalg.norm(W, ord="fro") + MetricasDinamicasPesos.EPS
        norm_d = np.linalg.norm(delta, ord="fro") + MetricasDinamicasPesos.EPS
        cos_sim = float(np.trace(W.T @ delta) / (norm_w * norm_d))
        return float(np.clip(abs(cos_sim), 0.0, 1.0))

    @staticmethod
    def barra_progreso(valor: float, max_val: float = 1.0, ancho: int = 15,
                       color: str = ANSI.CYAN) -> str:
        """Genera una barra de telemetria estilizada en bloques Unicode."""
        ratio = max(0.0, min(1.0, valor / max(max_val, MetricasDinamicasPesos.EPS)))
        llenos = int(round(ratio * ancho))
        vacios = ancho - llenos
        bloques = "█" * llenos + "░" * vacios
        return f"{color}{bloques}{ANSI.RESET} {ratio*100:5.1f}%"

    @staticmethod
    def calcular_indice_estabilidad(gsnr: float, entropia: float, deriva: float) -> Tuple[float, str]:
        """Calcula el indice de salud sinaptica compuesto 2026."""
        estabilidad = float(np.clip(0.4 * min(gsnr / 2.0, 1.0) + 0.3 * entropia + 0.3 * (1.0 / (1.0 + deriva)), 0.0, 1.0))
        if estabilidad > 0.75:
            diagnostico = "OPTIMA (Convergencia Armonica)"
        elif estabilidad > 0.45:
            diagnostico = "ESTABLE (Plasticidad Adaptativa)"
        else:
            diagnostico = "ALERTA (Dispersión o Saturación)"
        return estabilidad, diagnostico


# ===========================================================================
# VISUALIZADOR DE PESOS VIVOS EN TERMINAL
# ===========================================================================
_gestor_global: Optional[GestorPesosVivos] = None


def get_gestor_pesos_vivos() -> GestorPesosVivos:
    """Retorna la instancia global del gestor de pesos vivos."""
    global _gestor_global
    if _gestor_global is None:
        _gestor_global = GestorPesosVivos()
    return _gestor_global


def inyectar_pesos_turno(cerebro: Any, conversor: Any, pregunta: str, respuesta: str,
                         info_pesos: Dict[str, Any], mostrar_en_terminal: bool = True) -> int:
    """
    Funcion compatible de inyeccion en vivo por turno.
    Inyecta el estado de modulos anatomicos y la respuesta global a las neuronas,
    desplegando las metricas de pesos vivos en terminal.
    """
    gestor = get_gestor_pesos_vivos()
    res = gestor.inyectar_turno_en_vivo(
        conversor=conversor,
        pregunta=pregunta,
        respuesta=respuesta,
        info_pesos=info_pesos,
        mostrar_en_terminal=mostrar_en_terminal,
    )
    return res.get("turno", 1)


def checkpoint_y_registrar(conversor: Any, sesion: Optional[Any] = None,
                          etiqueta: str = "pesos_activos") -> List[Path]:
    """Guarda checkpoint en PSNRL y lo registra para sincronizacion en IPFS."""
    gestor = get_gestor_pesos_vivos()
    return gestor.checkpoint_y_registrar(conversor=conversor, sesion=sesion, etiqueta=etiqueta)


# ===========================================================================
# EJECUCION Y PRUEBA EN TERMINAL
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"\n{ANSI.BOLD}{ANSI.CYAN}--- INICIANDO TEST EN VIVO DE PESOS VIVOS 2026 ---{ANSI.RESET}\n")

    from LC.celebro.CMFG.PSNRCV import get_conversor_pesos
    conv = get_conversor_pesos()

    turnos_demo = [
        ("Hola LucIA, ¿cuál es el estado actual de tus conexiones sinápticas?",
         "Mis 50 neuronas en ENRN, RF_SL, RF_EN, RNP y SLRN están calibradas y sincronizadas."),
        ("Explícame cómo funciona la optimización de pesos 2026 en tu celebro.",
         "Utilizo ortogonalización Muon Newton-Schulz 5 con preacondicionador de curvatura SOAP."),
        ("Consolida el aprendizaje y prepara el almacenamiento en PSNRL.",
         "Pesos activos estabilizados y registrados en checkpoints de alta velocidad en PSNRL.")
    ]

    for q, a in turnos_demo:
        info_sim = conv.procesar_consulta_a_pesos(q)
        inyectar_pesos_turno(cerebro=None, conversor=conv, pregunta=q, respuesta=a,
                             info_pesos=info_sim, mostrar_en_terminal=True)
        time.sleep(0.2)

    archivos = checkpoint_y_registrar(conv, sesion=None, etiqueta="pesos_vivos_test")
    print(f"\n{ANSI.GREEN}Test finalizado con éxito. Archivos en PSNRL: {len(archivos)}{ANSI.RESET}\n")
from LC.celebro.CMFG.pesos_vivos_VisualizadorTerminalPesos import VisualizadorTerminalPesos  # CLASSPACK
from LC.celebro.CMFG.pesos_vivos_GestorPesosVivos import GestorPesosVivos  # CLASSPACK
