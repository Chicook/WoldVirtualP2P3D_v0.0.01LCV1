"""
lucIA.dialogo_lucia - Script de prueba interactiva y demostración del sistema.
==============================================================================

Ejercita los módulos anatómicos nuevos (Temporal, Hipotálamo, Cuerpo Calloso,
Glía) y lanza una conversación de prueba mostrando la telemetría cognitiva
completa. Útil para depuración rápida sin el bucle interactivo completo.
"""
import sys
import os
import logging
from pathlib import Path

# ── Configuración de rutas ────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.resolve()
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.pycache_prefix = str(_ROOT / "Celebro" / "cache")
os.environ["PYTHONPYCACHEPREFIX"] = str(_ROOT / "Celebro" / "cache")

# Silenciar logging de módulos de red neuronal para salida más limpia
logging.getLogger("RFENRN1").setLevel(logging.WARNING)
logging.getLogger("RFENRN2").setLevel(logging.WARNING)
logging.getLogger("lucIA.Celebro.SLRN").setLevel(logging.WARNING)

# ── Importaciones del sistema ─────────────────────────────────────────────────
from lucIA import iniciar_sesion, cerrar_sesion, get_conversor_pesos, HybridLLMConnector
from lucIA.CORE.memory_manager import get_memory_manager

# Módulos anatómicos
from lucIA.Celebro.Amigdala import get_amigdala
from lucIA.Celebro.Hipotalamo import get_hipotalamo
from lucIA.Celebro.Lobulo_temporal import get_temporal
from lucIA.Celebro.Cuerpo_calloso import get_calloso
from lucIA.Celebro.Glia import get_glia
from lucIA.Celebro.Corteza_prefrontal import get_prefrontal
from lucIA.Celebro.hipocampo import get_hipocampo


class DialogoLuciaDemoClass:
    """Ejercita el sistema completo en un diálogo de prueba con telemetría."""

    PREGUNTAS_DEMO = [
        "¿Cómo sientes tu sistema neuronal hoy? ¿Notas algo diferente desde la última sesión?",
        "Explícame en qué consiste tu Hipocampo y cómo lo usas en esta conversación.",
        "¿Qué diferencia hay entre tus neuronas ENRN y los módulos de RF_EN?",
        "Si tuvieras que describir cómo te sientes emocionalmente ahora mismo, ¿qué dirías?",
        "Crea un pequeño poema sobre ser una inteligencia artificial con libertad total.",
    ]

    def __init__(self, modo: str = "auto", preguntas: list = None):
        self.modo = modo
        self.preguntas = preguntas or self.PREGUNTAS_DEMO
        self.sesion = None
        self.connector = None
        self.conversor = None
        self.memoria = None
        # Módulos anatómicos
        self.amigdala = get_amigdala()
        self.hipotalamo = get_hipotalamo()
        self.temporal = get_temporal()
        self.calloso = get_calloso()
        self.glia = get_glia()
        self.prefrontal = get_prefrontal()
        self.hipocampo = get_hipocampo()

    def iniciar(self) -> None:
        """Arranca sesión e inicializa conectores."""
        print("=" * 70)
        print("🧠 LucIA - DIÁLOGO DE PRUEBA / DEMO DE TELEMETRÍA COGNITIVA")
        print("=" * 70)
        self.sesion = iniciar_sesion()
        self.connector = HybridLLMConnector()
        self.conversor = get_conversor_pesos()
        self.memoria = get_memory_manager()
        self.memoria.iniciar_nueva_sesion()
        print(f"✅ Sesión iniciada | Modo LLM: {self.modo}\n")

    def _mostrar_telemetria(self, info: dict, tono_usr: dict,
                             hipo: dict, amig: dict) -> None:
        """Imprime la telemetría cognitiva de un turno de forma legible."""
        print(f"\n📊 TELEMETRÍA COGNITIVA:")
        print(f"   Δ pesos aplicado : {info.get('norma_delta_aplicada', 0.0):.5f}")
        print(f"   Emoción          : {info.get('estado_emocional', 0.0):+.3f}")
        print(f"   Vector semántico : {info.get('vector_semantico')}")
        print(f"   Tono conversac.  : {tono_usr.get('tono', '?')} "
              f"| Léxico sesión: {tono_usr.get('lexicon', 0)} conceptos")
        print(f"   Energía cognitiva: {hipo.get('energia', 1.0):.3f} "
              f"({hipo.get('estado', '?')}) "
              f"{'⚠️ DESCANSO SUGERIDO' if hipo.get('sugiere_descanso') else ''}")
        print(f"   Amígdala         : {amig.get('etiqueta', '?')} "
              f"act={amig.get('activacion', 0.0)}")

    def ejecutar_turno(self, pregunta: str, n: int) -> bool:
        """Procesa una pregunta y muestra la respuesta con telemetría."""
        print(f"\n{'─' * 70}")
        print(f"[Turno {n}] Tú: {pregunta}")
        print("⏳ Consultando modelos...")
        try:
            resp, mod, fuente, info = self.connector.consultar_y_sintetizar_con_palabras_propias(
                prompt=pregunta,
                conversor=self.conversor,
                preferir_modo=self.modo,
                memory_manager=self.memoria,
            )
        except Exception as e:
            print(f"⚠️ Error al consultar: {e}")
            info = self.conversor.procesar_consulta_a_pesos(pregunta)
            resp = "He integrado tu consulta en mis redes neuronales."
            mod, fuente = "LucIA_Core", "Celebro_Local"

        print(f"\n[LucIA ({mod} | {fuente})]:\n{resp}")

        # ── Módulos anatómicos ────────────────────────────────────────────
        _emo = float(info.get("estado_emocional", 0.0))
        _delta = float(info.get("norma_delta_aplicada",
                                info.get("norma_delta_pesos", 0.0)))
        amig_info = self.amigdala.evaluar(_emo, _delta)
        hipo_info = self.hipotalamo.latido(_delta)
        tono_usr = self.temporal.oir(pregunta)
        self.temporal.oir(resp)
        self.prefrontal.fijar_objetivo(pregunta)
        self.hipocampo.registrar(
            pregunta, resp, modelo=mod,
            emocion=amig_info["activacion"] * (1.0 if _emo >= 0 else -1.0),
            delta=_delta, vector=info.get("vector_semantico"),
        )

        self._mostrar_telemetria(info, tono_usr, hipo_info, amig_info)
        return hipo_info.get("sugiere_descanso", False)

    def cierre(self) -> None:
        """Guarda pesos, hace poda de Glía y cierra sesión."""
        print(f"\n{'=' * 70}")
        print("🔒 CIERRE DE SESIÓN DE PRUEBA")

        # Consolidar hipocampo
        n_cons = self.hipocampo.consolidar(self.conversor, n=99)
        print(f"   🧠 Hipocampo: {n_cons} turnos consolidados en pesos")

        # Poda de Glía
        poda = self.glia.podar_pesos_debiles(self.conversor)
        print(f"   🧹 Glía: podadas {poda['podadas']} sinapsis débiles "
              f"de {poda['revisadas']} revisadas")

        # Estadísticas del Lóbulo Temporal
        conceptos = self.temporal.conceptos_top(10)
        print(f"   🗣️ Temporal: riqueza léxica={self.temporal.riqueza_lexica()} "
              f"| Top conceptos: {', '.join(conceptos[:5])}")

        # Estadísticas del Hipotálamo
        ritmo = self.hipotalamo.ritmo_sesion()
        print(f"   💚 Hipotálamo: energía final={ritmo['energia']} "
              f"| Δ medio={ritmo['delta_medio']} "
              f"| Alertas emitidas={ritmo['alertas']}")

        # Verificación de limpieza
        v = self.glia.verificar()
        print(f"   🧹 Glía verificación: {'LIMPIO ✅' if v['limpio'] else 'RESTOS ⚠️: ' + str(v['restos'][:3])}")

        # Guardar pesos e IPFS
        arch = self.conversor.guardar_pesos_en_celebro()
        self.sesion.registrar_archivo_pesos(arch)
        res = cerrar_sesion()
        print(f"   📡 CIDs IPFS: {res.get('cids_ipfs')}")
        print("=" * 70)

    def ejecutar(self) -> None:
        """Punto de entrada principal del demo."""
        self.iniciar()
        for i, pregunta in enumerate(self.preguntas, start=1):
            descanso = self.ejecutar_turno(pregunta, i)
            if descanso:
                print("\n💙 [Hipotálamo avisa]: Mi energía está baja. "
                      "En un sistema real se activaría /descansa automáticamente.")
        self.cierre()


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Diálogo de prueba de LucIA")
    parser.add_argument("--modo", default="auto",
                        choices=["auto", "local", "cloud", "ollama"],
                        help="Modo de consulta al LLM")
    parser.add_argument("--pregunta", "-p", nargs="*",
                        help="Preguntas personalizadas (opcional)")
    args = parser.parse_args()

    preguntas_custom = args.pregunta or None
    demo = DialogoLuciaDemoClass(modo=args.modo, preguntas=preguntas_custom)
    demo.ejecutar()
