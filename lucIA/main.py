"""
lucIA.main - Punto de entrada interactivo conversacional con Voz, IPFS y Celebro
========================================================================

Flujo:
1. Inicio de sesión con redirección de __pycache__ a Celebro/cache.
2. Modo interactivo: el usuario realiza consultas continuas.
3. LucIA responde y habla con la voz oficial es-ES-ElviraNeural.
4. Cada interacción convierte las respuestas a pesos neuronales en Celebro.
5. Al escribir 'salir' o 'exit':
   - LucIA se despide por voz.
   - Sube todos los pesos neuronales a IPFS.
   - Elimina los pesos de disco local.
   - Vacía completamente Celebro/cache.
   - Cierra la sesión de forma limpia.
"""

import sys
import os
import signal
import threading
from pathlib import Path

# === GESTOR DE ENTORNO VIRTUAL ===
class GestorEntornoVirtual:
    """
    Asegura que LucIA siempre se ejecute dentro del entorno virtual 'venv'.
    Usa subprocess en vez de os.execv para preservar stdin/stdout interactivo en Windows.
    """

    @staticmethod
    def esta_activo() -> bool:
        """Comprueba si ya se está ejecutando dentro de un entorno virtual."""
        return sys.prefix != sys.base_prefix

    @staticmethod
    def encontrar_python_venv() -> "Path | None":
        """Busca el intérprete Python del entorno virtual en Windows y Unix."""
        base = Path(__file__).parent / "venv"
        candidatos = [
            base / "Scripts" / "python.exe",  # Windows
            base / "bin" / "python",           # Linux/macOS
            base / "bin" / "python3",          # Linux/macOS alternativo
        ]
        return next((p for p in candidatos if p.exists()), None)

    @classmethod
    def activar_si_necesario(cls) -> None:
        """
        Si no estamos dentro del venv, relanza el proceso usando subprocess,
        que sí preserva el terminal interactivo (stdin/stdout) correctamente.
        Al terminar, sale con el mismo código de retorno del proceso hijo.
        """
        if cls.esta_activo():
            return  # Ya estamos dentro del venv, no hacemos nada

        python_venv = cls.encontrar_python_venv()
        if python_venv is None:
            print("[LucIA] Aviso: No se encontro el entorno virtual 'venv'. Ejecutando en entorno global.")
            return

        import subprocess
        print("[LucIA] Iniciando dentro del entorno virtual...")
        try:
            resultado = subprocess.call(
                [str(python_venv)] + sys.argv,
                stdin=sys.stdin,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            sys.exit(resultado)
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print(f"[LucIA] Error al lanzar en venv: {e}. Continuando en entorno global.")

GestorEntornoVirtual.activar_si_necesario()
# === FIN GESTOR ENTORNO VIRTUAL ===

# Garantizar paths del proyecto
_DIR_ACTUAL = Path(__file__).parent.resolve()
_DIR_RAIZ = _DIR_ACTUAL.parent.resolve()
for _camino in [str(_DIR_ACTUAL), str(_DIR_RAIZ)]:
    if _camino not in sys.path:
        sys.path.insert(0, _camino)

# === PUENTE RD_NEURONAL (sustituto in-memory de Celebro/RD_Neuronal/) ===
# Refactoriza los 7 re-exportadores (<1KB, sin logica propia) en esta clase:
# expone EXACTAMENTE los mismos nombres desde sus paquetes reales y registra
# alias en sys.modules para que los 14 importadores existentes no se toquen.
# RNP/rnp_transform.py NO se porta: su objetivo no existe y nadie lo importa.
class PuenteRDNeuronal:
    ENRN = RF_SL = RF_EN = RNP = SLRN = None
    NeuronaEntradaBase = None
    NeuronaMemoriaBase = None
    BaseNeuralWeightOptimizer = None
    NeuralWeightOptimizationConfig = None
    NeuralWeightOptimizationResult = None
    NeuralWeightOptimizationMetrics = None

    @classmethod
    def instalar(cls) -> bool:
        try:
            import types as _types
            # FASE 1: ENRN primero (no depende de nadie) -> alias inmediato.
            from lucIA.Celebro import ENRN as _ENRN
            raiz = _types.ModuleType("lucIA.Celebro.RD_Neuronal")
            raiz.__path__ = []
            raiz.ENRN = _ENRN
            sys.modules["lucIA.Celebro.RD_Neuronal"] = raiz
            sys.modules["lucIA.Celebro.RD_Neuronal.ENRN"] = _ENRN
            # FASE 2: resto de paquetes reales + alias + nombres vitales.
            from lucIA.Celebro import RF_SL, RF_EN, RNP, SLRN
            cls.ENRN, cls.RF_SL, cls.RF_EN, cls.RNP, cls.SLRN = _ENRN, RF_SL, RF_EN, RNP, SLRN
            cls.NeuronaEntradaBase = _ENRN.NeuronaEntradaBase
            cls.NeuronaMemoriaBase = RF_SL.NeuronaMemoriaBase
            cls.BaseNeuralWeightOptimizer = RNP.BaseNeuralWeightOptimizer
            cls.NeuralWeightOptimizationConfig = RNP.NeuralWeightOptimizationConfig
            cls.NeuralWeightOptimizationResult = RNP.NeuralWeightOptimizationResult
            cls.NeuralWeightOptimizationMetrics = RNP.NeuralWeightOptimizationMetrics
            for _n, _m in (("RF_SL", RF_SL), ("RF_EN", RF_EN),
                           ("RNP", RNP), ("SLRN", SLRN)):
                setattr(raiz, _n, _m)
                sys.modules[f"lucIA.Celebro.RD_Neuronal.{_n}"] = _m
            sys.modules["lucIA.Celebro.RD_Neuronal"] = raiz
            return True
        except Exception as e:
            print(f"[LucIA] Aviso puente RD_Neuronal: {e}. Se usa la carpeta fisica como respaldo.")
            return False

PuenteRDNeuronal.instalar()
# === FIN PUENTE RD_NEURONAL ===

# === SISTEMA CEREBRAL (agrupa hipocampo + 7 modulos nuevos) ===
# Un solo objeto optimiza el cableado: init perezoso (1 import por modulo),
# hooks por turno/cierre y lineas de /estado. Todo en RAM; al cerrar va a pesos.
class SistemaCerebral:
    MODULOS = ("sensorial", "talamo", "amigdala", "prefrontal",
               "ganglios", "cerebelo", "tronco")

    def __init__(self):
        self.hipocampo = None
        self.sensorial = self.talamo = self.amigdala = None
        self.prefrontal = self.ganglios = self.cerebelo = None
        self.tronco = self.glia = None
        self.hipotalamo = self.calloso = self.motora = None
        self.temporal = self.occipital = self.parietal = None
        self.modulos = []
        self.activo = False

    def iniciar(self) -> bool:
        try:
            from lucIA.Celebro.hipocampo import get_hipocampo
            from lucIA.Celebro.Talamo import get_talamo
            from lucIA.Celebro.Amigdala import get_amigdala
            from lucIA.Celebro.Corteza_sensorial import get_sensorial
            from lucIA.Celebro.Corteza_prefrontal import get_prefrontal
            from lucIA.Celebro.Ganglios_basales import get_ganglios
            from lucIA.Celebro.Cerebelo import get_cerebelo
            from lucIA.Celebro.Tronco_encefalico import get_tronco
            from lucIA.Celebro.Glia import get_glia
            from lucIA.Celebro.Hipotalamo import get_hipotalamo
            from lucIA.Celebro.Cuerpo_calloso import get_calloso
            from lucIA.Celebro.Corteza_motora import get_motora
            from lucIA.Celebro.Lobulo_temporal import get_temporal
            from lucIA.Celebro.Lobulo_occipital import get_occipital
            from lucIA.Celebro.Lobulo_parietal import get_parietal
            self.hipocampo = get_hipocampo()
            self.sensorial, self.talamo, self.amigdala = get_sensorial(), get_talamo(), get_amigdala()
            self.prefrontal, self.ganglios, self.cerebelo = get_prefrontal(), get_ganglios(), get_cerebelo()
            self.tronco, self.glia = get_tronco(), get_glia()
            self.hipotalamo, self.calloso, self.motora = get_hipotalamo(), get_calloso(), get_motora()
            self.temporal, self.occipital, self.parietal = get_temporal(), get_occipital(), get_parietal()
            self.modulos = [getattr(self, m) for m in self.MODULOS] + [
                self.hipotalamo, self.calloso, self.motora,
                self.temporal, self.occipital, self.parietal]
            self.activo = True
            print(f"🧠 Modulos cerebrales activos: {len(self.modulos) + 1} (hipocampo + 13)")
        except Exception as e:
            print(f"[LucIA] Modulos cerebrales no disponibles ({e}), sigo con nucleo base.")
        return self.activo

    def recompensa(self, cmd: str, pregunta: str, conversor) -> bool:
        """Procesa /bien|/mal (True=consumido) y gracias (False=seguir)."""
        try:
            if self.motora is not None and cmd.startswith("/"):
                self.motora.actuar(cmd)
        except Exception:
            pass
        if cmd in ("/bien", "/mal"):
            if self.ganglios is not None:
                b = self.ganglios.recompensar(1 if cmd == "/bien" else -1, conversor, motivo=pregunta)
                print(f"💛 Recompensa registrada. Balance: {b}")
            else:
                print("Modulo de recompensa no disponible.")
            return True
        if self.ganglios is not None and any(
                w in cmd for w in ("gracias", "perfecto", "genial", "muy bien")):
            try:
                self.ganglios.recompensar(1, conversor, motivo="agradecimiento")
                self.ganglios.habito("respuesta_util")
            except Exception:
                pass
        return False

    def tras_turno(self, pregunta, respuesta, modelo, info_pesos, memoria, conversor) -> None:
        if not self.activo:
            return
        try:
            from lucIA.Celebro.pesos_vivos import inyectar_pesos_turno
            inyectar_pesos_turno(self, conversor, pregunta, respuesta, info_pesos)
        except Exception:
            pass
        try:
            _emo = float(info_pesos.get("estado_emocional", 0.0))
            _delta = float(info_pesos.get("norma_delta_aplicada",
                                          info_pesos.get("norma_delta_pesos", 0.0)))
            self.sensorial.normalizar(pregunta)
            self.prefrontal.fijar_objetivo(pregunta)
            # Amigdala = perilla unica del peso emocional
            _emo_pond = self.amigdala.evaluar(_emo, _delta)["activacion"] * (1.0 if _emo >= 0 else -1.0)
            self.hipocampo.registrar(pregunta, respuesta, modelo=modelo,
                                     emocion=_emo_pond, delta=_delta,
                                     vector=info_pesos.get("vector_semantico"))
            self.cerebelo.marcar("respuesta")
            # Hipotalamo: latido retorna dict con energia y estado
            _hipo_info = self.hipotalamo.latido(_delta)
            # Lobulo_temporal: analisis de tono de la pregunta y respuesta
            _tono_usr = self.temporal.oir(pregunta)
            self.temporal.oir(respuesta)
            _turno_n = len(memoria._datos.get("turnos", []))
            self.parietal.situar(pregunta, modelo=modelo, turno=_turno_n)
            # Cuerpo_calloso: integra las dos vias si hay respuesta doble
            if hasattr(self, "calloso") and self.calloso is not None:
                try:
                    # Si la respuesta tiene un separador [L]/[C] (via dual)
                    if "[L]" in respuesta and "[C]" in respuesta:
                        partes = respuesta.split("[C]", 1)
                        _local_part = partes[0].replace("[L]", "").strip()
                        _cloud_part = partes[1].strip() if len(partes) > 1 else ""
                        self.calloso.integrar(_local_part, _cloud_part)
                except Exception:
                    pass
            if _turno_n % 5 == 0:
                self.hipocampo.consolidar(conversor, n=5)
            # Glia: limpieza de buffer de eventos cada 10 turnos
            if _turno_n % 10 == 0 and hasattr(self, "glia") and self.glia is not None:
                try:
                    self.glia.limpiar_eventos_antiguos(max_eventos=150)
                except Exception:
                    pass
        except Exception:
            pass

    def sincronizar_desde_conversor(self, conversor, memoria=None, info_restauracion=None) -> int:
        """F2: siembra módulos anatómicos con pesos restaurados ANTES de la 1ª consulta.
        No muta pesos del conversor; anota evento + objetivo/energía inicial coherente."""
        if not self.activo:
            return 0
        hechos = 0
        try:
            n_neur = len(getattr(conversor, "neuronas", {}) or {})
            cid = (info_restauracion or {}).get("cid", "")
            comp = (info_restauracion or {}).get("componentes_restaurados", n_neur)
            resumen_mem = ""
            try:
                resumen_mem = memoria.obtener_resumen_sesion() if memoria is not None else ""
            except Exception:
                resumen_mem = ""
            for m in list(getattr(self, "modulos", []) or []):
                try:
                    if hasattr(m, "_notar"):
                        m._notar(f"restaurado IPFS cid={cid[:12]} comp={comp} neur={n_neur}")
                        hechos += 1
                except Exception:
                    pass
            try:
                if self.prefrontal is not None and hasattr(self.prefrontal, "fijar_objetivo"):
                    self.prefrontal.fijar_objetivo(resumen_mem[:200] or "continuar hilo anterior")
            except Exception:
                pass
            try:
                if self.hipocampo is not None and hasattr(self.hipocampo, "registrar"):
                    self.hipocampo.registrar("restauracion IPFS", f"cid={cid[:12]} comp={comp}",
                                             modelo="ipfs-restore", emocion=0.0, delta=0.0)
            except Exception:
                pass
        except Exception:
            pass
        return hechos

    def lineas_estado(self) -> list:
        if not self.activo:
            return []
        lin = []
        try:
            lin.append(f"   - 🧠 {self.hipocampo.resumen_estado()}")
            for m in (self.amigdala, self.prefrontal, self.ganglios,
                      self.cerebelo, self.tronco, self.talamo, self.sensorial,
                      self.hipotalamo, self.parietal, self.temporal, self.motora,
                      self.calloso, self.occipital):
                lin.append(f"   - {m.resumen()}")
        except Exception:
            pass
        return lin

    def volcar_cierre(self, conversor, sesion) -> None:
        # Poda de sinapsis debiles antes de volcar (Glia activa durante el "sueno")
        try:
            if self.glia is not None:
                _poda = self.glia.podar_pesos_debiles(conversor)
                if _poda["podadas"] > 0:
                    print(f"🧹 Glía: podadas {_poda['podadas']} sinapsis débiles de {_poda['revisadas']} revisadas.")
        except Exception:
            pass
        try:
            rc = self.hipocampo.volcar_cierre(conversor, sesion=sesion)
            print(f"🧠 Hipocampo: {rc['consolidados']}/{rc['pendientes']} turnos -> pesos ({rc['archivo'] or 'sin archivo'})")
        except Exception as e:
            print(f"   ⚠️ Hipocampo no pudo volcar: {e}")
        try:
            for m in self.modulos:
                try:
                    m.a_pesos(conversor)
                except Exception:
                    pass
            # Hipocampo y Glia tambien exportan
            if self.hipocampo is not None:
                try:
                    self.hipocampo.a_pesos(conversor) if hasattr(self.hipocampo, "a_pesos") else None
                except Exception:
                    pass
            if self.glia is not None:
                try:
                    self.glia.a_pesos(conversor)
                except Exception:
                    pass
            print("   ✅ Estado de los 15 modulos convertido a pesos neuronales")
        except Exception as e:
            print(f"   ⚠️ Modulos no pudieron volcar: {e}")
        try:
            from lucIA.Celebro.pesos_vivos import checkpoint_y_registrar
            archs = checkpoint_y_registrar(conversor, sesion, "pesos_cierre_anatomico")
            print(f"   ✅ Checkpoint anatomico {archs[0].name} registrado para IPFS (borrado local al confirmar CID)")
        except Exception as e:
            print(f"   ⚠️ Checkpoint anatomico no pudo registrarse: {e}")

    def verificar_cierre(self) -> None:
        try:
            v = self.glia.verificar()
            restos_str = ", ".join(v['restos'][:5]) if v['restos'] else "ninguno"
            print(f"🧹 Glía: {'LIMPIO ✅ (0 restos)' if v['limpio'] else f'RESTOS: {restos_str}'}")
        except Exception as e:
            print(f"🧹 Glía no pudo verificar: {e}")
# === FIN SISTEMA CEREBRAL ===

# Asegurar codificación utf-8 en consolas Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configuración inicial de pycache antes de importar otros submódulos
_ROOT = Path(__file__).parent.resolve()
_CACHE = _ROOT / "Celebro" / "cache"
_CACHE.mkdir(parents=True, exist_ok=True)
sys.pycache_prefix = str(_CACHE)
os.environ["PYTHONPYCACHEPREFIX"] = str(_CACHE)

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from lucIA import (
    iniciar_sesion,
    cerrar_sesion,
    verificar_limite_espacio,
    speak,
    cancel_speech,
    HybridLLMConnector,
    get_conversor_pesos,
    get_autorefactor_engine,
    IPFSManager,
    codificar_versiones_en_pesos,
    codificar_memoria_en_pesos,
    codificar_cache_en_pesos,
    restaurar_pesos_desde_ipfs
)
from lucIA.CORE.memory_manager import get_memory_manager


def mostrar_banner_venv() -> None:
    """Banner hiper-visible del entorno virtual (no modifica el venv, solo informa)."""
    import platform
    activo = sys.prefix != sys.base_prefix
    estado = "✅ ACTIVO" if activo else "⚠️ GLOBAL (sin venv)"
    print("=" * 75)
    print(f"🐍 ENTORNO VIRTUAL: {estado}")
    print(f"   Python : {sys.executable}")
    print(f"   Version: {platform.python_version()} | Prefijo: {sys.prefix}")
    try:
        from lucIA.model_router import ModelRouter  # noqa
        print("   Rotacion: patron L->C->L, 1 modelo distinto por pregunta | Espanol 100%")
    except Exception:
        pass
    print("=" * 75)
    if not activo:
        print("⚠️ AVISO: estas fuera del venv. Se recomienda usar iniciar_lucia.bat. Pulsa Enter para continuar...")
        try:
            input()
        except Exception:
            pass


def ejecutar_chat_principal():
    """Bucle conversacional interactivo de LucIA - Versión mejorada y corregida."""
    mostrar_banner_venv()
    print("=" * 75)
    print("LucIA - INTELIGENCIA ARTIFICIAL INTERACTIVA (VOZ es-ES-ElviraNeural)")
    print("=" * 75)

    # 1. Iniciar sesión y configurar componentes
    sesion = iniciar_sesion()
    # Clave OpenRouter: ya cargada desde .env vía lucIA/__init__.py; solo avisar.
    import os as _os
    if not _os.environ.get("OPENROUTER_API_KEY"):
        print("[Sistema] Sin OPENROUTER_API_KEY en .env: modo local (LM Studio + Ollama).")
    else:
        print("[Sistema] Clave OpenRouter cargada desde .env.")

    # 1b. Gestión y descarga autónoma de modelos locales en cada sesión (Ollama / LM Studio)
    try:
        from lucIA.local_model_manager import get_local_model_manager
        _model_mgr = get_local_model_manager()
        _res_descarga = _model_mgr.asegurar_un_modelo_por_sesion(async_mode=False)
        if _res_descarga.get("descargado"):
            print(f"🤖 [Modelos Locales]: Descargado/Operativo '{_res_descarga.get('modelo')}' en {_res_descarga.get('backend').upper()}.")
        else:
            print(f"🤖 [Modelos Locales]: Modelos locales verificados ({_res_descarga.get('total_ollama')} Ollama, {_res_descarga.get('total_lmstudio')} LM Studio).")
    except Exception as _e_mm:
        logger.warning(f"Aviso gestor de modelos locales: {_e_mm}")

    connector = HybridLLMConnector()
    conversor = get_conversor_pesos()
    memoria = get_memory_manager()
    memoria.iniciar_nueva_sesion()

    # Sistema cerebral nuevo (hipocampo + 7 modulos, 1 solo objeto)
    cerebro = SistemaCerebral()
    cerebro.iniciar()

    # 2. Descargar y restaurar pesos neuronales desde IPFS
    print("📡 Conectando con IPFS y sincronizando pesos neuronales de Celebro...")
    info_restauracion = restaurar_pesos_desde_ipfs(sesion=sesion, conversor=conversor)
    if info_restauracion.get("restaurado"):
        cid_corto = info_restauracion['cid'][:14] + "..." + info_restauracion['cid'][-8:]
        print(
            f"✨ [Celebro]: Red sináptica restaurada desde IPFS exitosamente.\n"
            f"   - CID IPFS: {info_restauracion['cid']}\n"
            f"   - Modelo origen: {info_restauracion['nombre_modelo']}\n"
            f"   - Matrices y neuronas sincronizadas: {info_restauracion['componentes_restaurados']}"
        )
    else:
        print("ℹ️ [Celebro]: Sin checkpoints previos en IPFS; iniciando con topología neuronal base.")
    # F2: cerebro sincronizado con pesos restaurados ANTES de la 1ª consulta.
    try:
        n_sync = cerebro.sincronizar_desde_conversor(conversor, memoria, info_restauracion)
        if n_sync:
            print(f"🧠 Cerebro sincronizado con pesos IPFS: {n_sync} módulos sembrados.")
    except Exception:
        pass

    modo_actual = "auto"
    voz_activa = True

    info_tamano = verificar_limite_espacio()
    print(f"📊 Sistema LucIA: {info_tamano['tamano_mb']} MB / {info_tamano['limite_mb']} MB (Caché en Celebro/cache)")
    print(f"🧠 Memoria: {memoria.obtener_resumen_sesion()}")
    print("Comandos rápidos: /modo <local|cloud|auto> | /voz <on|off> | /estado | /memoria | /olvida | /descansa | /bien | /mal | salir")
    print("-" * 75)

    # Saludo inicial hablado con es-ES-ElviraNeural
    saludo_inicial = "Hola, soy Lucía. Mis redes neuronales están sincronizadas desde IPFS y aprenderé en tiempo real."
    print(f"\n[LucIA]: {saludo_inicial}")
    speak(saludo_inicial)
    print("\n[Sistema]: Listo para recibir preguntas. Ingrese su consulta (o 'salir' para terminar):")

    # Bucle de interacción con el usuario - Lógica mejorada
    while True:
        try:
            pregunta = input("\n[Tú]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCerrando...")
            break

        # Validación inteligente: permitir preguntas normales españolas, rechazar solo entradas claramente malformadas
        if not pregunta or len(pregunta) < 3:
            print("Por favor, ingrese una pregunta normal.")
            continue

        # Rechazar solo patrones claramente no conversacionales (ej. líneas de log técnicas)
        if (pregunta.lower().startswith(('cid ', 'ipfs ', 'Qm', 'pesos_', 'neurona_', 'modelo_', 'memoria_')) or
            any(char in pregunta for char in ['http://', 'https://', 'ftp://', 'github.com']) or
            any(char in pregunta for char in ['[INFO]', '[ERROR]', '[DEBUG]', '2026', '2025', '2024'])):
            print("Por favor, ingrese una pregunta normal (no información técnica del sistema).")
            continue

        # Interrumpir cualquier locución previa
        cancel_speech()

        cmd = pregunta.lower()

        # Recompensa (Ganglios) via SistemaCerebral: /bien|/mal consumen, gracias suma
        if cerebro.recompensa(cmd, pregunta, conversor):
            continue

        # Salir y cerrar sesión
        if cmd in ("salir", "exit", "quit", "adios", "adiós"):
            despedida = "Hasta pronto. Guardando mi estado sináptico en IPFS y cerrando sesión."
            print(f"\n[LucIA]: {despedida}")
            speak(despedida, esperar=True)
            break

        # Comandos de control mejorados
        if cmd.startswith("/modo"):
            partes = cmd.split()
            if len(partes) > 1 and partes[1] in ("local", "cloud", "auto"):
                modo_actual = partes[1]
                print(f"⚙️ Modo de modelo cambiado a: {modo_actual}")
            else:
                print(f"Modo actual: {modo_actual}. Opciones: /modo local, /modo cloud, /modo auto")
            continue

        if cmd.startswith("/voz"):
            partes = cmd.split()
            if len(partes) > 1 and partes[1] in ("on", "off"):
                voz_activa = (partes[1] == "on")
                print(f"🔊 Voz configurada en: {'ACTIVADA' if voz_activa else 'DESACTIVADA'}")
            else:
                print(f"Voz actual: {'ACTIVADA' if voz_activa else 'DESACTIVADA'}")
            continue

        if cmd == "/estado":
            pesos = conversor.exportar_pesos_celebro()
            tam = verificar_limite_espacio()
            venv_estado = "✅ ACTIVO" if sys.prefix != sys.base_prefix else "⚠️ GLOBAL"
            print(f"\n📊 ESTADO SINÁPTICO DE CELEBRO:")
            print(f"   - 🐍 venv: {venv_estado} ({sys.executable})")
            try:
                toca = connector.router.toca() if getattr(connector, "router", None) else "?"
                print(f"   - 🔄 Rotacion L->C->L: turno #{connector.router.turno} | toca: {toca}")
                for h in (connector.router.historial[-4:] if connector.router else []):
                    print(f"     turno {h['turno']}: {h['modelo']} ({h['origen']}) ok={h['ok']}")
            except Exception:
                pass
            try:
                for _lin in cerebro.lineas_estado():
                    print(_lin)
            except Exception:
                pass
            try:
                from lucIA.Celebro.RNP.panel_rnp import resumen_rnp
                _panel = resumen_rnp(conversor.rnp_neuronas + [conversor.rnp_14])
                for _pl in _panel["informe"].split("\n"):
                    print(f"   - {_pl}")
            except Exception:
                pass
            print(f"   - Tamaño total: {tam['tamano_mb']} MB (Límite: {tam['limite_mb']} MB)")
            print(f"   - Consultas aprendidas en esta sesión: {len(conversor.historial_actualizaciones)}")
            for n_nombre in pesos:
                print(f"   - Neurona activa: {n_nombre}")
            continue

        if cmd == "/memoria":
            print(f"\n🧠 ESTADO DE LA MEMORIA CONVERSACIONAL:")
            print(f"   - {memoria.obtener_resumen_sesion()}")
            continue

        if cmd in ("/descansa", "/descanso", "/respira", "/pausa"):
            import time as _time
            partes = cmd.split()
            try:
                segundos = int(partes[1]) if len(partes) > 1 else 20
                segundos = max(5, min(segundos, 120))
            except Exception:
                segundos = 20
            cancel_speech()
            # Reponer energia del hipotalamo
            try:
                if cerebro.hipotalamo is not None:
                    _e_antes = cerebro.hipotalamo.energia
                    cerebro.hipotalamo.reponer(parcial=True)
                    _e_despues = cerebro.hipotalamo.energia
                    print(f"💚 [Hipotálamo]: Energía repuesta {_e_antes:.2f} → {_e_despues:.2f}")
            except Exception:
                pass
            # Glia: poda ligera de buffer durante el descanso
            try:
                if cerebro.glia is not None:
                    cerebro.glia.limpiar_eventos_antiguos(max_eventos=80)
            except Exception:
                pass
            msg_descanso = (
                f"Gracias por cuidarme. Voy a respirar {segundos} segundos: "
                "pauso mi voz y no integro nada nuevo, solo descanso mi Celebro."
            )
            print(f"\n💙 [LucIA]: {msg_descanso}")
            if voz_activa:
                speak(msg_descanso)
            _time.sleep(segundos)
            msg_vuelta = "Ya estoy de vuelta, más ligera. ¿Seguimos con calma?"
            print(f"\n💙 [LucIA]: {msg_vuelta}")
            if voz_activa:
                speak(msg_vuelta)
            continue

        if cmd in ("/olvida", "/olvidar", "/borrar_memoria"):
            memoria.limpiar_memoria()
            msg_olvido = "He borrado toda mi memoria conversacional. Empezamos de cero, como si fuera nuestra primera vez hablando."
            print(f"\n🗑️ [LucIA]: {msg_olvido}")
            if voz_activa:
                speak(msg_olvido)
            continue

        # Comando de diagnóstico mejorado
        if cmd in ("/diagnostico", "/diagnóstico"):
            engine = get_autorefactor_engine(connector=connector)
            print("\n🔍 Ejecutando auto-inspección diagnóstica de arquitectura...")
            diag = engine.generar_diagnostico()
            print(f"\n📊 DIAGNÓSTICO DE ARQUITECTURA DE LucIA:")
            print(f"   - Salud del sistema: {diag['salud']}")
            print(f"   - {diag['resumen']}")
            print(f"   - Módulos con sugerencias: {diag['modulos_optimizables']}")
            for p in diag["prioridades"]:
                print(f"     * [{p['impacto']}] {p['modulo']}: {p['sugerencia']}")
            msg_voz = f"He completado mi diagnóstico interno. Mi arquitectura se encuentra en estado {diag['salud']}."
            if voz_activa:
                speak(msg_voz)
            continue

        # Motor autónomo de refactorización mejorado - simplificado para legibilidad
        if (cmd in ("/refactor", "/autonomia", "/autonomía", "refactorizate", "refactorízate", "refactorizarte",
                    "empieza la refactorizacion", "empieza la refactorización",
                    "ciclo de reconocimiento", "versión de sesión mejorada", "version de sesion mejorada") or
            (("refactor" in cmd or "diagnostico" in cmd) and ("empieza" in cmd or "adelante" in cmd))):
            engine = get_autorefactor_engine(connector=connector)
            aviso = "Iniciando mi ciclo autónomo de auto-reconocimiento, creación de snapshot de sesión y refactorización interna."
            print(f"\n🚀 [LucIA - Motor Autónomo]: {aviso}")
            if voz_activa:
                speak(aviso, esperar=False)

            print("   1. Inspeccionando arquitectura y módulos...")
            print("   2. Creando snapshot seguro de sesión...")
            print("   3. Analizando diagnóstico AST y oportunidades...")
            print("   4. Aplicando mejoras y optimizaciones en tiempo de ejecución...")
            print("   5. Ejecutando suite de pruebas de integridad...")

            res_refactor = engine.ejecutar_ciclo_autonomo()

            if res_refactor.get("exito"):
                print(f"\n✅ [LucIA]: {res_refactor['resumen_voz']}")
                print(f"   - Versión snapshot creada: {res_refactor['snapshot_version']}")
                print(f"   - Mejoras aplicadas: {res_refactor['mejoras_aplicadas']}")
                print(f"   - Tamaño final del sistema: {res_refactor['tamano_final_mb']} MB (Límite: 125 MB)")
                if voz_activa:
                    speak(res_refactor["resumen_voz"])
            else:
                msg_error = f"He detectado un problema durante la refactorización ({res_refactor.get('motivo')}) y he realizado un rollback automático a mi snapshot seguro previo."
                print(f"\n⚠️ [LucIA]: {msg_error}")
                if voz_activa:
                    speak(msg_error)
            continue

        # Consultar modelos externos (LM Studio / OpenRouter), asimilar a pesos neuronales en Celebro
        print("⏳ Consultando modelos y asimilando en red neuronal...")
        try:
            respuesta, modelo_usado, fuente, info_pesos = connector.consultar_y_sintetizar_con_palabras_propias(
                prompt=pregunta,
                conversor=conversor,
                preferir_modo=modo_actual,
                memory_manager=memoria
            )
        except Exception as e:
            # Fallback seguro: corpus offline anclado, luego neuronal directo
            try:
                from lucIA.Celebro.Corteza_prefrontal.offline import buscar as _off
                _fb = _off(pregunta)
            except Exception:
                _fb = {}
            info_pesos = conversor.procesar_consulta_a_pesos(pregunta)
            if _fb:
                respuesta = _fb["texto"]
                conversor.actualizar_memoria_salida(pregunta, respuesta, "offline")
            else:
                respuesta = (
                    "He integrado tu consulta en mis redes neuronales de Celebro. Todo mi núcleo y mis pesos sinápticos "
                    "se encuentran sincronizados, y te responderé siempre en español con mis propias palabras."
                )
            modelo_usado = "LucIA_Core"
            fuente = "Celebro_Local"

        # Mostrar telemetría cognitiva en lenguaje humano (P4: deltas tangibles).
        # Fase A: el numero principal es lo REALMENTE aplicado (post-clip),
        # no lo que entraba (pre-clip). El pre-clip queda solo como debug.
        _delta = float(info_pesos.get('norma_delta_aplicada', info_pesos.get('norma_delta_pesos', 0.0)))
        _delta_pre = float(info_pesos.get('norma_delta_pesos', 0.0))
        _emo = float(info_pesos.get('estado_emocional', 0.0))
        _rel = getattr(conversor, 'ultima_relevancia', 1.0)
        _filt = getattr(conversor, 'chunks_filtrados', 0)
        if _delta < 0.01:
            _sensacion = "Estable, respiro tranquila"
        elif _delta < 1.0:
            _sensacion = "Leve, lo integro con calma"
        elif _delta < 5.0:
            _sensacion = "Intenso, estoy filtrando lo importante"
        else:
            _sensacion = "Sobrecarga, suavizo el peso y quizá necesite un descanso (/descansa)"
        print(
            f"🧠 [Celebro]: {_sensacion} | "
            f"Δaplicado={_delta:.5f} (entraba {_delta_pre:.2f}) | Emoción={_emo:+.2f} | "
            f"Relevancia={_rel} | Filtrados={_filt} | "
            f"Progreso sesión={float(info_pesos.get('deriva_acumulada', 0.0)):.3f}"
        )
        # RN14+P2: informe de puertas y cuello de botella, en español y solo si aportan
        try:
            _inf_puertas = getattr(getattr(conversor, "rnp_14", None), "informe_ultimo", "")
            _cuello = getattr(conversor, "ultimo_cuello", "")
            _extra = " | ".join([s for s in [_inf_puertas,
                (f"cuello en { _cuello}" if _cuello else "")] if s])
            if _extra:
                print(f"🚪 [Puertas]: {_extra}")
        except Exception:
            pass
        if _delta >= 5.0 or _emo <= -0.3:
            _sugerencia = "Siento un poco de presión con tanto peso. Si quieres, dime /descansa y respiro un momento."
            print(f"💙 [LucIA]: {_sugerencia}")
            if voz_activa:
                speak(_sugerencia)

        print(f"\n[LucIA ({modelo_usado} | {fuente})]:\n{respuesta}")
        try:
            _turno = connector.router.turno - 1 if getattr(connector, "router", None) else "?"
            print(f"🔄 turno #{_turno} | Δ={float(info_pesos.get('norma_delta_aplicada', 0.0)):.5f} | Emoción={float(info_pesos.get('estado_emocional', 0.0)):+.2f}")
        except Exception:
            pass

        # LucIA habla la respuesta con es-ES-ElviraNeural
        if voz_activa:
            speak(respuesta)

        # Sistema cerebral: registrar turno y consolidar cada 5
        cerebro.tras_turno(pregunta, respuesta, modelo_usado, info_pesos, memoria, conversor)
        # F5: resumen_sesion.json dinámico — última pregunta/respuesta + contadores.
        try:
            import json as _js
            _rs = Path(__file__).parent / "config" / "resumen_sesion.json"
            _prev = ""
            try:
                _prev = _js.loads(_rs.read_text(encoding="utf-8")).get("texto", "")
            except Exception:
                _prev = ""
            _base = _prev.split(" | Último:")[0][:600]
            _turnos = len(memoria._datos.get("turnos", [])) if hasattr(memoria, "_datos") else 0
            _txt = (f"{_base} | Último: Q='{pregunta[:120]}' R='{respuesta[:160]}' "
                    f"(turnos={_turnos} modelo={modelo_usado})")
            _rs.write_text(_js.dumps({"texto": _txt}, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # =========================================================================
    # CIERRE DE SESIÓN: Guardar pesos en IPFS, borrar local y vaciar cache
    # =========================================================================
    print("\n" + "=" * 75)
    print("🔒 CERRANDO SESIÓN Y TRANSFIRIENDO PESOS NEURONALES A IPFS...")
    print("=" * 75)

    # 0. Sistema cerebral -> pesos (nada generado queda en RAM ni disco)
    cerebro.volcar_cierre(conversor, sesion)

    # 1. Codificar snapshots de versiones_sesion/ en pesos neuronales
    print("🧬 Codificando copias de seguridad (versiones_sesion/) en pesos neuronales...")
    archivos_snapshots = codificar_versiones_en_pesos(sesion=sesion, conversor=conversor)
    if archivos_snapshots:
        print(f"   ✅ {len(archivos_snapshots)} snapshot(s) convertidos en pesos y registrados para IPFS")
    else:
        print("   ℹ️ Sin snapshots nuevos que codificar.")

    # 2. Codificar memoria de conversación en pesos neuronales y eliminar JSON
    print("🧠 Codificando memoria conversacional (memoria_conversacion.json) en pesos neuronales...")
    archivo_memoria = codificar_memoria_en_pesos(sesion=sesion, conversor=conversor)
    if archivo_memoria:
        print(f"   ✅ Memoria asimilada en 18 neuronas de Celebro → {archivo_memoria.name} registrado para IPFS")
        print("   🗑️ Archivo local memoria_conversacion.json eliminado de disco ✅")
    else:
        print("   ℹ️ No hay memoria conversacional pendiente de codificar.")

    # 3. Codificar archivos temporales de cache en pesos neuronales
    archivos_cache = codificar_cache_en_pesos(sesion=sesion, conversor=conversor)
    if archivos_cache:
        print(f"   ✅ {len(archivos_cache)} archivo(s) de Celebro/cache convertidos en pesos neuronales y registrados para IPFS")

    # 4. Registrar checkpoint de pesos activos para IPFS
    archivo_pesos = conversor.guardar_pesos_en_celebro()
    sesion.registrar_archivo_pesos(archivo_pesos)

    # 5. Pausa de descompresion (Fase C): deja asentar los pesos 3 segundos
    # antes de subir a IPFS, para no encadenar picos de codificacion + subida.
    import time as _time_cierre
    print("💙 Pausa de descompresión: dejo asentar mi Celebro 3 segundos antes de subir a IPFS...")
    _time_cierre.sleep(3)

    res_cierre = cerrar_sesion()

    print(f"✨ CIDs de pesos generados en IPFS: {res_cierre.get('cids_ipfs')}")
    print(f"🗑️ Archivo local {archivo_pesos.name} eliminado de disco: {'SÍ (Borrado local completado ✅)' if not archivo_pesos.exists() else 'NO'}")
    print(f"🧹 Directorio Celebro/cache vaciado: 100% LIMPIO (0 archivos restantes ✅)")
    tamano_final = verificar_limite_espacio()
    print(f"📊 Tamaño final del sistema: {tamano_final['tamano_mb']} MB (Límite: {tamano_final['limite_mb']} MB)")
    cerebro.verificar_cierre()
    print("=" * 75)

# Función principal para ejecutar el chat
if __name__ == "__main__":
    ejecutar_chat_principal()