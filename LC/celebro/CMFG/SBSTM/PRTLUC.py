"""
PRTLUC.py - Arranque del orquestador LucIA (2026)
__init__, inicializar_subsistemas (pasos 0-8) y helpers de IA local.
Mixin de OrquestadorSistemaLucIA; importa desde BASELUC.
"""
from __future__ import annotations

from LC.celebro.CMFG.SBSTM.BASELUC import (
    Any, Dict, List, Optional, Tuple, atexit, logger, os, threading, time,
    EstiloTerminalLucIA, GestorEntornoSeguro, RefactorizadorSesion,
    _HRCTRC_DISPONIBLE, _HRCTRC_RFCT_DISPONIBLE, _IALOCAL_DISPONIBLE,
    _INTEGRACIONRF_DISPONIBLE, _MDSTM_DISPONIBLE, _descargar_recomendados,
    _perfilar_hw_local, _recomendar_ia_local, banner_bienvenida,
    consultar_lucia_local, get_cliente_iafree, get_gestor_hrctrc,
    get_gestor_mdstm, get_integrador, refactorizar_overlay,
    _ROTACIONIA_DISPONIBLE, get_rotador_ia,
)


class PRTLUCMixin:
    """Mezcla de arranque: estado inicial y secuencia de subsistemas."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.servidor_bks: Optional[Any] = None
        self.conversor_psn: Optional[Any] = None
        self.sesion_p2p: Optional[Any] = None
        self.cliente_iafree: Optional[Any] = None
        self.rotador_ia: Optional[Any] = None
        self.activa = False
        self.turno_actual = 0
        self.puerto_bks = int(os.getenv("LUCIA_BKS_PORT", "8545"))
        self.sesion_id = f"LUCIA_{time.strftime('%Y%m%d_%H%M%S')}"
        # DSIALCLGRG: perfil HW + estado IA local (autonomo, sin bloquear arranque)
        self.perfil_hw_local: Optional[Dict[str, Any]] = None
        self.ia_local_lista: bool = False
        if _ROTACIONIA_DISPONIBLE and get_rotador_ia is not None:
            try:
                self.rotador_ia = get_rotador_ia()
            except Exception:
                self.rotador_ia = None
        # MDSTM: gestor de descarga autonoma (clase real, no promesa)
        self.gestor_mdstm: Optional[Any] = None
        if _MDSTM_DISPONIBLE and get_gestor_mdstm is not None:
            try:
                self.gestor_mdstm = get_gestor_mdstm()
            except Exception:
                self.gestor_mdstm = None
        # HRCTRC: herrero constructor (overlay de sesion en LC/Constructor)
        self.gestor_hrctrc: Optional[Any] = None
        if _HRCTRC_DISPONIBLE and get_gestor_hrctrc is not None:
            try:
                self.gestor_hrctrc = get_gestor_hrctrc(sesion_id=self.sesion_id)
            except Exception:
                self.gestor_hrctrc = None
        # HRCTRC_RFCT: refactorizador de version de sesion (regla 400/450)
        self.refactorizador: Optional[Any] = None
        # INTEGRACIONRF: bitacora .md -> pesos -> IPFS -> devopencode
        self.integrador: Optional[Any] = None
        if _INTEGRACIONRF_DISPONIBLE and get_integrador is not None:
            try:
                self.integrador = get_integrador(sesion_id=self.sesion_id)
            except Exception:
                self.integrador = None
        atexit.register(self.cerrar_sistema)

    def inicializar_subsistemas(self) -> bool:
        """Inicializa en secuencia la blockchain, transductor, IAFREE y credenciales."""
        # 0. Recolectar caché Python pre-existente hacia CHG/
        try:
            from LC.celebro.CMFG.SBSTM.PURGADOR import recolectar_pycache_en_chg
            n_chg = recolectar_pycache_en_chg()
            if n_chg:
                print(f"  [0/4] Cache Python → CHG   : \033[38;5;51m{n_chg} elementos\033[0m")
        except Exception:
            pass
        # 1. Cargar .env y subsistema IAFREE
        GestorEntornoSeguro.cargar_variables()
        if get_cliente_iafree is not None:
            self.cliente_iafree = get_cliente_iafree()
            modelo_ini = self.cliente_iafree.gestor.obtener_modelo_activo()["id"]
        else:
            modelo_ini = "openrouter/free"

        # Banner visual futurista estilo OpenCode
        try:
            if EstiloTerminalLucIA:
                banner_bienvenida(self.sesion_id, modelo_ini)
            else:
                print(f"\n[+] WoldVirtualP2P3D -- LucIA Console 2026 | Mod: {modelo_ini}")
        except UnicodeEncodeError:
            print(f"\n[+] WoldVirtualP2P3D -- LucIA Console 2026 | Mod: {modelo_ini} | sesion={self.sesion_id}")

        # 2. Cargar Blockchain BKSVCB
        try:
            from LC.celebro.BKSVCB import get_blockchain_server, iniciar_servidor_blockchain
            self.servidor_bks = get_blockchain_server()
            print(f"  [1/4] Blockchain BKSVCB    : \033[38;5;48mACTIVA\033[0m | Bloques: \033[38;5;220m{len(self.servidor_bks.cadena)}\033[0m")
            self.servidor_bks.mostrar_cadena_hashes_terminal()
            for intento in range(3):
                try:
                    iniciar_servidor_blockchain(puerto=self.puerto_bks)
                    break
                except OSError as exc:
                    logger.warning("Puerto %s ocupado (intento %s): %s", self.puerto_bks, intento + 1, exc)
                    self.puerto_bks += 1
            else:
                print("  [1/4] HTTP REST: no disponible, sigo sin servidor HTTP")
        except Exception as e_bks:
            print(f"  [1/4] Blockchain BKSVCB    : \033[38;5;203mERROR ({e_bks})\033[0m")
            return False

        # 3. Cargar Conversor PSNRCV (50 Neuronas)
        try:
            from LC.celebro.CMFG.PSNRCV import get_conversor_pesos
            self.conversor_psn = get_conversor_pesos()
            num_neu = len(self.conversor_psn.neuronas)
            print(f"  [2/4] Conversor PSNRCV     : \033[38;5;48mOK\033[0m | Sinapsis: \033[38;5;141m{num_neu} Activas\033[0m")
        except Exception as e_psn:
            print(f"  [2/4] Conversor PSNRCV     : \033[38;5;203mERROR ({e_psn})\033[0m")
            return False

        # 4. Transducir Ledger a Pesos y arrancar latido
        try:
            res_t = self.servidor_bks.transformar_ledger_a_pesos_neuronales()
            self.servidor_bks.arrancar_actualizador_red()
            print(f"  [3/4] Ledger -> Pesos PSNRL: \033[38;5;48m{res_t.get('total_bloques', 0)} bloques\033[0m | Norma: \033[38;5;214m{res_t.get('norma_acumulada', 0.0)}\033[0m")
        except Exception as e_led:
            print(f"  [3/4] Ledger -> Pesos      : \033[38;5;214mAVISO ({e_led})\033[0m")

        # 5. Enlace con SubSistema de Sesion SBSTM / SNSBSTNPRB
        try:
            from LC.celebro.CMFG.SBSTM import obtener_clase_sesion
            cls_sesion = obtener_clase_sesion()
            self.sesion_p2p = cls_sesion()
            print(f"  [4/4] Sesion SNSBSTNPRB    : \033[38;5;48mACOPLADA\033[0m | Motor STYLOS: \033[38;5;51mACTIVO\033[0m\n")
        except Exception as e_sbs:
            print(f"  [4/4] Sesion SNSBSTNPRB    : \033[38;5;214mAVISO ({e_sbs})\033[0m\n")

        # 6. DSIALCLGRG: registra VRAM/RAM y deja IA local lista (autonomo)
        self._inicializar_ia_local()

        # 7. HRCTRC: abre la copia de trabajo en LC/Constructor para la sesion
        if self.gestor_hrctrc is not None:
            try:
                rep = self.gestor_hrctrc.iniciar_sesion()
                print(f"  [6/6] Constructor HRCTRC    : \033[38;5;48mACTIVO\033[0m | "
                      f"Overlay: \033[38;5;51m{rep.get('archivos_versionados', 0)} archivos\033[0m")
            except Exception as exc:
                print(f"  [6/6] Constructor HRCTRC    : \033[38;5;214mAVISO ({exc})\033[0m")

        # 8. HRCTRC_RFCT: version de sesion con barra de progreso (modelo local)
        self._version_sesion_inicial()

        # 9. HRCNTR: monitor y refactorizador neural del sistema
        self._hrctnr_refactor_inicial()

        self.activa = True
        return True

    def _version_sesion_inicial(self) -> None:
        """Barra de progreso al arrancar: divide oversized a 400/450 con IA local."""
        if (not _HRCTRC_RFCT_DISPONIBLE or refactorizar_overlay is None
                or self.gestor_hrctrc is None):
            return
        try:
            overlay = self.gestor_hrctrc.overlay
            self.refactorizador = RefactorizadorSesion(overlay)
            print("  [7/7] Version de sesion 400/450 (modelo local):")
            rep = self.refactorizador.ejecutar(mostrar_barra=True)
            print(f"        \033[38;5;48m{rep.get('mensaje')}\033[0m")
        except Exception as exc:
            print(f"  [7/7] Version de sesion      : \033[38;5;214mAVISO ({exc})\033[0m")

        # 8. INTEGRACIONRF: snapshot de rutas originales + bitacora viva
        if self.integrador is not None:
            try:
                snap = self.integrador.snapshot_inicio()
                print(f"  [8/8] Integracion RF          : \033[38;5;48mARMADA\033[0m | "
                      f"Rutas: \033[38;5;51m{snap.get('rutas', 0)}\033[0m | .md->pesos->IPFS->devopencode")
            except Exception as exc:
                print(f"  [8/8] Integracion RF          : \033[38;5;214mAVISO ({exc})\033[0m")

    def _hrctnr_refactor_inicial(self) -> None:
        """Barra de progreso: refactor neural al arrancar."""
        try:
            from LC.celebro.CMFG.SBSTM.HRCNTR import ejecutar_hrctnr
            print("  [9/9] Refactor Neural HRCNTR:")
            rep = ejecutar_hrctnr(mostrar_barra=True)
            print(f"        \033[38;5;48m{rep.get('mensaje')}\033[0m")
        except Exception as exc:
            print(f"  [9/9] Refactor Neural HRCNTR: "
                  f"\033[38;5;214mAVISO ({exc})\033[0m")

    def _inicializar_ia_local(self) -> None:
        """Perfila el hardware y marca la IA local como disponible para LucIA."""
        if not _IALOCAL_DISPONIBLE or _perfilar_hw_local is None:
            print("  [5/5] IA local DSIALCLGRG   : \033[38;5;214mNO DISPONIBLE\033[0m")
            return
        try:
            self.perfil_hw_local = _perfilar_hw_local(guardar=True)
            recs = _recomendar_ia_local(self.perfil_hw_local) if _recomendar_ia_local else []
            self.ia_local_lista = True
            print(f"  [5/5] IA local DSIALCLGRG   : \033[38;5;48mLISTA\033[0m | "
                  f"RAM: \033[38;5;220m{self.perfil_hw_local.get('ram_total_gb')}GB\033[0m | "
                  f"VRAM: \033[38;5;220m{self.perfil_hw_local.get('vram_total_gb')}GB\033[0m | "
                  f"Sugerido: \033[38;5;51m{recs[0]['id'] if recs else 'ninguno'}\033[0m")
        except Exception as exc:
            print(f"  [5/5] IA local DSIALCLGRG   : \033[38;5;214mAVISO ({exc})\033[0m")

    def descargar_ia_local_autonomo(self, limite: int = 2) -> List[Dict[str, Any]]:
        """Descarga autonoma de modelos ligeros segun el perfil HW (hilo o turno)."""
        if not _IALOCAL_DISPONIBLE or _descargar_recomendados is None:
            return [{"exito": False, "mensaje": "DSIALCLGRG no disponible"}]
        try:
            return list(_descargar_recomendados(limite=limite))
        except Exception as exc:
            return [{"exito": False, "mensaje": str(exc)}]

    def _responder_ia_local(self, prompt: str, estado_previo: Dict[str, Any]) -> Tuple[str, str, float]:
        """Fallback autonomo: consulta Ollama/GGUF local cuando OpenRouter falla."""
        if not _IALOCAL_DISPONIBLE or consultar_lucia_local is None:
            return "", "sin-ia-local", 0.0
        try:
            texto, mid, lat = consultar_lucia_local(prompt, estado_previo)
            if texto and mid != "local:reflejo":
                return texto, mid, lat
            return "", "local:reflejo", 0.0
        except Exception:
            return "", "local:error", 0.0
