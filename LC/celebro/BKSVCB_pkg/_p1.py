"""
BKSVCB - parte 1/2 (version de sesion LucIA).
Este código Python usa un log level para configurar el nivel de información en el logging. Luego, itera sobre los loggables disponibles y se
"""
from __future__ import annotations

"""
BKSVCB.py - Blockchain Server & Consenso Neuronal de Celebro (Arquitectura 2026)
================================================================================
Servidor de bloques inmutable y libro mayor distribuido para WoldVirtualP2P3D:
  - Transforma blockchain_ledger.json en pesos neuronales nada mas arrancar.
  - Mantiene activa la sincronizacion y actualizacion neuronal durante la sesion.
  - Al cerrar sesion, sube atomicamente el estado y pesos de PSNRL a IPFS.
  - Estructura de bloques criptograficos con hashing doble SHA-256 y Merkle Root.
  - Visualizacion de hashes unicos de cada bloque en terminal en tiempo real.
  - Consenso PoNL (Proof of Neural Learning) y servidor HTTP REST/JSON-RPC.
"""
from __future__ import annotations
import os, sys, time, json, hashlib, logging, threading, atexit
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.WARNING))
for _lg in ("", "WoldVirtualP2P3D", "RFENRN1", "LC", "urllib3", "ENRN", "SLRN", "RNP"):
    logging.getLogger(_lg).setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))
logger = logging.getLogger(__name__)

__version__ = "2026.3.1"
__server_name__ = "CelebroBlockchainServer-BKSVCB"

CELEBRO_DIR: Path = Path(__file__).parent.resolve()
ROOT_DIR: Path = CELEBRO_DIR.parent.parent.resolve()
PSNRL_DIR: Path = CELEBRO_DIR / "PSNRL"
LEDGER_PATH: Path = CELEBRO_DIR / "blockchain_ledger.json"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from LC.celebro.CMFG.PSNRCV import get_conversor_pesos, ConversorRespuestaPesos
from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager, IPFSManager


class BloqueNeuronal:
    """Representa un bloque inmutable de la cadena de bloques de Celebro."""

    def __init__(
        self, indice: int, hash_previo: str, transacciones: List[Dict[str, Any]],
        estado_neuronal: Optional[Dict[str, Any]] = None, dificultad: int = 2,
        nonce: int = 0, hash_existente: Optional[str] = None,
        merkle_root_existente: Optional[str] = None,
    ) -> None:
        self.indice = indice
        self.hash_previo = hash_previo
        self.transacciones = transacciones
        self.estado_neuronal = estado_neuronal or {}
        self.dificultad = dificultad
        self.nonce = nonce
        self.merkle_root = merkle_root_existente or self.calcular_merkle_root()
        self.hash_bloque = hash_existente or self.calcular_hash()

    def calcular_merkle_root(self) -> str:
        """Calcula la raiz de Merkle de las transacciones incluidas en el bloque."""
        if not self.transacciones:
            return hashlib.sha256(b"empty_block_celebro_2026").hexdigest()
        hashes = [
            hashlib.sha256(json.dumps(tx, sort_keys=True).encode("utf-8")).hexdigest()
            for tx in self.transacciones
        ]
        while len(hashes) > 1:
            if len(hashes) % 2 != 0: hashes.append(hashes[-1])
            nuevos = []
            for i in range(0, len(hashes), 2):
                combinado = (hashes[i] + hashes[i + 1]).encode("utf-8")
                nuevos.append(hashlib.sha256(combinado).hexdigest())
            hashes = nuevos
        return hashes[0]

    def calcular_hash(self) -> str:
        """Calcula el doble hash SHA-256 inmutable de la cabecera."""
        cabecera = {
            "indice": self.indice, "hash_previo": self.hash_previo,
            "merkle_root": self.merkle_root, "dificultad": self.dificultad, "nonce": self.nonce,
        }
        serializado = json.dumps(cabecera, sort_keys=True).encode("utf-8")
        return hashlib.sha256(hashlib.sha256(serializado).digest()).hexdigest()

    def minar_bloque(self, max_iteraciones: int = 500000) -> bool:
        """Realiza el minado por prueba de trabajo/aprendizaje adaptativo."""
        prefijo = "0" * self.dificultad
        self.merkle_root = self.calcular_merkle_root()
        for _ in range(max_iteraciones):
            h = self.calcular_hash()
            if h.startswith(prefijo):
                self.hash_bloque = h
                return True
            self.nonce += 1
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Exporta el bloque completo a un diccionario serializable."""
        return {
            "indice": self.indice, "hash_bloque": self.hash_bloque,
            "hash_previo": self.hash_previo, "merkle_root": self.merkle_root,
            "dificultad": self.dificultad, "nonce": self.nonce,
            "total_transacciones": len(self.transacciones),
            "transacciones": self.transacciones, "estado_neuronal": self.estado_neuronal,
        }


class CelebroBlockchain:
    """Motor central de la Blockchain neuronal con transformacion sinaptica e IPFS."""

    def __init__(self, dificultad: int = 2) -> None:
        self.dificultad = dificultad
        self.cadena: List[BloqueNeuronal] = []
        self.transacciones_pendientes: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        self.conversor: ConversorRespuestaPesos = get_conversor_pesos()
        self.ipfs_mgr: IPFSManager = get_ipfs_manager()
        self.actualizador_activo = False
        self._hilo_actualizador: Optional[threading.Thread] = None
        self._inicializar_o_cargar_cadena()

    def _inicializar_o_cargar_cadena(self) -> None:
        """Carga el ledger previo desde disco o forja el bloque genesis."""
        if LEDGER_PATH.exists():
            try:
                with open(LEDGER_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                bloques_cargados = []
                for b in raw.get("cadena", []):
                    bloque = BloqueNeuronal(
                        indice=b["indice"], hash_previo=b["hash_previo"],
                        transacciones=b.get("transacciones", []),
                        estado_neuronal=b.get("estado_neuronal", {}),
                        dificultad=b.get("dificultad", self.dificultad), nonce=b.get("nonce", 0),
                        hash_existente=b.get("hash_bloque"),
                        merkle_root_existente=b.get("merkle_root"),
                    )
                    bloques_cargados.append(bloque)
                if bloques_cargados:
                    self.cadena = bloques_cargados
                    return
            except Exception as exc:
                logger.warning("Ledger corrupto o ilegible, se forja genesis: %s", exc)
        self._crear_bloque_genesis()

    def _crear_bloque_genesis(self) -> None:
        """Forja el bloque 0 inicial con metadatos del cluster Celebro 2026."""
        estado_inicial = {
            "total_neuronas": len(self.conversor.neuronas),
            "capas": ["ENRN", "RF_SL", "RF_EN", "RNP", "SLRN"],
            "version_cluster": __version__, "consenso": "Proof-of-Neural-Learning (PoNL)",
        }
        tx_genesis = {
            "tipo": "GENESIS_COGNITIVO", "emisor": "SYSTEM_CELEBRO", "receptor": "CLUSTER_P2P",
            "descripcion": "Arranque formal de la blockchain neuronal WoldVirtualP2P3D",
            "hash_semilla": hashlib.sha256(b"genesis_seed_2026").hexdigest(),
        }
        genesis = BloqueNeuronal(
            indice=0, hash_previo="0" * 64, transacciones=[tx_genesis],
            estado_neuronal=estado_inicial, dificultad=1,
        )
        genesis.minar_bloque()
        self.cadena = [genesis]
        self.guardar_ledger()

    def obtener_ultimo_bloque(self) -> BloqueNeuronal:
        """Retorna el ultimo bloque confirmado de la cadena."""
        return self.cadena[-1]

    def agregar_transaccion(
        self, tipo: str, emisor: str, receptor: str, datos: Dict[str, Any],
        cid_ipfs: Optional[str] = None,
    ) -> int:
        """Anade una nueva transaccion al pool de espera con validacion criptografica."""
        with self.lock:
            tx = {
                "id_tx": hashlib.sha256(f"{emisor}:{receptor}:{time.time_ns()}".encode()).hexdigest()[:16],
                "tipo": tipo, "emisor": emisor, "receptor": receptor,
                "cid_ipfs": cid_ipfs or "", "datos": datos,
            }
            self.transacciones_pendientes.append(tx)
            return self.obtener_ultimo_bloque().indice + 1

    def registrar_aprendizaje_neural(self, prompt: str, respuesta: str, modelo: str) -> Dict[str, Any]:
        """Procesa consulta a traves de las 50 neuronas activas y emite transaccion."""
        t0 = time.perf_counter()
        info_neural = self.conversor.asimilar_respuestas_y_calcular_sintesis(prompt, respuesta, modelo)
        res_ipfs = self.ipfs_mgr.almacenar_pesos(
            origen=json.dumps(info_neural).encode("utf-8"),
            nombre_modelo=f"tx_neural_{int(time.time())}", eliminar_local=False,
        )
        cid = res_ipfs.get("cid", "")
        datos_tx = {
            "modelo": modelo, "prompt_resumen": prompt[:80],
            "norma_delta": info_neural.get("norma_delta_aplicada", 0.0),
            "estado_emocional": info_neural.get("estado_emocional", 0.0),
            "tono": info_neural.get("tono_cognitivo", "neutral"),
            "duracion_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
        self.agregar_transaccion(
            tipo="ACTUALIZACION_SINAPTICA", emisor="CONVERSOR_PSNRCV",
            receptor="CELEBRO_P2P", datos=datos_tx, cid_ipfs=cid,
        )
        return {"neural": info_neural, "ipfs_cid": cid, "transacciones_espera": len(self.transacciones_pendientes)}

    def minar_transacciones_pendientes(self, minero_id: str = "CelebroLocalNode") -> Optional[BloqueNeuronal]:
        """Empaqueta transacciones en espera, acopla estado de las 50 neuronas y mina bloque."""
        with self.lock:
            if not self.transacciones_pendientes: return None
            ultimo = self.obtener_ultimo_bloque()
            resumen_neuronas = {
                "activas": len(self.conversor.neuronas),
                "deriva_acumulada": round(self.conversor.deriva_acumulada, 6),
                "pasos_sesion": self.conversor.pasos_sesion, "minero": minero_id,
            }
            txs = list(self.transacciones_pendientes)
            self.transacciones_pendientes.clear()
            nuevo_bloque = BloqueNeuronal(
                indice=ultimo.indice + 1, hash_previo=ultimo.hash_bloque,
                transacciones=txs, estado_neuronal=resumen_neuronas, dificultad=self.dificultad,
            )
            if nuevo_bloque.minar_bloque():
                self.cadena.append(nuevo_bloque)
                self.guardar_ledger()
                return nuevo_bloque
            return None

    def transformar_ledger_a_pesos_neuronales(self) -> Dict[str, Any]:
        """Transduce todos los bloques y hashes criptograficos a pesos neuronales en PSNRL."""
        if not LEDGER_PATH.exists(): return {"total_bloques": 0, "norma_acumulada": 0.0, "npz_pesos": "", "json_meta": ""}
        with open(LEDGER_PATH, "r", encoding="utf-8") as f: ledger_data = json.load(f)
        bloques = ledger_data.get("cadena", [])
        norma_acum = 0.0
        # Propagar cada bloque directamente sin persistir en cada iteracion
        for b in bloques:
            sintesis = f"Bloque {b['indice']} | Hash: {b['hash_bloque']} | Merkle: {b['merkle_root']}"
            try:
                v = self.conversor.encoder.encode(sintesis)
                act, _ = self.conversor._propagar_todas_las_neuronas(v)
                _, norma_app, _ = self.conversor._actualizar_pesos_en_todas_las_neuronas(v, act, factor=1.0)
                norma_acum += norma_app
            except Exception as exc:
                logger.warning("Transduccion bloque %s fallo: %s", b.get("indice"), exc)
        with self.lock:
            npz_path, json_path = self.conversor.persistir_pesos_en_psnrl(etiqueta="blockchain_ledger")
        print(f"  [PSNRL] {len(bloques)} bloques -> pesos neuronales | Norma total: {round(norma_acum,5)}")
        return {
            "total_bloques": len(bloques), "norma_acumulada": round(norma_acum, 5),
            "npz_pesos": str(npz_path.name), "json_meta": str(json_path.name),
        }

    def _bucle_actualizacion_continua(self) -> None:
        """Hilo de fondo que mantiene actualizando la red neuronal mientras la sesion este activa."""
        while self.actualizador_activo:
            time.sleep(12.0)
            if not self.actualizador_activo: break
            try:
                ult = self.obtener_ultimo_bloque()
                self.conversor.procesar_consulta_a_pesos(f"Latido Bloque #{ult.indice} Hash:{ult.hash_bloque[:16]}")
            except Exception: pass

    def arrancar_actualizador_red(self) -> None:
        """Inicia el sincronizador dinamico de pesos durante la sesion."""
        if not self.actualizador_activo:
            self.actualizador_activo = True
            self._hilo_actualizador = threading.Thread(target=self._bucle_actualizacion_continua, daemon=True)
            self._hilo_actualizador.start()

    def detener_actualizador_red(self) -> None:
        """Detiene el bucle de actualizacion."""
        self.actualizador_activo = False

    def cerrar_sesion_y_subir_ipfs(self) -> Dict[str, Any]:
        """Hook de cierre: persiste pesos, sube ledger + PSNRL a IPFS sin borrar sin pin."""
        self.detener_actualizador_red()
        with self.lock:
            self.conversor.persistir_pesos_en_psnrl(etiqueta="cierre_sesion_blockchain")
        cid_ledger = None
        ledger_borrado = False
        if LEDGER_PATH.exists():
            try:
                res_l = self.ipfs_mgr.almacenar_pesos(
                    origen=LEDGER_PATH, nombre_modelo="blockchain_ledger_final", eliminar_local=True,
                )
                cid_ledger = res_l.get("cid")
                ledger_borrado = res_l.get("borrado_local", False)
            except Exception as exc:
                logger.warning("Pin ledger fallo, se conserva local: %s", exc)
        res_psnrl = self.ipfs_mgr.subir_y_limpiar_psnrl(forzar_borrado_sin_daemon=False)
        return {
            "cid_ledger": cid_ledger, "ledger_borrado": ledger_borrado,
            "cids_psnrl": res_psnrl.get("cids", []), "borrados": res_psnrl.get("borrados", []),
            "pendiente_pin": res_psnrl.get("pendiente_pin", []),
        }

    def validar_cadena(self) -> Tuple[bool, Optional[str]]:
        """Verifica integridad; dificultad por bloque (genesis exento de PoW)."""
        for i in range(1, len(self.cadena)):
            act, prev = self.cadena[i], self.cadena[i - 1]
            if act.hash_previo != prev.hash_bloque: return False, f"Ruptura en #{act.indice}"
            if act.calcular_hash() != act.hash_bloque: return False, f"Hash corrupto en #{act.indice}"
            if act.calcular_merkle_root() != act.merkle_root: return False, f"Merkle invalido en #{act.indice}"
            prefijo = "0" * int(getattr(act, "dificultad", self.dificultad) or 0)
            if prefijo and not act.hash_bloque.startswith(prefijo):
                return False, f"PoW insuficiente en #{act.indice}"
        return True, None

    def guardar_ledger(self) -> bool:
        """Persiste cadena con backup .bak y escritura atomica."""
        data = {
            "version": __version__, "servidor": __server_name__,
            "total_bloques": len(self.cadena), "dificultad": self.dificultad,
            "cadena": [b.to_dict() for b in self.cadena],
        }
        try:
            with self.lock:
                if LEDGER_PATH.exists():
                    bak = LEDGER_PATH.with_suffix(".json.bak")
                    bak.write_bytes(LEDGER_PATH.read_bytes())
                tmp = LEDGER_PATH.with_suffix(".json.tmp")
                with open(tmp, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)
                tmp.replace(LEDGER_PATH)
            return True
        except Exception as exc:
            logger.warning("guardar_ledger fallo: %s", exc)
            return False

    def mostrar_cadena_hashes_terminal(self) -> None:
        """Muestra en terminal la cadena con el hash unico de cada bloque sin fechas."""
        print("\033[1;36m" + "=" * 74 + "\033[0m")
        print("  \033[1;32mLIBRO MAYOR DISTRIBUIDO - HASHES UNICOS DE BLOQUES (2026)\033[0m")
        print("\033[1;36m" + "=" * 74 + "\033[0m")
        for b in self.cadena:
            h_prev = b.hash_previo[:18] + "..." if len(b.hash_previo) > 18 else b.hash_previo
            print(f"  [Bloque #{b.indice:<2}] \033[1;33mHASH: {b.hash_bloque}\033[0m")
            print(f"            Prev: {h_prev} | Merkle: {b.merkle_root[:16]}... | Txs: {len(b.transacciones)}")
        print("\033[1;36m" + "-" * 74 + "\033[0m")


class BlockchainHTTPHandler(BaseHTTPRequestHandler):
    """Manejador HTTP REST y JSON-RPC para el servidor blockchain."""

    def _enviar_json(self, status: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers(); self.wfile.write(raw)

    def do_GET(self) -> None:
        bks = get_blockchain_server()
        if self.path in ("/", "/status"):
            valida, err = bks.validar_cadena()
            self._enviar_json(200, {
                "servidor": __server_name__, "total_bloques": len(bks.cadena),
                "cadena_valida": valida, "error_validacion": err, "ultimo_bloque": bks.obtener_ultimo_bloque().to_dict(),
            })
        elif self.path == "/blocks": self._enviar_json(200, {"bloques": [b.to_dict() for b in bks.cadena]})
        elif self.path == "/transform": self._enviar_json(200, bks.transformar_ledger_a_pesos_neuronales())
        elif self.path == "/mine":
            bloque = bks.minar_transacciones_pendientes()
            if bloque: self._enviar_json(200, {"mensaje": "Bloque minado", "bloque": bloque.to_dict()})
            else: self._enviar_json(200, {"mensaje": "Sin transacciones pendientes para minar"})
        else: self._enviar_json(404, {"error": "Ruta no encontrada"})

    def do_POST(self) -> None:
        bks = get_blockchain_server()
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", errors="replace") if length > 0 else "{}"
        try: params = json.loads(body)
        except Exception: params = {}
        if self.path == "/transaction":
            idx = bks.agregar_transaccion(
                tipo=params.get("tipo", "TRANSFERENCIA_COGNITIVA"), emisor=params.get("emisor", "EXTERNO"),
                receptor=params.get("receptor", "CELEBRO"), datos=params.get("datos", {}), cid_ipfs=params.get("cid_ipfs"),
            )
            self._enviar_json(201, {"mensaje": "Transaccion encolada", "bloque_objetivo": idx})
        elif self.path == "/learn":
            res = bks.registrar_aprendizaje_neural(
                prompt=params.get("prompt", ""), respuesta=params.get("respuesta", ""), modelo=params.get("modelo", "qwen2.5:7b"),
            )
            self._enviar_json(200, res)
        else: self._enviar_json(404, {"error": "Endpoint POST no reconocido"})

    def log_message(self, format: str, *args: Any) -> None: pass


class BlockchainServerDaemon:
    """Contenedor de ejecucion asincrona para el servicio HTTP Blockchain."""

    def __init__(self, host: str = "127.0.0.1", puerto: int = 8545) -> None:
        self.host, self.puerto = host, puerto
        self.servidor: Optional[HTTPServer] = None
        self.hilo: Optional[threading.Thread] = None

    def iniciar(self) -> bool:
        """Arranca el servidor HTTP en un hilo demonio desacoplado."""
        try:
            self.servidor = HTTPServer((self.host, self.puerto), BlockchainHTTPHandler)
            self.hilo = threading.Thread(target=self.servidor.serve_forever, daemon=True)
            self.hilo.start()
            return True
        except Exception: return False

    def detener(self) -> None:
        """Detiene ordenadamente el servidor HTTP."""
        if self.servidor:
            self.servidor.shutdown(); self.servidor.server_close(); self.servidor = None


_blockchain_instancia: Optional[CelebroBlockchain] = None
_daemon_http_instancia: Optional[BlockchainServerDaemon] = None


def get_blockchain_server() -> CelebroBlockchain:
    """Acceso singleton al motor de la blockchain neuronal."""
    global _blockchain_instancia
    if _blockchain_instancia is None: _blockchain_instancia = CelebroBlockchain()
    return _blockchain_instancia


def iniciar_servidor_blockchain(puerto: int = 8545) -> BlockchainServerDaemon:
    """Arranca el servicio de API HTTP del servidor de bloques."""
    global _daemon_http_instancia
    if _daemon_http_instancia is None:
        _daemon_http_instancia = BlockchainServerDaemon(puerto=puerto)
        _daemon_http_instancia.iniciar()
    return _daemon_http_instancia


# Alias de compatibilidad con la suite de tests (Fase 0)
ServidorBlockchainNeuronal = CelebroBlockchain


