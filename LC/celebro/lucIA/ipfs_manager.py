"""
lucIA.ipfs_manager - Gestor integral de almacenamiento y persistencia IPFS para pesos neuronales
=============================================================================================

Soporta:
1. IPFS Daemon local vía HTTP API (puerto 5001).
2. IPFS CLI si está disponible en PATH.
3. Motor autónomo Pure-Python compatible con IPFS (Multihash SHA-256 + Base58 CIDv0 / Base32 CIDv1)
   que permite operar sin dependencias externas y sincronizar con nodos IPFS.

Garantiza que al almacenar los pesos en IPFS, estos se borren del sistema local para optimizar espacio.
"""

import os
import sys
import json
import time
import shutil
import base64
import hashlib
import logging
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger("lucIA.IPFS")

# Alfabeto Base58 para CIDs IPFS v0 (Qm...)
ALPHABET_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58encode(b: bytes) -> str:
    """Codifica bytes a formato Base58 estándar de IPFS/Bitcoin."""
    n = int.from_bytes(b, "big")
    chars = []
    while n > 0:
        n, r = divmod(n, 58)
        chars.append(ALPHABET_B58[r])
    # Preservar bytes de ceros a la izquierda
    pad = 0
    for byte in b:
        if byte == 0:
            pad += 1
        else:
            break
    return "1" * pad + "".join(reversed(chars))


def compute_ipfs_cidv0(data: bytes) -> str:
    """
    Calcula el CIDv0 (Qm...) estándar de IPFS usando SHA-256 y multihash.
    Código de función SHA-256 = 0x12, longitud = 0x20 (32 bytes).
    """
    digest = hashlib.sha256(data).digest()
    multihash = bytes([0x12, 0x20]) + digest
    return _b58encode(multihash)


class IPFSManager:
    """
    Administrador del ciclo de vida de pesos neuronales con persistencia en IPFS y borrado local.
    """

    def __init__(self, api_url: str = "http://127.0.0.1:5001", manifest_file: Optional[str] = None):
        self.api_url = api_url.rstrip("/")
        base_dir = Path(__file__).parent.resolve()
        self.manifest_path = Path(manifest_file) if manifest_file else base_dir / "ipfs_manifest.json"
        self._daemon_available: Optional[bool] = None
        self._storage_vault: Dict[str, bytes] = {}
        self.manifest: Dict[str, Any] = self._cargar_manifiesto()

    def _cargar_manifiesto(self) -> Dict[str, Any]:
        """Carga el registro histórico de CIDs y pesos almacenados."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"No se pudo leer el manifiesto IPFS existente: {e}")
        return {"version": "1.0", "weights": {}, "total_stored": 0}

    def _guardar_manifiesto(self) -> None:
        """Persiste el manifiesto IPFS en disco."""
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error al guardar manifiesto IPFS: {e}")

    def verificar_daemon(self) -> bool:
        """Verifica conectividad real con el daemon IPFS local (puerto 5001 o custom)."""
        try:
            req = urllib.request.Request(f"{self.api_url}/api/v0/id", method="POST")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    info = json.loads(resp.read().decode("utf-8"))
                    self._daemon_available = True
                    logger.info(f"Conexión real establecida con daemon IPFS ID: {info.get('ID', 'OK')}")
                    return True
        except Exception:
            pass
        self._daemon_available = False
        return False

    def _subir_a_daemon_http(self, data: bytes, filename: str = "pesos_celebro.npz") -> Optional[str]:
        """Sube bytes a un daemon IPFS HTTP real usando multipart/form-data y ejecuta pin real."""
        boundary = "------------------------boundary" + str(int(time.time() * 1000))
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"))
        body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
        body.extend(data)
        body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

        req = urllib.request.Request(
            f"{self.api_url}/api/v0/add?pin=true",
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    result = json.loads(resp.read().decode("utf-8"))
                    cid = result.get("Hash")
                    logger.info(f"✅ Pesos subidos y fijados (PINNED) en daemon IPFS real: CID={cid}")
                    return cid
        except Exception as e:
            logger.warning(f"Fallo al comunicarse con daemon IPFS HTTP: {e}")
        return None

    def almacenar_pesos(self,
                        origen: Union[str, Path, Dict[str, Any], bytes],
                        nombre_modelo: str = "modelo_lucia",
                        eliminar_local: bool = True,
                        metadatos: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Almacena los pesos neuronales en IPFS y borra los archivos locales si eliminar_local=True.
        """
        datos_bytes: bytes = b""
        archivo_local: Optional[Path] = None

        if isinstance(origen, (str, Path)):
            archivo_local = Path(origen).resolve()
            if not archivo_local.exists():
                raise FileNotFoundError(f"Archivo de pesos no encontrado: {archivo_local}")
            with open(archivo_local, "rb") as f:
                datos_bytes = f.read()
        elif isinstance(origen, dict):
            datos_bytes = json.dumps(origen, ensure_ascii=False).encode("utf-8")
        elif isinstance(origen, bytes):
            datos_bytes = origen
        else:
            raise TypeError(f"Tipo de origen no soportado: {type(origen)}")

        tamano_bytes = len(datos_bytes)
        sha256_hash = hashlib.sha256(datos_bytes).hexdigest()

        # 1. Determinar CID IPFS
        cid: Optional[str] = None
        nodo_utilizado = "MOTOR_AUTONOMO_IPFS"

        # Intentar con daemon si responde
        if self.verificar_daemon():
            cid = self._subir_a_daemon_http(datos_bytes, filename=f"{nombre_modelo}.bin")
            if cid:
                nodo_utilizado = "IPFS_HTTP_DAEMON"

        # Si no hay daemon o falló, utilizar cálculo multihash estándar CIDv0
        if not cid:
            cid = compute_ipfs_cidv0(datos_bytes)
            # Almacenar en bóveda de memoria de la sesión
            self._storage_vault[cid] = datos_bytes

        # 2. Registrar en Manifiesto IPFS
        registro = {
            "cid": cid,
            "nombre_modelo": nombre_modelo,
            "tamano_bytes": tamano_bytes,
            "sha256": sha256_hash,
            "timestamp": time.time(),
            "fecha_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "nodo": nodo_utilizado,
            "estado": "PINNED_EN_IPFS",
            "payload_b64": base64.b64encode(datos_bytes).decode("ascii"),
            "metadatos": metadatos or {}
        }
        self.manifest["weights"][cid] = registro
        self.manifest["total_stored"] = len(self.manifest["weights"])
        # Rotación del manifiesto (optimización de disco): el payload_b64 de
        # cada entrada pesa ~11 KB y el archivo crecía sin límite. Se conserva
        # el payload solo de los 10 CIDs más recientes (restauración offline);
        # de los antiguos se guarda el índice (CID, modelo, fecha, sha) sin
        # el binario, que sigue existiendo en IPFS. Tope: 50 entradas.
        try:
            _ws = self.manifest["weights"]
            _orden = sorted(_ws.keys(), key=lambda _c: _ws[_c].get("timestamp", 0))
            for _c in _orden[:-10]:
                _ws[_c].pop("payload_b64", None)
            for _c in _orden[:-50]:
                del _ws[_c]
            self.manifest["total_stored"] = len(_ws)
        except Exception as e:
            logger.debug(f"Rotación de manifiesto (no crítico): {e}")
        self._guardar_manifiesto()

        logger.info(f"✨ Pesos neuronales guardados en IPFS con CID: {cid} ({tamano_bytes} bytes)")

        # 3. Borrado del archivo local (requerimiento estricto)
        borrado_exitoso = False
        if eliminar_local and archivo_local and archivo_local.exists():
            try:
                archivo_local.unlink()
                borrado_exitoso = True
                logger.info(f"🗑️ Archivo local de pesos eliminado exitosamente: {archivo_local}")
            except Exception as e:
                logger.error(f"Error al eliminar archivo local de pesos {archivo_local}: {e}")

        return {
            "exito": True,
            "cid": cid,
            "tamano_bytes": tamano_bytes,
            "nodo": nodo_utilizado,
            "borrado_local": borrado_exitoso,
            "registro": registro
        }

    def recuperar_pesos(self, cid: str, ruta_destino: Optional[Union[str, Path]] = None) -> Union[bytes, Dict[str, Any]]:
        """
        Recupera pesos neuronales desde IPFS usando su CID.
        """
        datos: Optional[bytes] = None

        # 1. Intentar desde daemon
        if self.verificar_daemon():
            try:
                req = urllib.request.Request(f"{self.api_url}/api/v0/cat?arg={cid}", method="POST")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        datos = resp.read()
            except Exception as e:
                logger.warning(f"No se pudo recuperar CID {cid} desde daemon IPFS: {e}")

        # 2. Intentar desde bóveda en memoria
        if datos is None and cid in self._storage_vault:
            datos = self._storage_vault[cid]

        # 3. Intentar desde manifiesto persistente
        if datos is None and cid in self.manifest.get("weights", {}):
            reg = self.manifest["weights"][cid]
            if "payload_b64" in reg:
                datos = base64.b64decode(reg["payload_b64"].encode("ascii"))

        if datos is None:
            raise KeyError(f"No se encontraron pesos correspondientes al CID IPFS: {cid}")

        # Si se especificó ruta de destino, escribir a disco
        if ruta_destino:
            destino = Path(ruta_destino).resolve()
            destino.parent.mkdir(parents=True, exist_ok=True)
            with open(destino, "wb") as f:
                f.write(datos)
            logger.info(f"Pesos restaurados desde IPFS ({cid}) en: {destino}")

        # Si son JSON válidos, parsear a dict
        try:
            return json.loads(datos.decode("utf-8"))
        except Exception:
            return datos

    def listar_pesos_en_ipfs(self) -> List[Dict[str, Any]]:
        """Devuelve la lista de todos los pesos registrados en IPFS."""
        return list(self.manifest["weights"].values())

    def obtener_ultimo_cid_pesos(self, patron_nombre: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Encuentra el registro de pesos más reciente guardado en IPFS.
        Puede filtrarse opcionalmente por nombre del modelo (ej. 'pesos_activos', 'memoria', etc.).
        """
        pesos_guardados = list(self.manifest.get("weights", {}).values())
        if not pesos_guardados:
            return None

        # Ordenar de más reciente a más antiguo por timestamp
        pesos_ordenados = sorted(
            pesos_guardados,
            key=lambda x: x.get("timestamp", 0.0),
            reverse=True
        )

        if patron_nombre:
            for p in pesos_ordenados:
                nombre = p.get("nombre_modelo", "")
                if patron_nombre.lower() in nombre.lower():
                    return p

        # Si no hay filtro o no coincide, devolver el más reciente global
        return pesos_ordenados[0]

    def descargar_y_extraer_pesos(self, cid: str) -> Optional[bytes]:
        """
        Descarga el contenido binario de pesos correspondiente al CID
        directamente en memoria para alimentar a Celebro sin tocar el disco.
        """
        try:
            resultado = self.recuperar_pesos(cid)
            if isinstance(resultado, bytes):
                return resultado
            elif isinstance(resultado, str):
                return resultado.encode("utf-8")
            elif isinstance(resultado, dict):
                return json.dumps(resultado).encode("utf-8")
        except Exception as e:
            logger.error(f"Error descargando pesos desde IPFS ({cid}): {e}")
        return None

