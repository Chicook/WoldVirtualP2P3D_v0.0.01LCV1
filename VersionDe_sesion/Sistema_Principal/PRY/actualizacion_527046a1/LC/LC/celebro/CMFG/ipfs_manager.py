"""
ipfs_manager.py - Gestor IPFS Real para Pesos Neuronales de Celebro (Arquitectura 2026)
========================================================================================
Conexion real a IPFS via Kubo HTTP RPC API v0 y CLI:
  - Auto-discovery de daemon Kubo en puertos 5001/5002/5003.
  - IPFS CLI como fallback real con binario local IPFSonl o PATH del sistema.
  - Motor puro Python CIDv1 (dag-raw + sha2-256 + base32lower) para operacion offline.
  - Subida en batch de todos los pesos de PSNRL al cerrar sesion.
  - Borrado garantizado del archivo local tras confirmacion de pin exitoso.
  - Manifiesto rotativo TTL 50 entradas (payload_b64 solo en las 10 mas recientes).
  - Compatibilidad directa con pesos_vivos.py / PSNRCV.py.
"""
from __future__ import annotations
import os, sys, json, time, base64, hashlib, logging, subprocess
import urllib.request, urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("WoldVirtualP2P3D.IPFS")

_KUBO_PORTS = [5001, 5002, 5003]
_KUBO_TIMEOUT_FAST = 1.5
_KUBO_TIMEOUT_UP = 30
_MULTICODEC_RAW = 0x55
_MULTIHASH_SHA256 = (0x12, 0x20)
_BASE32_PREFIX = "b"

_LOCAL_IPFS_BIN = Path(__file__).parent / "IPFSonl" / "ipfs.exe"
_LOCAL_IPFS_ENV = {**os.environ, "IPFS_PATH": str(Path(__file__).parent / "IPFSonl" / ".ipfs")}


def _encode_varint(n: int) -> bytes:
    """Codificacion protobuf varint para CIDv1."""
    buf = []
    while n > 0x7F:
        buf.append((n & 0x7F) | 0x80)
        n >>= 7
    buf.append(n)
    return bytes(buf)


def compute_cidv1_raw(data: bytes) -> str:
    """
    CIDv1 dag-raw + SHA2-256 + multibase base32lower. Estandar IPFS 2026.
    Formato: b + base32lower(varint(1) + varint(0x55) + multihash_sha256)
    """
    digest = hashlib.sha256(data).digest()
    mhash = bytes(_MULTIHASH_SHA256) + digest
    cid_raw = _encode_varint(1) + _encode_varint(_MULTICODEC_RAW) + mhash
    b32 = base64.b32encode(cid_raw).decode("ascii").lower().rstrip("=")
    return _BASE32_PREFIX + b32


class IPFSManager:
    """
    Gestor de ciclo de vida de pesos neuronales con persistencia real en IPFS.
    Prioridad de subida: daemon Kubo HTTP RPC > CLI kubo/ipfs > motor autonomo CIDv1.
    """

    def __init__(self, manifest_file: Optional[str] = None,
                 api_url: Optional[str] = None) -> None:
        self._base_dir = Path(__file__).parent.resolve()
        self._psnrl_dir = self._base_dir.parent / "PSNRL"
        self._psnrl_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(manifest_file) if manifest_file else self._base_dir / "ipfs_manifest.json"
        self._api_url = api_url
        self._daemon_url: Optional[str] = None
        self._vault: Dict[str, bytes] = {}
        self.manifest: Dict[str, Any] = self._cargar_manifiesto()

    def _cargar_manifiesto(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Manifiesto ilegible, reiniciando: %s", e)
        return {"version": "2.0", "weights": {}, "total_stored": 0}

    def _guardar_manifiesto(self) -> None:
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Error guardando manifiesto: %s", e)

    def _rotar_manifiesto(self) -> None:
        """Max 50 entradas; payload_b64 solo en las 10 mas recientes."""
        ws = self.manifest.get("weights", {})
        orden = sorted(ws.keys(), key=lambda c: ws[c].get("timestamp", 0))
        for c in orden[:-10]:
            ws[c].pop("payload_b64", None)
        for c in orden[:-50]:
            del ws[c]
        self.manifest["total_stored"] = len(ws)

    def _probe_url(self, url: str) -> bool:
        try:
            req = urllib.request.Request(f"{url}/api/v0/id", method="POST")
            with urllib.request.urlopen(req, timeout=_KUBO_TIMEOUT_FAST) as r:
                if r.status == 200:
                    info = json.loads(r.read().decode("utf-8", errors="replace"))
                    logger.info("Daemon IPFS OK | PeerID: %.16s | %s", info.get("ID", "?"), url)
                    return True
        except Exception:
            pass
        return False

    def descubrir_daemon(self) -> Optional[str]:
        """Auto-discovery Kubo en localhost puertos 5001/5002/5003."""
        if self._daemon_url and self._probe_url(self._daemon_url):
            return self._daemon_url
        candidatos: List[str] = []
        if self._api_url:
            candidatos.append(self._api_url.rstrip("/"))
        for p in _KUBO_PORTS:
            candidatos.append(f"http://127.0.0.1:{p}")
        for url in candidatos:
            if self._probe_url(url):
                self._daemon_url = url
                return url
        self._daemon_url = None
        return None

    def _subir_daemon(self, data: bytes, filename: str) -> Optional[str]:
        """Multipart POST a Kubo /api/v0/add con CIDv1 + pin=true."""
        url = self.descubrir_daemon()
        if not url:
            return None
        boundary = f"kubo2026{int(time.time_ns())}"
        body = bytearray()
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()
        body += data
        body += f"\r\n--{boundary}--\r\n".encode()
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        }
        try:
            req = urllib.request.Request(
                f"{url}/api/v0/add?pin=true&cid-version=1&hash=sha2-256&raw-leaves=true",
                data=bytes(body), headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=_KUBO_TIMEOUT_UP) as r:
                if r.status == 200:
                    raw = r.read().decode("utf-8", errors="replace")
                    last = [ln for ln in raw.strip().splitlines() if ln.strip()][-1]
                    res = json.loads(last)
                    cid = res.get("Hash") or res.get("Cid", {}).get("/")
                    if cid:
                        logger.info("IPFS daemon subida OK | CID=%s | %d B", cid, len(data))
                        return cid
        except Exception as e:
            logger.warning("Fallo daemon IPFS: %s", e)
        return None

    def _subir_cli(self, data: bytes, nombre: str) -> Optional[str]:
        """ipfs add via CLI con CIDv1 y raw-leaves. Prioriza binario local de IPFSonl."""
        tmp = self._base_dir / f"_tmp_{int(time.time_ns())}.bin"
        try:
            tmp.write_bytes(data)
            candidatos = []
            if _LOCAL_IPFS_BIN.exists():
                candidatos.append(([str(_LOCAL_IPFS_BIN)], _LOCAL_IPFS_ENV))
            candidatos += [(["ipfs"], None), (["kubo"], None)]
            for cmd, env in candidatos:
                try:
                    kw: Dict[str, Any] = {"capture_output": True, "text": True, "timeout": 30}
                    if env:
                        kw["env"] = env
                    res = subprocess.run(
                        cmd + ["add", "--pin=true", "--cid-version=1",
                               "--hash=sha2-256", "--raw-leaves", "-q", str(tmp)],
                        **kw
                    )
                    if res.returncode == 0:
                        cid = res.stdout.strip().splitlines()[-1].strip()
                        if cid.startswith(("Qm", "b", "B")):
                            logger.info("IPFS CLI OK | CID=%s | bin=%s", cid, cmd[0])
                            return cid
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
        except Exception as e:
            logger.warning("Error CLI IPFS: %s", e)
        finally:
            if tmp.exists():
                tmp.unlink()
        return None

    def almacenar_pesos(
        self,
        origen: Union[str, Path, bytes, dict],
        nombre_modelo: str = "pesos_celebro",
        eliminar_local: bool = True,
        metadatos: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Almacena pesos en IPFS (daemon > CLI > autonomo). Borra local si pin OK."""
        archivo_local: Optional[Path] = None
        if isinstance(origen, (str, Path)):
            archivo_local = Path(origen).resolve()
            if not archivo_local.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {archivo_local}")
            data = archivo_local.read_bytes()
            filename = archivo_local.name
        elif isinstance(origen, dict):
            data = json.dumps(origen, ensure_ascii=False).encode("utf-8")
            filename = f"{nombre_modelo}.json"
        elif isinstance(origen, bytes):
            data = origen
            filename = f"{nombre_modelo}.bin"
        else:
            raise TypeError(f"Tipo no soportado: {type(origen)}")

        sha256 = hashlib.sha256(data).hexdigest()
        cid = self._subir_daemon(data, filename)
        nodo = "AUTONOMO_CIDv1"
        subida_real = False

        if cid:
            nodo = "KUBO_HTTP_RPC"
            subida_real = True
        else:
            cid = self._subir_cli(data, nombre_modelo)
            if cid:
                nodo = "KUBO_CLI"
                subida_real = True
        if not cid:
            cid = compute_cidv1_raw(data)
            self._vault[cid] = data

        registro: Dict[str, Any] = {
            "cid": cid, "nombre_modelo": nombre_modelo,
            "tamano_bytes": len(data), "sha256": sha256,
            "timestamp": time.time(),
            "fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "nodo": nodo, "subida_real": subida_real,
            "estado": "PINNED" if subida_real else "LOCAL_CIDv1",
            "payload_b64": base64.b64encode(data).decode("ascii"),
            "metadatos": metadatos or {},
        }
        self.manifest.setdefault("weights", {})[cid] = registro
        self._rotar_manifiesto()
        self._guardar_manifiesto()
        logger.info("Almacenado | CID=%s | %s | %d B", cid, nodo, len(data))

        borrado = False
        if eliminar_local and archivo_local and archivo_local.exists() and subida_real:
            try:
                archivo_local.unlink()
                borrado = True
                logger.info("Local borrado: %s", archivo_local.name)
            except Exception as e:
                logger.error("Error borrando %s: %s", archivo_local, e)

        return {"exito": True, "cid": cid, "nodo": nodo, "subida_real": subida_real,
                "tamano_bytes": len(data), "borrado_local": borrado, "registro": registro}

    def recuperar_pesos(self, cid: str,
                        destino: Optional[Union[str, Path]] = None) -> Union[bytes, dict]:
        """Recupera pesos desde daemon > CLI > vault > manifiesto."""
        data: Optional[bytes] = None
        url = self.descubrir_daemon()
        if url:
            try:
                req = urllib.request.Request(f"{url}/api/v0/cat?arg={cid}", method="POST")
                with urllib.request.urlopen(req, timeout=_KUBO_TIMEOUT_UP) as r:
                    if r.status == 200:
                        data = r.read()
            except Exception as e:
                logger.warning("Cat daemon fallo (%s...): %s", cid[:10], e)

        if data is None:
            for cmd in (["ipfs"], ["kubo"]):
                try:
                    res = subprocess.run(cmd + ["cat", cid], capture_output=True, timeout=20)
                    if res.returncode == 0 and res.stdout:
                        data = res.stdout
                        break
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue

        if data is None and cid in self._vault:
            data = self._vault[cid]
        if data is None:
            reg = self.manifest.get("weights", {}).get(cid, {})
            if "payload_b64" in reg:
                data = base64.b64decode(reg["payload_b64"].encode("ascii"))
        if data is None:
            raise KeyError(f"CID no encontrado: {cid}")
        if destino:
            p = Path(destino)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            logger.info("Restaurado en: %s", p)
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            return data

    def subir_y_limpiar_psnrl(self, forzar_borrado_sin_daemon: bool = False) -> Dict[str, Any]:
        """Sube todos los archivos de PSNRL a IPFS y los borra localmente tras confirmar."""
        archivos = [f for f in self._psnrl_dir.iterdir()
                    if f.is_file() and not f.name.startswith(".")]
        if not archivos:
            return {"archivos_procesados": 0, "cids": [], "borrados": [], "errores": []}

        cids, borrados, errores = [], [], []
        daemon_ok = bool(self.descubrir_daemon())
        cli_ok = self._cli_disponible()

        for archivo in archivos:
            try:
                res = self.almacenar_pesos(
                    origen=archivo, nombre_modelo=archivo.stem,
                    eliminar_local=(daemon_ok or cli_ok or forzar_borrado_sin_daemon),
                )
                cids.append(res["cid"])
                if res["borrado_local"]:
                    borrados.append(archivo.name)
            except Exception as e:
                errores.append(f"{archivo.name}: {e}")
                logger.error("Error subiendo %s: %s", archivo.name, e)

        logger.info("PSNRL->IPFS | %d arch | %d borrados | %d errores",
                    len(archivos), len(borrados), len(errores))
        return {"archivos_procesados": len(archivos), "cids": cids,
                "borrados": borrados, "errores": errores,
                "daemon_real": daemon_ok, "cli_real": cli_ok}

    def listar_pesos(self) -> List[Dict[str, Any]]:
        ws = self.manifest.get("weights", {})
        return sorted(ws.values(), key=lambda x: x.get("timestamp", 0), reverse=True)

    def ultimo_cid(self, patron: Optional[str] = None) -> Optional[Dict[str, Any]]:
        items = self.listar_pesos()
        if patron:
            items = [i for i in items if patron.lower() in i.get("nombre_modelo", "").lower()]
        return items[0] if items else None

    def descargar_bytes(self, cid: str) -> Optional[bytes]:
        try:
            res = self.recuperar_pesos(cid)
            if isinstance(res, bytes): return res
            if isinstance(res, str):   return res.encode("utf-8")
            if isinstance(res, dict):  return json.dumps(res).encode("utf-8")
        except Exception as e:
            logger.error("descargar_bytes(%s...): %s", cid[:10], e)
        return None

    def estado_conexion(self) -> Dict[str, Any]:
        url = self.descubrir_daemon()
        cli_ok = self._cli_disponible()
        return {
            "daemon_url": url, "daemon_activo": bool(url),
            "cli_disponible": cli_ok, "motor_autonomo": True,
            "total_en_manifiesto": len(self.manifest.get("weights", {})),
        }

    def _cli_disponible(self) -> bool:
        """Comprueba si hay CLI disponible (binario local IPFSonl o CLI de sistema)."""
        if _LOCAL_IPFS_BIN.exists():
            try:
                r = subprocess.run([str(_LOCAL_IPFS_BIN), "version"],
                                   capture_output=True, timeout=3, env=_LOCAL_IPFS_ENV)
                if r.returncode == 0:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        for cmd in (["ipfs", "version"], ["kubo", "version"]):
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=3)
                if r.returncode == 0:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return False

    def obtener_ultimo_cid_pesos(self, patron_nombre: Optional[str] = None):
        return self.ultimo_cid(patron=patron_nombre)

    def descargar_y_extraer_pesos(self, cid: str) -> Optional[bytes]:
        return self.descargar_bytes(cid)


_manager_global: Optional[IPFSManager] = None


def get_ipfs_manager(api_url: Optional[str] = None) -> IPFSManager:
    global _manager_global
    if _manager_global is None:
        _manager_global = IPFSManager(api_url=api_url)
    return _manager_global


def cerrar_sesion_y_subir_psnrl(forzar: bool = False) -> Dict[str, Any]:
    """Hook de cierre: sube todos los pesos de PSNRL a IPFS y los borra."""
    return get_ipfs_manager().subir_y_limpiar_psnrl(forzar_borrado_sin_daemon=forzar)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    print("\n=== IPFS MANAGER 2026 - TEST EN VIVO ===\n")
    print(f"  Binario Kubo local : {_LOCAL_IPFS_BIN} | existe={_LOCAL_IPFS_BIN.exists()}")

    try:
        from LC.celebro.CMFG.IPFSonl.ipfs_launcher import arrancar_daemon
        print("  Arrancando daemon IPFS local (Kubo)...")
        daemon_ok = arrancar_daemon(esperar_segundos=20)
        print(f"  Daemon arrancado: {daemon_ok}")
    except Exception as ex:
        print(f"  Launcher no disponible ({ex}). Continuando sin daemon.")

    mgr = get_ipfs_manager()
    estado = mgr.estado_conexion()
    print(f"  Daemon IPFS activo  : {estado['daemon_activo']} ({estado['daemon_url']})")
    print(f"  CLI kubo disponible : {estado['cli_disponible']}")
    print(f"  Motor autonomo CIDv1: {estado['motor_autonomo']}\n")

    psnrl = Path(__file__).parent.parent / "PSNRL"
    archivos_psnrl = list(psnrl.glob("*.*")) if psnrl.exists() else []
    print(f"  Archivos en PSNRL: {len(archivos_psnrl)}")

    if archivos_psnrl:
        print("  Subiendo + limpiando PSNRL hacia IPFS...")
        resumen = cerrar_sesion_y_subir_psnrl()
        print(f"  Procesados : {resumen['archivos_procesados']}")
        print(f"  CIDs subidos: {len(resumen['cids'])}")
        print(f"  Borrados   : {len(resumen['borrados'])}")
        print(f"  Errores    : {len(resumen['errores'])}")
        for cid in resumen["cids"][:5]:
            print(f"    {cid}")
    else:
        print("  PSNRL vacia. Demo con payload ficticio (subida CLI real si daemon activo)...")
        payload = b'{"demo":"pesos_celebro_2026","neuronas":50}' * 100
        res = mgr.almacenar_pesos(payload, nombre_modelo="demo_pesos", eliminar_local=False)
        print(f"  Demo CID   : {res['cid']}")
        print(f"  Nodo usado : {res['nodo']} | subida_real={res['subida_real']}")
        if res['subida_real']:
            print(f"  Ver en gateway: https://ipfs.io/ipfs/{res['cid']}")
        rec = mgr.descargar_bytes(res["cid"])
        print(f"  Bytes rec  : {len(rec) if rec else 0}")

    print(f"\n  Total en manifiesto: {mgr.manifest.get('total_stored', 0)} entradas")
    print("\n=== TEST FINALIZADO ===\n")