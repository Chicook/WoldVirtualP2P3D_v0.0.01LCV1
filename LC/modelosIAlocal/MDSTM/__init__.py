"""
MDSTM - Modulo de Descarga del Sistema de Transduccion de Modelos (LucIA 2026)
==============================================================================
Capacidad interna REAL de LucIA para descargar modelos de IA ligeros de forma
autonoma en LC/modelosIAlocal. Expone la clase GestorDescargaModelos que el
orquestador mainLCSTM importa y usa sin pedir permiso al modelo remoto.

Capacidades:
  - Perfil de hardware (RAM/VRAM/CPU/disco) con persistencia JSON.
  - Catalogo de modelos ligeros (Ollama + GGUF HuggingFace).
  - Descarga autonoma via Ollama (`ollama pull` / /api/pull) o GGUF directo.
  - Interprete de ordenes en lenguaje natural ("descarga el modelo X").
  - Deteccion de intencion de descarga para interceptar antes de OpenRouter.
  - Ejecucion en hilo (no bloquea el turno de dialogo).
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

__version__: Final[str] = "2026.3.1"
__subsystem__: Final[str] = "LucIA-MDSTM-Descargas"
__author__: Final[str] = "Equipo WoldVirtualP2P3D"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.MDSTM")

MDSTM_DIR: Final[Path] = Path(__file__).parent.resolve()
MODELOS_DIR: Final[Path] = MDSTM_DIR.parent.resolve()
PERFIL_HW: Final[Path] = MODELOS_DIR / "perfil_hardware.json"
REGISTRO: Final[Path] = MODELOS_DIR / "mdstm_registro.json"
OLLAMA_HOST: Final[str] = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

CATALOGO: Final[List[Dict[str, Any]]] = [
    {"id": "qwen2.5:0.5b", "tag": "qwen2.5:0.5b", "gb": 0.4, "ram": 2.0, "vram": 0.0,
     "gguf": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
     "desc": "Ultraligero multilingue, ideal sin GPU."},
    {"id": "tinyllama:1.1b", "tag": "tinyllama", "gb": 0.7, "ram": 2.0, "vram": 0.0,
     "gguf": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
     "desc": "Conversacional 1.1B, corre en cualquier CPU."},
    {"id": "qwen2.5:1.5b", "tag": "qwen2.5:1.5b", "gb": 1.0, "ram": 4.0, "vram": 0.0,
     "gguf": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
     "desc": "Equilibrio calidad/velocidad en PCs modestos."},
    {"id": "llama3.2:1b", "tag": "llama3.2:1b", "gb": 1.3, "ram": 4.0, "vram": 0.0,
     "gguf": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
     "desc": "Instrucciones Meta optimizadas para CPU."},
    {"id": "gemma2:2b", "tag": "gemma2:2b", "gb": 1.6, "ram": 6.0, "vram": 2.0,
     "gguf": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
     "desc": "Google eficiente para seguir instrucciones."},
    {"id": "qwen2.5:3b", "tag": "qwen2.5:3b", "gb": 2.0, "ram": 8.0, "vram": 3.0,
     "gguf": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
     "desc": "Buen programador local con 3B parametros."},
    {"id": "llama3.2:3b", "tag": "llama3.2:3b", "gb": 2.0, "ram": 8.0, "vram": 3.0,
     "gguf": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
     "desc": "Uso general solido si hay 8GB RAM."},
    {"id": "phi3:mini", "tag": "phi3:mini", "gb": 2.3, "ram": 6.0, "vram": 2.0,
     "gguf": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
     "desc": "Razonamiento compacto de Microsoft."},
]

PATRON_DESCARGA: Final[re.Pattern] = re.compile(
    r"(descarg\w*|baj\w*|instal\w*|trae\w*|pon\w*)\s+(el\s+)?(modelo\s+)?(.+)?",
    re.IGNORECASE,
)
PALABRAS_MODELO: Final[List[str]] = ["modelo", "modelos", "ia", "qwen", "llama",
                                     "tinyllama", "phi", "gemma", "ollama", "gguf", "local"]

_LOCK: Final[threading.Lock] = threading.Lock()


def _ram_gb() -> Tuple[float, float]:
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return round(vm.total / 1e9, 2), round(vm.available / 1e9, 2)
    except Exception:
        pass
    try:
        out = subprocess.run(["wmic", "computersystem", "get", "totalphysicalmemory"],
                             capture_output=True, text=True, timeout=5)
        nums = "".join(c if c.isdigit() else " " for c in out.stdout).split()
        if nums:
            total = int(nums[0]) / 1e9
            return round(total, 2), round(total * 0.5, 2)
    except Exception:
        pass
    return 8.0, 4.0


def _vram_gb() -> Tuple[float, str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,name",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            parte = out.stdout.strip().splitlines()[0].split(",")
            return round(float(parte[0].strip()) / 1024.0, 2), parte[1].strip() if len(parte) > 1 else "NVIDIA"
    except Exception:
        pass
    return 0.0, "sin GPU dedicada (CPU)"


class GestorDescargaModelos:
    """Clase principal: LucIA descarga modelos ligeros de forma autonoma."""

    def __init__(self) -> None:
        MODELOS_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._hilo_fondo: Optional[threading.Thread] = None
        self.perfil: Dict[str, Any] = self.perfilar_hardware()

    # ── Hardware ──────────────────────────────────────────────
    def perfilar_hardware(self, guardar: bool = True) -> Dict[str, Any]:
        ram_total, ram_libre = _ram_gb()
        vram_total, gpu = _vram_gb()
        try:
            disco = round(shutil.disk_usage(str(MODELOS_DIR)).free / 1e9, 2)
        except Exception:
            disco = 0.0
        perfil = {"timestamp": time.time(), "ram_total_gb": ram_total,
                  "ram_libre_gb": ram_libre, "vram_total_gb": vram_total,
                  "gpu": gpu, "disco_libre_gb": disco,
                  "cpu_nucleos": os.cpu_count() or 4, "sistema": platform.system(),
                  "ollama_host": OLLAMA_HOST, "mdstm_version": __version__}
        if guardar:
            try:
                PERFIL_HW.write_text(json.dumps(perfil, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                logger.warning("MDSTM no pudo guardar perfil: %s", exc)
        self.perfil = perfil
        return perfil

    def capacidad_real(self) -> str:
        """Frase factual que demuestra a LucIA que SI puede descargar."""
        p = self.perfil
        return (f"Puedo descargar modelos YA: dispongo de {p['ram_total_gb']}GB RAM, "
                f"{p['vram_total_gb']}GB VRAM ({p['gpu']}), {p['disco_libre_gb']}GB libres "
                f"en {MODELOS_DIR} y modulo MDSTM v{__version__} operativo.")

    # ── Catalogo ──────────────────────────────────────────────
    def recomendar(self, perfil: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        p = perfil or self.perfil
        ram, vram = float(p.get("ram_total_gb", 8.0)), float(p.get("vram_total_gb", 0.0))
        disco = float(p.get("disco_libre_gb", 10.0))
        aptos = [m for m in CATALOGO if m["ram"] <= ram and m["vram"] <= vram + 0.01 and m["gb"] <= disco]
        return sorted(aptos or [CATALOGO[0]], key=lambda m: m["gb"])

    def buscar(self, texto: str) -> Optional[Dict[str, Any]]:
        t = texto.lower().strip()
        for m in CATALOGO:
            if m["id"] in t or m["tag"] in t:
                return m
        corto = re.sub(r"[^a-z0-9.+ ]", " ", t)
        for m in CATALOGO:
            base = m["tag"].split(":")[0]
            if base in corto:
                return m
        return None

    # ── Registro ──────────────────────────────────────────────
    def _leer_registro(self) -> Dict[str, Any]:
        if REGISTRO.exists():
            try:
                data = json.loads(REGISTRO.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {"descargas": {}}

    def _anotar(self, modelo_id: str, origen: str, exito: bool) -> None:
        with self._lock:
            reg = self._leer_registro()
            reg.setdefault("descargas", {})[modelo_id] = {
                "origen": origen, "exito": exito, "fecha": time.time()}
            try:
                REGISTRO.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

    def listar_descargados(self) -> List[Dict[str, Any]]:
        reg = self._leer_registro().get("descargas", {})
        out: List[Dict[str, Any]] = []
        for mid, meta in reg.items():
            out.append({"id": mid, "origen": meta.get("origen", "?"),
                        "exito": meta.get("exito", False), "fecha": meta.get("fecha", 0.0)})
        for f in sorted(MODELOS_DIR.glob("*.gguf")):
            if not any(o["id"] == f.stem for o in out):
                out.append({"id": f.stem, "origen": "gguf-manual",
                            "exito": True, "fecha": f.stat().st_mtime})
        return out

    # ── Motores de descarga ───────────────────────────────────
    def ollama_online(self) -> bool:
        try:
            req = urllib.request.Request(f"{OLLAMA_HOST.rstrip('/')}/api/tags",
                                         headers={"User-Agent": "LucIA-MDSTM/2026"})
            with urllib.request.urlopen(req, timeout=3.0):
                return True
        except Exception:
            return False

    def _pull_ollama(self, tag: str) -> bool:
        try:
            if shutil.which("ollama"):
                r = subprocess.run(["ollama", "pull", tag], capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=900)
                return r.returncode == 0
            payload = json.dumps({"name": tag}).encode("utf-8")
            req = urllib.request.Request(f"{OLLAMA_HOST.rstrip('/')}/api/pull", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=900):
                return True
        except Exception as exc:
            logger.warning("MDSTM pull %s fallo: %s", tag, exc)
            return False

    def _bajar_gguf(self, url: str, destino: Path) -> bool:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LucIA-MDSTM/2026"})
            with urllib.request.urlopen(req, timeout=120.0) as resp, open(destino, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            return destino.exists() and destino.stat().st_size > 1024
        except Exception as exc:
            logger.warning("MDSTM GGUF fallo: %s", exc)
            try:
                destino.unlink(missing_ok=True)
            except Exception:
                pass
            return False

    def descargar(self, modelo_id: str, via: str = "auto") -> Dict[str, Any]:
        """Descarga REAL del modelo en LC/modelosIAlocal. Retorna reporte factual."""
        meta = next((m for m in CATALOGO if m["id"] == modelo_id), None) or self.buscar(modelo_id)
        if meta is None:
            return {"exito": False, "modelo": modelo_id,
                    "mensaje": f"No conozco '{modelo_id}'. Opciones: {', '.join(m['id'] for m in CATALOGO)}."}
        t0, ok, metodo = time.time(), False, ""
        if via in ("auto", "ollama") and self._pull_ollama(meta["tag"]):
            ok, metodo = True, "ollama"
        if not ok and via in ("auto", "gguf"):
            destino = MODELOS_DIR / f"{meta['tag'].replace(':', '_')}.gguf"
            if destino.exists() and destino.stat().st_size > 1024:
                ok, metodo = True, "gguf-cache"
            elif self._bajar_gguf(str(meta["gguf"]), destino):
                ok, metodo = True, "gguf"
        self._anotar(meta["id"], metodo or "fallido", ok)
        seg = round(time.time() - t0, 1)
        if ok:
            return {"exito": True, "modelo": meta["id"], "metodo": metodo, "segundos": seg,
                    "mensaje": f"He descargado {meta['id']} ({meta['gb']}GB, via {metodo}) en {seg}s. "
                               f"Esta en LC/modelosIAlocal y listo para usar offline."}
        return {"exito": False, "modelo": meta["id"], "metodo": "", "segundos": seg,
                "mensaje": f"No pude descargar {meta['id']}: sin red ni Ollama. "
                           "Instala Ollama (ollama.com) y reintenta."}

    def descargar_recomendados(self, limite: int = 2) -> List[Dict[str, Any]]:
        self.perfilar_hardware()
        return [self.descargar(m["id"]) for m in self.recomendar()[:max(1, limite)]]

    def descargar_en_fondo(self, modelo_id: str = "") -> str:
        """Lanza descarga sin bloquear el turno; LucIA sigue conversando."""
        if self._hilo_fondo and self._hilo_fondo.is_alive():
            return "Ya hay una descarga en curso en segundo plano; espera a que termine."
        objetivo = modelo_id or (self.recomendar()[0]["id"] if self.recomendar() else "")
        if not objetivo:
            return "No hay modelo objetivo para descargar."
        self._hilo_fondo = threading.Thread(target=self.descargar, args=(objetivo,),
                                            daemon=True, name="MDSTM-descarga")
        self._hilo_fondo.start()
        return f"Descargando {objetivo} en segundo plano; te aviso al terminar."

    # ── Lenguaje natural ──────────────────────────────────────
    def es_orden_descarga(self, texto: str) -> bool:
        t = texto.lower()
        return bool(PATRON_DESCARGA.search(texto)) and any(p in t for p in PALABRAS_MODELO)

    def ejecutar_orden(self, texto: str) -> Optional[str]:
        """Si el usuario pide descargar, EJECUTA y devuelve respuesta factual. None si no aplica."""
        if not self.es_orden_descarga(texto):
            return None
        meta = self.buscar(texto)
        if meta is None:
            if re.search(r"recomendad|liger|que\s+quep|autonom", texto, re.IGNORECASE):
                reps = self.descargar_recomendados(limite=1)
                r = reps[0] if reps else {}
                return ("Claro que si: SI puedo descargar modelos. " + str(r.get("mensaje", "sin resultado.")))
            lista = ", ".join(m["id"] for m in self.recomendar())
            return (f"Por supuesto, SI tengo capacidad de descarga ({self.capacidad_real()}). "
                    f"Dime cual de estos: {lista}.")
        rep = self.descargar(meta["id"])
        return ("Claro que si, ya lo estoy haciendo. " if rep["exito"] else "Lo he intentado: ") + rep["mensaje"]

    def estado(self) -> Dict[str, Any]:
        return {"version": __version__, "perfil": self.perfil,
                "capacidad": self.capacidad_real(),
                "recomendados": [m["id"] for m in self.recomendar()],
                "descargados": self.listar_descargados(),
                "ollama_online": self.ollama_online(),
                "carpeta": str(MODELOS_DIR), "catalogo": len(CATALOGO)}

    # ── Mantenimiento ─────────────────────────────────────────
    def verificar_integridad(self) -> List[Dict[str, Any]]:
        """Comprueba que cada GGUF registrado exista y pese mas de 1KB."""
        reporte: List[Dict[str, Any]] = []
        for item in self.listar_descargados():
            arch = MODELOS_DIR / f"{item['id'].replace(':', '_')}.gguf"
            ok = arch.exists() and arch.stat().st_size > 1024
            reporte.append({"id": item["id"], "integro": ok,
                            "bytes": arch.stat().st_size if arch.exists() else 0})
        return reporte

    def eliminar_modelo(self, modelo_id: str) -> Dict[str, Any]:
        """Borra el GGUF local para liberar disco y lo anota en el registro."""
        meta = next((m for m in CATALOGO if m["id"] == modelo_id), None) or self.buscar(modelo_id)
        if meta is None:
            return {"exito": False, "mensaje": f"Modelo desconocido: {modelo_id}."}
        arch = MODELOS_DIR / f"{meta['tag'].replace(':', '_')}.gguf"
        try:
            if arch.exists():
                arch.unlink()
            with self._lock:
                reg = self._leer_registro()
                reg.get("descargas", {}).pop(meta["id"], None)
                REGISTRO.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
            return {"exito": True, "mensaje": f"He eliminado {meta['id']} y liberado {meta['gb']}GB."}
        except Exception as exc:
            return {"exito": False, "mensaje": f"No pude eliminar {meta['id']}: {exc}."}

    def espacio_usado_gb(self) -> float:
        """Suma el peso real de los GGUF presentes en la carpeta."""
        total = 0
        try:
            for f in MODELOS_DIR.glob("*.gguf"):
                total += f.stat().st_size
        except Exception:
            pass
        return round(total / 1e9, 2)

    def informe_para_lucia(self) -> str:
        """Texto factual completo para que LucIA responda con datos reales."""
        est = self.estado()
        perf = est["perfil"]
        lineas = [
            f"Soy LucIA y SI puedo descargar modelos de IA localmente (MDSTM v{__version__}).",
            f"Hardware: {perf['ram_total_gb']}GB RAM, {perf['vram_total_gb']}GB VRAM, "
            f"{perf['disco_libre_gb']}GB libres. Ollama: {'online' if est['ollama_online'] else 'offline'}.",
            f"Recomendados para este PC: {', '.join(est['recomendados'])}.",
            f"Descargados ({self.espacio_usado_gb()}GB): "
            + (", ".join(d["id"] for d in est["descargados"]) or "ninguno aun; pideme 'descarga el modelo X' "
               "y lo descargo de inmediato con ollama pull o GGUF directo."),
        ]
        return " ".join(lineas)

    def ayuda_comandos(self) -> str:
        """Ayuda breve de las ordenes de descarga que LucIA entiende."""
        return ("Puedo descargar modelos YA. Prueba: 'descarga el modelo qwen2.5:0.5b', "
                "'descarga un modelo ligero recomendado', 'que modelos tienes descargados', "
                "'cuanta RAM y VRAM tengo', 'elimina el modelo X'. "
                f"Opciones: {', '.join(m['id'] for m in CATALOGO)}.")


_GESTOR: Optional[GestorDescargaModelos] = None


def get_gestor_mdstm() -> GestorDescargaModelos:
    """Singleton: la instancia de descarga autonoma de LucIA."""
    global _GESTOR
    with _LOCK:
        if _GESTOR is None:
            _GESTOR = GestorDescargaModelos()
        return _GESTOR


def ordenar_descarga_lucia(texto_usuario: str) -> Optional[str]:
    """Atajo: interpreta y ejecuta ordenes de descarga. None si no es orden."""
    return get_gestor_mdstm().ejecutar_orden(texto_usuario)


def capacidad_mdstm() -> str:
    """Frase lista para inyectar en el prompt del sistema de LucIA."""
    return get_gestor_mdstm().capacidad_real()


if __name__ == "__main__":
    g = get_gestor_mdstm()
    print("=" * 70)
    print(f"  MDSTM v{__version__} - Descargas autonomas de LucIA")
    print("=" * 70)
    print(" ", g.capacidad_real())
    print("  Recomendados:", ", ".join(m["id"] for m in g.recomendar()[:3]))
    print("  Ollama:", g.ollama_online(), "| Descargados:", len(g.listar_descargados()))
    print("=" * 70)