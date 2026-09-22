"""
DSIALCLGRG.py - Descargador e Inferencia Local de IA Ligera para LucIA (2026)
==============================================================================
Registra la capacidad de VRAM/RAM del PC, descarga modelos de IA ligeros
(quantizados GGUF / Ollama) en LC/modelosIAlocal y ofrece fallback local
cuando OpenRouter (IAFREE) falla o se satura.

Flujo:
  1. perfilar_hardware() -> RAM, VRAM, CPU, disco libre.
  2. recomendar_modelos() -> filtra catalogo segun perfil.
  3. descargar_modelo() / descargar_todos_recomendados() -> LC/modelosIAlocal.
  4. consultar_lucia_local() -> usado por LucIA si OpenRouter falla.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

__version__: Final[str] = "2026.3.1"
__subsystem__: Final[str] = "LucIA-DSIALCLGRG-LocalAI"
__author__: Final[str] = "Equipo WoldVirtualP2P3D"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.SBSTM.DSIALCLGRG")

PACKAGE_ROOT: Final[Path] = Path(__file__).parent.resolve()
CMFG_DIR: Final[Path] = PACKAGE_ROOT.parent.resolve()
CELEBRO_DIR: Final[Path] = CMFG_DIR.parent.resolve()
LC_DIR: Final[Path] = CELEBRO_DIR.parent.resolve()
ROOT_DIR: Final[Path] = LC_DIR.parent.resolve()
MODELOS_DIR: Final[Path] = LC_DIR / "modelosIAlocal"
REGISTRO_HW: Final[Path] = MODELOS_DIR / "perfil_hardware.json"
REGISTRO_MODELOS: Final[Path] = MODELOS_DIR / "modelos_locales.json"
OLLAMA_HOST: Final[str] = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

CATALOGO_LIGEROS: Final[List[Dict[str, Any]]] = [
    {"id": "tinyllama:1.1b", "nombre": "TinyLlama 1.1B Chat", "ram_min_gb": 2.0,
     "vram_min_gb": 0.0, "tamano_gb": 0.7, "ollama_tag": "tinyllama",
     "hf_url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
     "categoria": "ultraligero", "descripcion": "1.1B conversacional, corre en cualquier CPU."},
    {"id": "qwen2.5:0.5b", "nombre": "Qwen2.5 0.5B Instruct", "ram_min_gb": 2.0,
     "vram_min_gb": 0.0, "tamano_gb": 0.4, "ollama_tag": "qwen2.5:0.5b",
     "hf_url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
     "categoria": "ultraligero", "descripcion": "Respuestas rapidas multilingues con 0.5B params."},
    {"id": "qwen2.5:1.5b", "nombre": "Qwen2.5 1.5B Instruct", "ram_min_gb": 4.0,
     "vram_min_gb": 0.0, "tamano_gb": 1.0, "ollama_tag": "qwen2.5:1.5b",
     "hf_url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
     "categoria": "ligero", "descripcion": "Equilibrio calidad/velocidad para PCs modestos."},
    {"id": "llama3.2:1b", "nombre": "Llama 3.2 1B Instruct", "ram_min_gb": 4.0,
     "vram_min_gb": 0.0, "tamano_gb": 1.3, "ollama_tag": "llama3.2:1b",
     "hf_url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
     "categoria": "ligero", "descripcion": "Instrucciones Meta optimizadas para CPU."},
    {"id": "phi3:mini", "nombre": "Phi-3 Mini 3.8B", "ram_min_gb": 6.0,
     "vram_min_gb": 2.0, "tamano_gb": 2.3, "ollama_tag": "phi3:mini",
     "hf_url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
     "categoria": "razonamiento", "descripcion": "Razonamiento compacto de Microsoft."},
    {"id": "gemma2:2b", "nombre": "Gemma2 2B Instruct", "ram_min_gb": 6.0,
     "vram_min_gb": 2.0, "tamano_gb": 1.6, "ollama_tag": "gemma2:2b",
     "hf_url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
     "categoria": "instruccion", "descripcion": "Google eficiente para instrucciones."},
    {"id": "qwen2.5:3b", "nombre": "Qwen2.5 3B Instruct", "ram_min_gb": 8.0,
     "vram_min_gb": 3.0, "tamano_gb": 2.0, "ollama_tag": "qwen2.5:3b",
     "hf_url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
     "categoria": "codigo", "descripcion": "Buen programador local con 3B params."},
    {"id": "llama3.2:3b", "nombre": "Llama 3.2 3B Instruct", "ram_min_gb": 8.0,
     "vram_min_gb": 3.0, "tamano_gb": 2.0, "ollama_tag": "llama3.2:3b",
     "hf_url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
     "categoria": "general", "descripcion": "Uso general solido si hay 8GB RAM."},
]

_LOCK: Final[threading.Lock] = threading.Lock()


def _asegurar_directorio() -> Path:
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELOS_DIR


def _leer_memoria_ram_gb() -> Tuple[float, float]:
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return round(vm.total / 1e9, 2), round(vm.available / 1e9, 2)
    except Exception:
        pass
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["wmic", "computersystem", "get", "totalphysicalmemory"],
                                 capture_output=True, text=True, timeout=5)
            nums = "".join(c if c.isdigit() else " " for c in out.stdout).split()
            if nums:
                total = int(nums[0]) / 1e9
                return round(total, 2), round(total * 0.5, 2)
    except Exception:
        pass
    return 8.0, 4.0


def _leer_vram_gb() -> Tuple[float, str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,name",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            parte = out.stdout.strip().splitlines()[0].split(",")
            vram = float(parte[0].strip()) / 1024.0
            nombre = parte[1].strip() if len(parte) > 1 else "NVIDIA GPU"
            return round(vram, 2), nombre
    except Exception:
        pass
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            idx = 0
            total = torch.cuda.get_device_properties(idx).total_memory / 1e9
            return round(total, 2), str(torch.cuda.get_device_name(idx))
    except Exception:
        pass
    return 1.5, "GPU integrada (configurado)"


def _leer_cpu_info() -> Dict[str, Any]:
    return {"sistema": platform.system(), "version": platform.version(),
            "maquina": platform.machine(), "procesador": platform.processor() or "desconocido",
            "nucleos": os.cpu_count() or 4}


def _leer_disco_gb() -> float:
    try:
        return round(shutil.disk_usage(str(MODELOS_DIR)).free / 1e9, 2)
    except Exception:
        return 0.0


def perfilar_hardware(guardar: bool = True) -> Dict[str, Any]:
    """Detecta RAM/VRAM/CPU/disco y guarda el perfil en perfil_hardware.json."""
    _asegurar_directorio()
    ram_total, ram_libre = _leer_memoria_ram_gb()
    vram_total, gpu_nombre = _leer_vram_gb()
    perfil = {"timestamp": time.time(), "cpu": _leer_cpu_info(),
              "ram_total_gb": ram_total, "ram_libre_gb": ram_libre,
              "vram_total_gb": vram_total, "gpu": gpu_nombre,
              "disco_libre_gb": _leer_disco_gb(), "ollama_host": OLLAMA_HOST,
              "dsialclgrg_version": __version__}
    if guardar:
        try:
            REGISTRO_HW.write_text(json.dumps(perfil, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("No se pudo guardar perfil HW: %s", exc)
    logger.info("Perfil HW: RAM %.1fGB | VRAM %.1fGB (%s)", ram_total, vram_total, gpu_nombre)
    return perfil


def cargar_perfil_hardware() -> Dict[str, Any]:
    """Carga el perfil guardado o genera uno nuevo si no existe."""
    if REGISTRO_HW.exists():
        try:
            return json.loads(REGISTRO_HW.read_text(encoding="utf-8"))
        except Exception:
            pass
    return perfilar_hardware(guardar=True)


def recomendar_modelos(perfil: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Filtra CATALOGO_LIGEROS segun RAM/VRAM/disco disponibles."""
    perfil = perfil or cargar_perfil_hardware()
    ram = float(perfil.get("ram_total_gb", 8.0))
    vram = float(perfil.get("vram_total_gb", 0.0))
    disco = float(perfil.get("disco_libre_gb", 10.0))
    aptos = [m for m in CATALOGO_LIGEROS
             if m["ram_min_gb"] <= ram and m["vram_min_gb"] <= vram + 0.01
             and m["tamano_gb"] <= disco]
    if not aptos:
        aptos = [CATALOGO_LIGEROS[0]]
    return sorted(aptos, key=lambda m: m["tamano_gb"])


def _leer_registro_modelos() -> Dict[str, Any]:
    if REGISTRO_MODELOS.exists():
        try:
            data = json.loads(REGISTRO_MODELOS.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {"modelos": {}}


def _guardar_registro_modelos(data: Dict[str, Any]) -> None:
    _asegurar_directorio()
    REGISTRO_MODELOS.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def listar_modelos_locales() -> List[Dict[str, Any]]:
    """Lista modelos fisicamente presentes en LC/modelosIAlocal + registro."""
    _asegurar_directorio()
    reg = _leer_registro_modelos().get("modelos", {})
    out: List[Dict[str, Any]] = []
    for mid, meta in reg.items():
        ruta = MODELOS_DIR / meta.get("archivo", "")
        out.append({"id": mid, "archivo": meta.get("archivo", ""),
                    "tamano_gb": meta.get("tamano_gb", 0.0),
                    "descargado": ruta.exists(), "origen": meta.get("origen", "?"),
                    "fecha": meta.get("fecha", 0.0)})
    for f in sorted(MODELOS_DIR.glob("*.gguf")):
        if not any(o["archivo"] == f.name for o in out):
            out.append({"id": f.stem, "archivo": f.name,
                        "tamano_gb": round(f.stat().st_size / 1e9, 2),
                        "descargado": True, "origen": "manual", "fecha": f.stat().st_mtime})
    return out


def _descargar_url(url: str, destino: Path, timeout: float = 60.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LucIA-DSIALCLGRG/2026"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(destino, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        return destino.exists() and destino.stat().st_size > 1024
    except Exception as exc:
        logger.warning("Descarga GGUF fallida %s: %s", url, exc)
        try:
            if destino.exists():
                destino.unlink()
        except Exception:
            pass
        return False


def _ollama_disponible() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST.rstrip('/')}/api/tags",
                                     headers={"User-Agent": "LucIA-DSIALCLGRG/2026"})
        with urllib.request.urlopen(req, timeout=3.0):
            return True
    except Exception:
        return False


def _ollama_pull(tag: str) -> bool:
    if shutil.which("ollama") is None and not _ollama_disponible():
        return False
    try:
        if shutil.which("ollama"):
            r = subprocess.run(["ollama", "pull", tag], capture_output=True,
                               text=True, timeout=600)
            return r.returncode == 0
        payload = json.dumps({"name": tag}).encode("utf-8")
        req = urllib.request.Request(f"{OLLAMA_HOST.rstrip('/')}/api/pull", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=600):
            return True
    except Exception as exc:
        logger.warning("Ollama pull %s fallo: %s", tag, exc)
        return False


def descargar_modelo(modelo_id: str, via: str = "auto") -> Dict[str, Any]:
    """Descarga un modelo ligero en LC/modelosIAlocal. via: auto|ollama|gguf."""
    _asegurar_directorio()
    meta = next((m for m in CATALOGO_LIGEROS if m["id"] == modelo_id), None)
    if meta is None:
        return {"exito": False, "mensaje": f"Modelo desconocido: {modelo_id}"}
    t0 = time.time()
    ok, metodo = False, ""
    if via in ("auto", "ollama"):
        if _ollama_pull(meta["ollama_tag"]):
            ok, metodo = True, "ollama"
    if not ok and via in ("auto", "gguf"):
        destino = MODELOS_DIR / f"{meta['ollama_tag'].replace(':', '_')}.gguf"
        if destino.exists() and destino.stat().st_size > 1024:
            ok, metodo = True, "gguf-cache"
        elif _descargar_url(str(meta["hf_url"]), destino):
            ok, metodo = True, "gguf"
    with _LOCK:
        reg = _leer_registro_modelos()
        reg.setdefault("modelos", {})[modelo_id] = {
            "archivo": f"{meta['ollama_tag'].replace(':', '_')}.gguf" if metodo.startswith("gguf") else "",
            "tamano_gb": meta["tamano_gb"], "origen": metodo or "fallido",
            "fecha": time.time(), "ollama_tag": meta["ollama_tag"]}
        try:
            _guardar_registro_modelos(reg)
        except Exception:
            pass
    return {"exito": ok, "modelo": modelo_id, "metodo": metodo,
            "segundos": round(time.time() - t0, 1),
            "mensaje": "Listo en modelosIAlocal" if ok else "Fallo descarga (sin red u Ollama off)"}


def descargar_todos_recomendados(limite: int = 2) -> List[Dict[str, Any]]:
    """Perfil HW + descarga los N modelos mas adecuados. Retorna reporte."""
    perfil = perfilar_hardware(guardar=True)
    recs = recomendar_modelos(perfil)[:max(1, limite)]
    return [descargar_modelo(m["id"]) for m in recs]


def _consultar_ollama(prompt: str, tag: str, timeout: float = 60.0) -> Optional[str]:
    try:
        payload = json.dumps({"model": tag, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(f"{OLLAMA_HOST.rstrip('/')}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            txt = str(data.get("response", "")).strip()
            return txt or None
    except Exception as exc:
        logger.debug("Ollama local fallo (%s): %s", tag, exc)
        return None


def _respuesta_reflejo(prompt: str, perfil: Dict[str, Any]) -> str:
    return ("[LucIA-local] OpenRouter no disponible; respondo con nucleo local. "
            f"Tu consulta ('{prompt[:160]}') fue registrada. "
            f"PC: {perfil.get('ram_total_gb', '?')}GB RAM / {perfil.get('vram_total_gb', '?')}GB VRAM. "
            "Instala Ollama y descarga un modelo ligero para respuestas completas.")


def consultar_lucia_local(prompt: str, contexto_neuronal: Optional[Dict[str, Any]] = None,
                          timeout: float = 60.0) -> Tuple[str, str, float]:
    """Fallback local cuando OpenRouter falla. Retorna (texto, modelo_id, latencia_ms)."""
    t0 = time.perf_counter()
    perfil = cargar_perfil_hardware()
    tags: List[str] = []
    for m in listar_modelos_locales():
        reg = _leer_registro_modelos().get("modelos", {})
        for mid, meta in reg.items():
            if meta.get("ollama_tag") and meta.get("origen") == "ollama":
                tags.append(str(meta["ollama_tag"]))
    for rec in recomendar_modelos(perfil):
        if rec["ollama_tag"] not in tags:
            tags.append(rec["ollama_tag"])
    tono = (contexto_neuronal or {}).get("tono_cognitivo", "claro y directo")
    prompt_env = f"Eres LucIA local (offline, tono {tono}). Responde breve y util: {prompt}"
    for tag in tags:
        txt = _consultar_ollama(prompt_env, tag, timeout=timeout)
        if txt:
            return txt, f"local:{tag}", (time.perf_counter() - t0) * 1000.0
    return _respuesta_reflejo(prompt, perfil), "local:reflejo", (time.perf_counter() - t0) * 1000.0


def consultar_con_fallback(prompt: str, contexto_neuronal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Intenta OpenRouter (IAFREE); si falla, usa modelo local. Para LucIA."""
    try:
        from LC.celebro.CMFG.SBSTM.IAFREE import get_cliente_iafree
        cliente = get_cliente_iafree()
        texto, mid, lat = cliente.generar_respuesta(prompt, contexto_neuronal=contexto_neuronal,
                                                    max_reintentos=2, stream_en_vivo=False)
        if mid != "fallback_agotado" and "temporalmente saturados" not in texto:
            return {"texto": texto, "modelo": mid, "latencia_ms": lat, "fuente": "openrouter"}
    except Exception as exc:
        logger.debug("IAFREE no disponible, usando local: %s", exc)
    texto, mid, lat = consultar_lucia_local(prompt, contexto_neuronal)
    return {"texto": texto, "modelo": mid, "latencia_ms": lat, "fuente": "local"}


def estado_dsialclgrg() -> Dict[str, Any]:
    """Diagnostico consolidado: hardware + modelos + Ollama."""
    perfil = cargar_perfil_hardware()
    return {"version": __version__, "perfil": perfil,
            "recomendados": [m["id"] for m in recomendar_modelos(perfil)],
            "locales": listar_modelos_locales(),
            "ollama_online": _ollama_disponible(),
            "carpeta": str(MODELOS_DIR), "total_catalogo": len(CATALOGO_LIGEROS)}


_INSTANCIA: Optional[Dict[str, Any]] = None


def get_gestor_local() -> Dict[str, Any]:
    """Singleton ligero con perfil + recomendados (cache en memoria)."""
    global _INSTANCIA
    with _LOCK:
        if _INSTANCIA is None:
            perfil = cargar_perfil_hardware()
            _INSTANCIA = {"perfil": perfil, "recomendados": recomendar_modelos(perfil)}
        return _INSTANCIA


def verificar_sha256(ruta: Path, esperado: str = "") -> bool:
    """Verifica integridad opcional de un GGUF descargado."""
    try:
        h = hashlib.sha256()
        with open(ruta, "rb") as fh:
            for bloque in iter(lambda: fh.read(1 << 20), b""):
                h.update(bloque)
        return (not esperado) or h.hexdigest().lower() == esperado.lower()
    except Exception:
        return False

if __name__ == "__main__":
    print("=" * 70)
    print(f"  DSIALCLGRG v{__version__} - IA local ligera para LucIA")
    print("=" * 70)
    perfil = perfilar_hardware(guardar=True)
    print(f"  RAM: {perfil['ram_total_gb']}GB | VRAM: {perfil['vram_total_gb']}GB ({perfil['gpu']})")
    print(f"  Carpeta: {MODELOS_DIR}")
    print("  Recomendados:")
    for m in recomendar_modelos(perfil)[:3]:
        print(f"    * {m['id']} | {m['tamano_gb']}GB | {m['descripcion']}")
    print(f"  Ollama online: {_ollama_disponible()}")
    print("  Locales:", len(listar_modelos_locales()))
    print("=" * 70)