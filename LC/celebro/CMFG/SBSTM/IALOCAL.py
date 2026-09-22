"""
IALOCAL.py - Inferencia Local: Descarga y Gestion de Modelos de IA desde HuggingFace
(Arquitectura WoldVirtualP2P3D 2026)
=================================================================================
1. PERFILADO DE HARDWARE: RAM, VRAM, CPU, disco. Registro en perfil_hardware.json.
2. DESCARGA INTELLIGENTE desde HuggingFace: solo modelos que quepan en el PC.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

__version__: Final[str] = "2026.3.1"
__subsystem__: Final[str] = "LucIA-IALOCAL-HuggingFace"
__author__: Final[str] = "Equipo WoldVirtualP2P3D"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.SBSTM.IALOCAL")

PACKAGE_ROOT: Final[Path] = Path(__file__).parent.resolve()
CMFG_DIR: Final[Path] = PACKAGE_ROOT.parent.resolve()
CELEBRO_DIR: Final[Path] = CMFG_DIR.parent.resolve()
LC_DIR: Final[Path] = CELEBRO_DIR.parent.resolve()
ROOT_DIR: Final[Path] = LC_DIR.parent.resolve()
MODELOS_DIR: Final[Path] = LC_DIR / "modelosIAlocal"
REGISTRO_HW: Final[Path] = MODELOS_DIR / "perfil_hardware.json"
REGISTRO_DESCARGAS: Final[Path] = MODELOS_DIR / "mdstm_registro.json"
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
HF_USER_AGENT: str = "LucIA-IALOCAL-2026"
LOCK_DESCARGA: Final[threading.Lock] = threading.Lock()

ANSI_RESET: Final[str] = "\033[0m"
ANSI_BOLD: Final[str] = "\033[1m"
ANSI_CYAN: Final[str] = "\033[96m"
ANSI_GREEN: Final[str] = "\033[92m"
ANSI_YELLOW: Final[str] = "\033[93m"
ANSI_RED: Final[str] = "\033[91m"
ANSI_MAGENTA: Final[str] = "\033[95m"
ANSI_WHITE: Final[str] = "\033[97m"
ANSI_DIM: Final[str] = "\033[2m"

CATALOGO_HF_LIGEROS: Final[List[Dict[str, Any]]] = [
    {
        "id": "qwen2.5:0.5b", "nombre": "Qwen2.5 0.5B Instruct",
        "ram_min_gb": 2.0, "vram_min_gb": 0.0, "tamano_gb": 0.4,
        "categoria": "ultraligero",
        "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "hf_archivo": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "descripcion": "Respuestas rapidas multilingues.",
    },
    {
        "id": "qwen2.5:1.5b", "nombre": "Qwen2.5 1.5B Instruct",
        "ram_min_gb": 4.0, "vram_min_gb": 0.0, "tamano_gb": 1.0,
        "categoria": "general",
        "hf_repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "hf_archivo": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "descripcion": "Equilibrio calidad/velocidad.",
    },
    {
        "id": "gemma2:2b", "nombre": "Gemma2 2B Instruct",
        "ram_min_gb": 6.0, "vram_min_gb": 2.0, "tamano_gb": 1.6,
        "categoria": "instruccion",
        "hf_repo": "bartowski/gemma-2-2b-it-GGUF",
        "hf_archivo": "gemma-2-2b-it-Q4_K_M.gguf",
        "hf_url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        "descripcion": "Google eficiente para instrucciones.",
    },
    {
        "id": "llama3.2:1b", "nombre": "Llama 3.2 1B Instruct",
        "ram_min_gb": 4.0, "vram_min_gb": 0.0, "tamano_gb": 1.3,
        "categoria": "ligero",
        "hf_repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "hf_archivo": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "hf_url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "descripcion": "Instrucciones Meta optimizadas para CPU.",
    },
    {
        "id": "qwen2.5:3b", "nombre": "Qwen2.5 3B Instruct",
        "ram_min_gb": 8.0, "vram_min_gb": 3.0, "tamano_gb": 2.0,
        "categoria": "general",
        "hf_repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "hf_archivo": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "hf_url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "descripcion": "Buen programador local con 3B params.",
    },
]


def _asegurar_directorio() -> Path:
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELOS_DIR


def _leer_memoria_ram_gb() -> Tuple[float, float]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.total / 1e9, 2), round(vm.available / 1e9, 2)
    except Exception:
        pass
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["wmic", "computersystem", "get", "totalphysicalmemory"],
                                  capture_output=True, text=True, timeout=5)
            nums = "".join(c for c in out.stdout if c.isdigit() or c == " ").split()
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
        import torch
        if torch.cuda.is_available():
            idx = 0
            total = torch.cuda.get_device_properties(idx).total_memory / 1e9
            return round(total, 2), str(torch.cuda.get_device_name(idx))
    except Exception:
        pass
    return 0.0, "sin GPU dedicada (CPU)"


def _leer_cpu_info() -> Dict[str, Any]:
    return {"sistema": platform.system(), "version": platform.version(),
            "maquina": platform.machine(), "procesador": platform.processor() or "desconocido",
            "nucleos": os.cpu_count() or 4}


def _leer_disco_gb() -> float:
    try:
        return round(shutil.disk_usage(str(MODELOS_DIR)).free / 1e9, 2)
    except Exception:
        return 0.0


def perfilar_hardware_ialocal(guardar: bool = True) -> Dict[str, Any]:
    ram_total, ram_libre = _leer_memoria_ram_gb()
    vram_total, gpu_nombre = _leer_vram_gb()
    perfil = {"timestamp": time.time(), "cpu": _leer_cpu_info(),
              "ram_total_gb": ram_total, "ram_libre_gb": ram_libre,
              "vram_total_gb": vram_total, "gpu": gpu_nombre,
              "disco_libre_gb": _leer_disco_gb(), "ollama_host": OLLAMA_HOST,
              "dsialclgrg_version": __version__, "modulo": __name__}
    if guardar:
        try:
            REGISTRO_HW.write_text(json.dumps(perfil, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        except Exception as exc:
            logger.warning("No se guardo perfil HW: %s", exc)
    return perfil


def cargar_perfil_hardware() -> Dict[str, Any]:
    if REGISTRO_HW.exists():
        try:
            return json.loads(REGISTRO_HW.read_text(encoding="utf-8"))
        except Exception:
            pass
    return perfilar_hardware_ialocal(guardar=True)


def verificar_capacidades(perfil: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if perfil is None:
        perfil = cargar_perfil_hardware()
    ram = float(perfil.get("ram_total_gb", 8.0))
    vram = float(perfil.get("vram_total_gb", 0.0))
    disco = float(perfil.get("disco_libre_gb", 10.0))
    cpu = perfil.get("cpu", {})
    capacidades: List[str] = []
    if ram >= 8.0: capacidades.append("ram_suficiente")
    if ram >= 16.0: capacidades.append("ram_alta")
    if vram >= 4.0: capacidades.append("vram_disponible")
    if vram >= 8.0: capacidades.append("vram_alta")
    if disco >= 10.0: capacidades.append("disco_suficiente")
    if "AMD64" in cpu.get("maquina", "") or "x86_64" in cpu.get("maquina", ""):
        capacidades.append("arquitectura_x86")
    nivel = sum(1 for c in ("ram_suficiente", "vram_disponible", "disco_suficiente",
                            "ram_alta", "vram_alta") if c in capacidades)
    clasificacion = "alta" if nivel >= 4 else ("media" if nivel >= 2 else "baja")
    return {
        "capacidades": capacidades, "nivel": nivel, "clasificacion": clasificacion,
        "ram_total_gb": ram, "vram_total_gb": vram, "disco_libre_gb": disco,
        "modelos_compatibles": [m["id"] for m in CATALOGO_HF_LIGEROS
                                if m["ram_min_gb"] <= ram and m["vram_min_gb"] <= vram + 0.01
                                and m["tamano_gb"] <= disco],
    }


def _mostrar_banner_perfil(perfil: Dict[str, Any], cap: Dict[str, Any]) -> None:
    cpu = perfil.get("cpu", {})
    print(f"\n{ANSI_BOLD}{ANSI_CYAN}{'=' * 70}{ANSI_RESET}")
    print(f"{ANSI_BOLD}{ANSI_WHITE}  PERFIL DE HARDWARE DETECTADO - IALOCAL{ANSI_RESET}")
    print(f"{ANSI_CYAN}{'=' * 70}{ANSI_RESET}")
    print(f"  {ANSI_GREEN}Procesador :{ANSI_RESET} {cpu.get('procesador', '?')}")
    print(f"  {ANSI_GREEN}Nucleos    :{ANSI_RESET} {cpu.get('nucleos', '?')}")
    print(f"  {ANSI_GREEN}Arquitectura:{ANSI_RESET} {cpu.get('maquina', '?')}")
    print(f"  {ANSI_YELLOW}RAM Total  :{ANSI_RESET} {perfil.get('ram_total_gb')} GB")
    print(f"  {ANSI_YELLOW}RAM Libre  :{ANSI_RESET} {perfil.get('ram_libre_gb')} GB")
    print(f"  {ANSI_MAGENTA}VRAM Total :{ANSI_RESET} {perfil.get('vram_total_gb')} GB")
    print(f"  {ANSI_MAGENTA}GPU        :{ANSI_RESET} {perfil.get('gpu')}")
    print(f"  {ANSI_CYAN}Disco Libre:{ANSI_RESET} {perfil.get('disco_libre_gb')} GB")
    print(f"  {ANSI_DIM}Clasificacion: {cap.get('clasificacion', '?')} "
          f"({cap.get('nivel', 0)}/5){ANSI_RESET}")
    print(f"{ANSI_CYAN}{'=' * 70}{ANSI_RESET}\n")


def _descargar_hf(url: str, destino: Path, timeout: float = 300.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": HF_USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(destino, "wb") as fh:
            tam_total = resp.length if hasattr(resp, 'length') and resp.length else None
            if tam_total:
                print(f"    {ANSI_YELLOW}Tamanio: {tam_total / 1e6:.1f} MB{ANSI_RESET}")
            bloques = 0
            for bloque in resp:
                fh.write(bloque)
                bloques += 1
                if bloques % 100 == 0 and tam_total:
                    progreso = fh.tell() / tam_total * 100
                    sys.stdout.write(f"\r    {ANSI_CYAN}Descarga: {progreso:.1f}%{ANSI_RESET}")
                    sys.stdout.flush()
            if tam_total:
                sys.stdout.write("\n"); sys.stdout.flush()
        ok = destino.exists() and destino.stat().st_size > 1024
        if ok:
            print(f"    {ANSI_GREEN}Archivo OK: {destino.name} ({destino.stat().st_size // 1024} KB){ANSI_RESET}")
        return ok
    except Exception as exc:
        logger.warning("Descarga HF fallida %s: %s", url, exc)
        try:
            if destino.exists(): destino.unlink()
        except Exception: pass
        print(f"    {ANSI_RED}Error: {exc}{ANSI_RESET}")
        return False


def _registrar_descarga(modelo_id: str, archivo: str, tamano_gb: float,
                        origen: str, exito: bool) -> None:
    try:
        reg = {}
        if REGISTRO_DESCARGAS.exists():
            raw = REGISTRO_DESCARGAS.read_text(encoding="utf-8")
            reg = json.loads(raw) if raw.strip() else {}
        reg.setdefault("descargas", {})
        reg["descargas"][modelo_id] = {
            "origen": "huggingface", "archivo": archivo,
            "tamano_gb": tamano_gb, "exito": exito, "origen_hf": origen,
            "fecha": time.time(), "modulo": __name__,
        }
        REGISTRO_DESCARGAS.write_text(json.dumps(reg, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    except Exception as exc:
        logger.warning("Registro descarga fallido: %s", exc)


def listar_modelos_hf() -> List[Dict[str, Any]]:
    return list(CATALOGO_HF_LIGEROS)


def recomendar_modelos_hf(perfil: Optional[Dict[str, Any]] = None,
                          limite: Optional[int] = None) -> List[Dict[str, Any]]:
    if perfil is None: perfil = cargar_perfil_hardware()
    ram = float(perfil.get("ram_total_gb", 8.0))
    vram = float(perfil.get("vram_total_gb", 0.0))
    disco = float(perfil.get("disco_libre_gb", 10.0))
    aptos = [m for m in CATALOGO_HF_LIGEROS
             if m["ram_min_gb"] <= ram and m["vram_min_gb"] <= vram + 0.01
             and m["tamano_gb"] <= disco]
    aptos.sort(key=lambda m: m["tamano_gb"])
    if limite and limite > 0: aptos = aptos[:limite]
    return aptos


def mostrar_modelos_compatibles(perfil: Optional[Dict[str, Any]] = None,
                                limite: Optional[int] = None) -> List[Dict[str, Any]]:
    compat = recomendar_modelos_hf(perfil=perfil, limite=limite)
    print(f"\n{ANSI_BOLD}{ANSI_GREEN}  MODELOS HUGGINGFACE COMPATIBLES CON TU PC:{ANSI_RESET}")
    print(f"  {ANSI_DIM}Filtrados por RAM/VRAM/disco disponible{ANSI_RESET}")
    print(f"  {'-' * 60}")
    for i, m in enumerate(compat, 1):
        etiq = (f"{ANSI_GREEN}[LIGERO]{ANSI_RESET}" if m["tamano_gb"] < 2.0
                else f"{ANSI_YELLOW}[MEDIO]{ANSI_RESET}" if m["tamano_gb"] < 3.0
                else f"{ANSI_MAGENTA}[PESADO]{ANSI_RESET}")
        print(f"  {ANSI_WHITE}{i}. {m['id']}{ANSI_RESET} {etiq}")
        print(f"      {m['nombre']} | {m['tamano_gb']} GB | {m['descripcion']}")
    if not compat:
        print(f"  {ANSI_RED}Ningun modelo cabe en tu configuracion actual.{ANSI_RESET}")
    print(f"  {'-' * 60}\n")
    return compat


def descargar_modelo_hf(modelo_id: str, limite_ram_gb: Optional[float] = None,
                        limite_vram_gb: Optional[float] = None) -> Dict[str, Any]:
    modelo = next((m for m in CATALOGO_HF_LIGEROS if m["id"] == modelo_id), None)
    if modelo is None:
        return {"exito": False, "mensaje": f"Modelo desconocido: {modelo_id}"}
    perfil = cargar_perfil_hardware()
    if limite_ram_gb is None: limite_ram_gb = float(perfil.get("ram_total_gb", 8.0))
    if limite_vram_gb is None: limite_vram_gb = float(perfil.get("vram_total_gb", 0.0))
    if modelo["ram_min_gb"] > limite_ram_gb or modelo["vram_min_gb"] > limite_vram_gb:
        return {"exito": False, "mensaje":
                f"Modelo {modelo_id} requiere {modelo['ram_min_gb']}GB RAM + "
                f"{modelo['vram_min_gb']}GB VRAM. Tu PC: "
                f"{limite_ram_gb}GB RAM + {limite_vram_gb}GB VRAM."}
    _asegurar_directorio()
    archivo = f"{modelo['id'].replace(':', '_')}.gguf"
    destino = MODELOS_DIR / archivo
    if destino.exists() and destino.stat().st_size > 1024:
        print(f"  {ANSI_GREEN}[YA EXISTE] {archivo}{ANSI_RESET}")
        _registrar_descarga(modelo_id, archivo, modelo["tamano_gb"], modelo["hf_url"], True)
        return {"exito": True, "modelo": modelo_id, "archivo": archivo,
                "mensaje": "Ya descargado previamente", "via": "cache"}
    print(f"\n  {ANSI_BOLD}{ANSI_CYAN}  Descargando de HuggingFace:{ANSI_RESET}")
    print(f"  Modelo : {ANSI_WHITE}{modelo['nombre']}{ANSI_RESET}")
    print(f"  Repo   : {ANSI_DIM}{modelo['hf_repo']}{ANSI_RESET}")
    print(f"  Tamano : {ANSI_YELLOW}{modelo['tamano_gb']} GB{ANSI_RESET}")
    t0 = time.time()
    ok = _descargar_hf(str(modelo["hf_url"]), destino, timeout=600.0)
    segundos = round(time.time() - t0, 1)
    if ok:
        _registrar_descarga(modelo_id, archivo, modelo["tamano_gb"], modelo["hf_url"], True)
        print(f"  {ANSI_GREEN}Completada en {segundos}s{ANSI_RESET}")
    else:
        _registrar_descarga(modelo_id, archivo, modelo["tamano_gb"], modelo["hf_url"], False)
        print(f"  {ANSI_RED}Fallo despues de {segundos}s{ANSI_RESET}")
    return {"exito": ok, "modelo": modelo_id, "archivo": archivo,
            "segundos": segundos, "mensaje": "Descargado" if ok else "Fallo"}


def recomendar_y_descargar(limite: int = 2, mostrar_banner: bool = True) -> List[Dict[str, Any]]:
    perfil = perfilar_hardware_ialocal(guardar=True)
    cap = verificar_capacidades(perfil=perfil)
    if mostrar_banner: _mostrar_banner_perfil(perfil, cap)
    print(f"{ANSI_BOLD}{ANSI_CYAN}  DESCARGA INTELLIGENTE DESDE HUGGINGFACE{ANSI_RESET}")
    print(f"  {ANSI_DIM}Solo modelos que caben en tu PC:{ANSI_RESET}")
    print(f"  RAM: {ANSI_YELLOW}{perfil['ram_total_gb']}GB{ANSI_RESET} | "
          f"VRAM: {ANSI_MAGENTA}{perfil['vram_total_gb']}GB{ANSI_RESET} | "
          f"Disco: {ANSI_CYAN}{perfil['disco_libre_gb']}GB{ANSI_RESET}\n")
    compat = recomendar_modelos_hf(perfil=perfil, limite=limite)
    if not compat:
        print(f"  {ANSI_RED}No hay modelos compatibles.{ANSI_RESET}")
        print(f"  {ANSI_DIM}Aumenta RAM/VRAM o libera disco.{ANSI_RESET}\n")
        return []
    resultados: List[Dict[str, Any]] = []
    total_req = sum(m["tamano_gb"] for m in compat)
    print(f"  {ANSI_WHITE}Modelos: {len(compat)} (total ~{total_req:.1f} GB){ANSI_RESET}\n")
    for i, m in enumerate(compat, 1):
        print(f"  {ANSI_DIM}[{i}/{len(compat)}] {m['id']} "
              f"({m['tamano_gb']} GB - {m['categoria']}){ANSI_RESET}")
        res = descargar_modelo_hf(m["id"])
        resultados.append(res)
        print()
    exitosos = sum(1 for r in resultados if r.get("exito"))
    print(f"{ANSI_BOLD}{ANSI_GREEN}  RESUMEN DE DESCARGAS:{ANSI_RESET}")
    print(f"  {ANSI_GREEN}Exitosos:{ANSI_RESET} {exitosos}/{len(compat)}")
    print(f"  {ANSI_CYAN}Carpeta  :{ANSI_RESET} {MODELOS_DIR}")
    print(f"  {ANSI_DIM}Registro : {REGISTRO_DESCARGAS}{ANSI_RESET}")
    for r in resultados:
        estado = f"{ANSI_GREEN}OK{ANSI_RESET}" if r.get("exito") else f"{ANSI_RED}FAIL{ANSI_RESET}"
        print(f"    [{estado}] {r.get('modelo', '?')} - {r.get('mensaje', '?')}")
    print(f"{ANSI_CYAN}{'=' * 70}{ANSI_RESET}\n")
    return resultados


def obtener_registro_descargas() -> Dict[str, Any]:
    if REGISTRO_DESCARGAS.exists():
        try:
            return json.loads(REGISTRO_DESCARGAS.read_text(encoding="utf-8"))
        except Exception: pass
    return {"descargas": {}}


def obtener_modelos_descargados() -> List[Dict[str, Any]]:
    reg = obtener_registro_descargas()
    out = []
    for mid, meta in reg.get("descargas", {}).items():
        if meta.get("exito", False):
            ruta = MODELOS_DIR / meta.get("archivo", "")
            out.append({"id": mid, "archivo": meta.get("archivo", ""),
                        "tamano_gb": meta.get("tamano_gb", 0.0),
                        "descargado": ruta.exists() and ruta.stat().st_size > 1024,
                        "fecha": meta.get("fecha", 0.0), "origen": "huggingface"})
    return out


def verificar_integridad_archivos() -> Dict[str, Any]:
    modelos = obtener_modelos_descargados()
    v = sum(1 for m in modelos if (MODELOS_DIR / m.get("archivo", "")).exists() and (MODELOS_DIR / m.get("archivo", "")).stat().st_size > 1024)
    return {"total_modelos": len(modelos), "verificados": v,
            "fallos": len(modelos) - v,
            "integridad": "COMPLETA" if len(modelos) - v == 0 else "DAÑADA"}


def info_completa() -> Dict[str, Any]:
    perfil = cargar_perfil_hardware()
    cap = verificar_capacidades(perfil=perfil)
    return {"perfil_hardware": perfil, "capacidades": cap,
            "modelos_compatibles": cap["modelos_compatibles"],
            "modelos_descargados": obtener_modelos_descargados(),
            "registro_descargas": obtener_registro_descargas(),
            "integridad": verificar_integridad_archivos()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print(f"\n{ANSI_BOLD}{ANSI_CYAN}{'=' * 60}{ANSI_RESET}")
    print(f"{ANSI_BOLD}{ANSI_WHITE}  IALOCAL v{__version__}{ANSI_RESET}")
    print(f"{ANSI_CYAN}  Descarga Inteligente desde HuggingFace{ANSI_RESET}")
    print(f"{ANSI_CYAN}{'=' * 60}{ANSI_RESET}")
    res = recomendar_y_descargar(limite=2)
    print(f"  Resultado final: {len(res)} descargas procesadas.\n")
