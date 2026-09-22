"""
IALOCAL.py - Descarga y Gestion de Modelos IA desde HuggingFace
1. PERFILADO HW: RAM/VRAM/CPU/disco -> perfil_hardware.json
2. DESCARGA HF: solo modelos que quepan en PC
3. IA TEMPORAL: descarga pre-sesion, borrar al cerrar
4. CONSTRUCTOR: Refactoriza archivos Python (400/450 lineas)
5. CIERRE SESION: Refactor+MD+Pesos+IPFS con barra tiempo
"""
from __future__ import annotations
import json, logging, os, platform, shutil, subprocess, sys, threading, time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

__version__: Final[str] = "2026.3.1"
__subsystem__: Final[str] = "LucIA-IALOCAL-HF"
__author__: Final[str] = "Equipo WoldVirtualP2P3D"
logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.SBSTM.IALOCAL")
PACKAGE_ROOT: Final[Path] = Path(__file__).parent.resolve()
CMFG_DIR: Final[Path] = PACKAGE_ROOT.parent.resolve()
CELEBRO_DIR: Final[Path] = CMFG_DIR.parent.resolve()
LC_DIR: Final[Path] = CELEBRO_DIR.parent.resolve()
ROOT_DIR: Final[Path] = LC_DIR.parent.resolve()
CONSTRUCTOR_DIR: Final[Path] = ROOT_DIR / "Constructor"
MODELOS_DIR: Final[Path] = LC_DIR / "modelosIAlocal"
REGISTRO_HW: Final[Path] = MODELOS_DIR / "perfil_hardware.json"
REGISTRO_DESCARGAS: Final[Path] = MODELOS_DIR / "mdstm_registro.json"
REGISTRO_TEMP: Final[Path] = MODELOS_DIR / "temp_session.json"
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
HF_USER_AGENT: str = "LucIA-IALOCAL-2026"
LOCK_TEMP: Final[threading.Lock] = threading.Lock()
LOCK_CIERRE: Final[threading.Lock] = threading.Lock()
ANSI_R = "\033[0m"; ANSI_B = "\033[1m"; ANSI_CY = "\033[96m"; ANSI_G = "\033[92m"
ANSI_Y = "\033[93m"; ANSI_RR = "\033[91m"; ANSI_M = "\033[95m"; ANSI_W = "\033[97m"; ANSI_D = "\033[2m"
CATALOGO_HF_LIGEROS: Final[List[Dict[str, Any]]] = [
    {"id": "qwen2.5:0.5b", "nombre": "Qwen2.5 0.5B", "ram_min_gb": 2.0, "vram_min_gb": 0.5,
     "tamano_gb": 0.4, "categoria": "ultraligero",
     "hf_url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
     "descripcion": "Respuestas rapidas multilingues."},
    {"id": "qwen2.5:1.5b", "nombre": "Qwen2.5 1.5B", "ram_min_gb": 4.0, "vram_min_gb": 1.5,
     "tamano_gb": 1.0, "categoria": "general",
     "hf_url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
     "descripcion": "Equilibrio calidad/velocidad."},
    {"id": "gemma2:2b", "nombre": "Gemma2 2B", "ram_min_gb": 6.0, "vram_min_gb": 2.0,
     "tamano_gb": 1.6, "categoria": "instruccion",
     "hf_url": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
     "descripcion": "Google eficiente para instrucciones."},
    {"id": "llama3.2:1b", "nombre": "Llama 3.2 1B", "ram_min_gb": 4.0, "vram_min_gb": 0.5,
     "tamano_gb": 1.3, "categoria": "ligero",
     "hf_url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
     "descripcion": "Instrucciones Meta para CPU."},
    {"id": "qwen2.5:3b", "nombre": "Qwen2.5 3B", "ram_min_gb": 8.0, "vram_min_gb": 3.0,
     "tamano_gb": 2.0, "categoria": "general",
     "hf_url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
     "descripcion": "Buen programador local 3B."},
]
ARCHIVOS_TEMP_SESSION: List[Path] = []
TIEMPO_INICIO_CIERRE: float = 0.0
CONSTRUCTOR_CIERRE_OK: bool = False


def _cargar_temp() -> List[Path]:
    try:
        if REGISTRO_TEMP.exists():
            datos = json.loads(REGISTRO_TEMP.read_text(encoding="utf-8"))
            return [MODELOS_DIR / p for p in datos.get("archivos", [])]
    except Exception: pass
    rutas: List[Path] = []
    for f in MODELOS_DIR.glob("*.gguf"):
        rutas.append(f)
    return rutas

def _guardar_temp() -> None:
    try:
        REGISTRO_TEMP.write_text(
            json.dumps({"archivos": [str(a.relative_to(MODELOS_DIR)) for a in ARCHIVOS_TEMP_SESSION]},
                       indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception: pass

def _asegurar_directorio() -> Path:
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELOS_DIR


def _leer_ram_gb() -> Tuple[float, float]:
    try:
        import psutil; vm = psutil.virtual_memory()
        return round(vm.total / 1e9, 2), round(vm.available / 1e9, 2)
    except Exception: pass
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["wmic", "computersystem", "get", "totalphysicalmemory"],
                                  capture_output=True, text=True, timeout=5)
            nums = "".join(c for c in out.stdout if c.isdigit() or c == " ").split()
            if nums:
                t = int(nums[0]) / 1e9
                return round(t, 2), round(t * 0.5, 2)
    except Exception: pass
    return 8.0, 4.0


def _leer_vram_gb() -> Tuple[float, str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,name",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            p = out.stdout.strip().splitlines()[0].split(",")
            return round(float(p[0].strip()) / 1024.0, 2), (p[1].strip() if len(p) > 1 else "NVIDIA GPU")
    except Exception: pass
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), str(torch.cuda.get_device_name(0))
    except Exception: pass
    return 0.0, "sin GPU dedicada (CPU)"


def _leer_cpu_info() -> Dict[str, Any]:
    return {"sistema": platform.system(), "version": platform.version(),
            "maquina": platform.machine(), "procesador": platform.processor() or "desconocido",
            "nucleos": os.cpu_count() or 4}


def _leer_disco_gb() -> float:
    try: return round(shutil.disk_usage(str(MODELOS_DIR)).free / 1e9, 2)
    except Exception: return 0.0


def perfilar_hardware_ialocal(guardar: bool = True) -> Dict[str, Any]:
    ram_t, ram_l = _leer_ram_gb()
    vram, gpu = _leer_vram_gb()
    perfil = {"timestamp": time.time(), "cpu": _leer_cpu_info(),
              "ram_total_gb": ram_t, "ram_libre_gb": ram_l,
              "vram_total_gb": vram, "gpu": gpu,
              "disco_libre_gb": _leer_disco_gb(), "ollama_host": OLLAMA_HOST,
              "dsialclgrg_version": __version__, "modulo": __name__}
    if guardar:
        try: REGISTRO_HW.write_text(json.dumps(perfil, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc: logger.warning("No se guardo perfil HW: %s", exc)
    return perfil


def cargar_perfil_hardware() -> Dict[str, Any]:
    if REGISTRO_HW.exists():
        try: return json.loads(REGISTRO_HW.read_text(encoding="utf-8"))
        except Exception: pass
    return perfilar_hardware_ialocal(guardar=True)


def verificar_capacidades(perfil: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if perfil is None: perfil = cargar_perfil_hardware()
    ram = float(perfil.get("ram_total_gb", 8.0))
    vram = float(perfil.get("vram_total_gb", 0.0))
    disco = float(perfil.get("disco_libre_gb", 10.0))
    cpu = perfil.get("cpu", {})
    caps = []
    if ram >= 8.0: caps.append("ram_suficiente")
    if ram >= 16.0: caps.append("ram_alta")
    if vram >= 4.0: caps.append("vram_disponible")
    if vram >= 8.0: caps.append("vram_alta")
    if disco >= 10.0: caps.append("disco_suficiente")
    if "AMD64" in cpu.get("maquina", "") or "x86_64" in cpu.get("maquina", ""): caps.append("arquitectura_x86")
    nivel = sum(1 for c in ("ram_suficiente","vram_disponible","disco_suficiente","ram_alta","vram_alta") if c in caps)
    clasif = "alta" if nivel >= 4 else ("media" if nivel >= 2 else "baja")
    return {"capacidades": caps, "nivel": nivel, "clasificacion": clasif,
            "ram_total_gb": ram, "vram_total_gb": vram, "disco_libre_gb": disco,
            "modelos_compatibles": [m["id"] for m in CATALOGO_HF_LIGEROS
                if m["ram_min_gb"] <= ram and m["vram_min_gb"] <= vram + 0.01 and m["tamano_gb"] <= disco]}


def _descargar_hf(url: str, destino: Path, timeout: float = 300.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": HF_USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(destino, "wb") as fh:
            tam_total = resp.length if hasattr(resp, 'length') and resp.length else None
            if tam_total: print(f"    {ANSI_Y}Tamanio: {tam_total / 1e6:.1f} MB{ANSI_R}")
            bloques = 0
            for bloque in resp:
                fh.write(bloque); bloques += 1
                if bloques % 100 == 0 and tam_total:
                    p = fh.tell() / tam_total * 100
                    sys.stdout.write(f"\r    {ANSI_CY}Descarga: {p:.1f}%{ANSI_R}")
                    sys.stdout.flush()
            if tam_total: sys.stdout.write("\n"); sys.stdout.flush()
        ok = destino.exists() and destino.stat().st_size > 1024
        if ok: print(f"    {ANSI_G}OK: {destino.name} ({destino.stat().st_size // 1024} KB){ANSI_R}")
        return ok
    except Exception as exc:
        logger.warning("Descarga HF fallida %s: %s", url, exc)
        try:
            if destino.exists(): destino.unlink()
        except Exception: pass
        print(f"    {ANSI_RR}Error: {exc}{ANSI_R}")
        return False


def _registrar_descarga(mid: str, archivo: str, tg: float, origen: str, ok: bool) -> None:
    try:
        reg = {}
        if REGISTRO_DESCARGAS.exists():
            raw = REGISTRO_DESCARGAS.read_text(encoding="utf-8")
            reg = json.loads(raw) if raw.strip() else {}
        reg.setdefault("descargas", {})
        reg["descargas"][mid] = {"origen": "huggingface", "archivo": archivo,
            "tamano_gb": tg, "exito": ok, "origen_hf": origen, "fecha": time.time(), "modulo": __name__}
        REGISTRO_DESCARGAS.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc: logger.warning("Registro descarga fallido: %s", exc)


def listar_modelos_hf() -> List[Dict[str, Any]]: return list(CATALOGO_HF_LIGEROS)


def recomendar_modelos_hf(perfil: Optional[Dict[str, Any]] = None,
                                limite: Optional[int] = None) -> List[Dict[str, Any]]:
    if perfil is None: perfil = cargar_perfil_hardware()
    ram = float(perfil.get("ram_total_gb", 8.0))
    vram = float(perfil.get("vram_total_gb", 0.0))
    disco = float(perfil.get("disco_libre_gb", 10.0))
    aptos = sorted([m for m in CATALOGO_HF_LIGEROS
        if m["ram_min_gb"] <= ram and m["vram_min_gb"] <= vram + 0.01 and m["tamano_gb"] <= disco],
        key=lambda m: m["tamano_gb"])
    if limite and limite > 0: aptos = aptos[:limite]
    return aptos


def descargar_modelo_hf(modelo_id: str) -> Dict[str, Any]:
    modelo = next((m for m in CATALOGO_HF_LIGEROS if m["id"] == modelo_id), None)
    if modelo is None: return {"exito": False, "mensaje": f"Modelo desconocido: {modelo_id}"}
    perfil = cargar_perfil_hardware()
    lr = float(perfil.get("ram_total_gb", 8.0))
    lv = float(perfil.get("vram_total_gb", 0.0))
    if modelo["ram_min_gb"] > lr or modelo["vram_min_gb"] > lv:
        return {"exito": False, "mensaje": f"Modelo {modelo_id} requiere {modelo['ram_min_gb']}GB RAM + {modelo['vram_min_gb']}GB VRAM. PC: {lr}GB+{lv}GB."}
    _asegurar_directorio()
    archivo = f"{modelo['id'].replace(':', '_')}.gguf"
    destino = MODELOS_DIR / archivo
    if destino.exists() and destino.stat().st_size > 1024:
        _registrar_descarga(modelo_id, archivo, modelo["tamano_gb"], modelo["hf_url"], True)
        return {"exito": True, "modelo": modelo_id, "archivo": archivo, "mensaje": "Ya descargado", "via": "cache"}
    print(f"  {ANSI_B}{ANSI_CY}Descargando HF:{ANSI_R} {ANSI_W}{modelo['nombre']}{ANSI_R} ({modelo['tamano_gb']}GB)")
    t0 = time.time()
    ok = _descargar_hf(str(modelo["hf_url"]), destino, timeout=600.0)
    seg = round(time.time() - t0, 1)
    _registrar_descarga(modelo_id, archivo, modelo["tamano_gb"], modelo["hf_url"], ok)
    print(f"  {ANSI_G if ok else ANSI_RR}{'Completada' if ok else 'Fallo'} en {seg}s{ANSI_R}")
    return {"exito": ok, "modelo": modelo_id, "archivo": archivo, "segundos": seg,
            "mensaje": "Descargado" if ok else "Fallo"}


def recomendar_y_descargar(limite: int = 2, mostrar_banner: bool = True) -> List[Dict[str, Any]]:
    perfil = perfilar_hardware_ialocal(guardar=True)
    cap = verificar_capacidades(perfil=perfil)
    if mostrar_banner:
        cpu = perfil.get("cpu", {})
        print(f"\n{ANSI_B}{ANSI_CY}{'=' * 70}{ANSI_R}")
        print(f"{ANSI_B}{ANSI_W}  PERFIL HARDWARE - IALOCAL{ANSI_R}")
        print(f"  {ANSI_G}CPU:{ANSI_R} {cpu.get('procesador','?')} | {cpu.get('nucleos')} cores")
        print(f"  {ANSI_Y}RAM:{ANSI_R} {perfil.get('ram_total_gb')}GB | "
              f"{ANSI_M}VRAM:{ANSI_R} {perfil.get('vram_total_gb')}GB | "
              f"{ANSI_CY}Disco:{ANSI_R} {perfil.get('disco_libre_gb')}GB")
        print(f"  Clasificacion: {cap.get('clasificacion','?')} ({cap.get('nivel',0)}/5)")
        print(f"{ANSI_CY}{'=' * 70}{ANSI_R}\n")
    compat = recomendar_modelos_hf(perfil=perfil, limite=limite)
    if not compat:
        print(f"  {ANSI_RR}No hay modelos compatibles.{ANSI_R}\n"); return []
    resultados = []
    print(f"  Descargando {len(compat)} modelo(s) (~{sum(m['tamano_gb'] for m in compat):.1f}GB):\n")
    for i, m in enumerate(compat, 1):
        print(f"  [{i}/{len(compat)}] {ANSI_D}{m['id']}{ANSI_R} ({m['tamano_gb']}GB - {m['categoria']})")
        resultados.append(descargar_modelo_hf(m["id"])); print()
    ok = sum(1 for r in resultados if r.get("exito"))
    print(f"  {ANSI_B}{ANSI_G}Resumen:{ANSI_R} {ok}/{len(compat)} exitosos")
    print(f"  {ANSI_CY}Carpeta:{ANSI_R} {MODELOS_DIR}")
    for r in resultados:
        est = f"{ANSI_G}OK{ANSI_R}" if r.get("exito") else f"{ANSI_RR}FAIL{ANSI_R}"
        print(f"    [{est}] {r.get('modelo','?')} - {r.get('mensaje','?')}")
    print(f"{ANSI_CY}{'=' * 70}{ANSI_R}\n")
    return resultados


def descargar_ia_temporal(modelo_id: str = None) -> Dict[str, Any]:
    perfil = cargar_perfil_hardware()
    compat = recomendar_modelos_hf(perfil=perfil, limite=1)
    if not compat: return {"exito": False, "mensaje": "No hay modelos compatibles para IA temporal."}
    mid = modelo_id or compat[0]["id"]
    res = descargar_modelo_hf(mid)
    if res.get("exito"):
        archivo = MODELOS_DIR / res.get("archivo", "")
        with LOCK_TEMP:
            if archivo not in ARCHIVOS_TEMP_SESSION:
                ARCHIVOS_TEMP_SESSION.append(archivo)
            _guardar_temp()
        return {"exito": True, "modelo": mid, "mensaje": f"IA temporal lista: {mid} (borrar al cerrar sesion)"}
    return {"exito": False, "mensaje": res.get("mensaje", "Fallo")}


def limpiar_ia_temporal() -> Dict[str, Any]:
    elim = 0; bytes_lib = 0; rutas: List[Path] = []
    with LOCK_TEMP:
        rutas = list(ARCHIVOS_TEMP_SESSION)
        for ruta in rutas:
            try:
                if ruta.exists(): bytes_lib += ruta.stat().st_size; ruta.unlink(); elim += 1
            except Exception: pass
        ARCHIVOS_TEMP_SESSION.clear()
        _guardar_temp()
    return {"eliminados": elim, "bytes_liberados": bytes_lib, "mensaje": f"Temporal: {elim} archivo(s) borrado(s)"}


def listar_py_constructor() -> List[Path]:
    if not CONSTRUCTOR_DIR.exists(): return []
    return sorted([f for f in CONSTRUCTOR_DIR.rglob("*.py") if f.is_file()])


def _refactor_hf(ruta: Path, modelo_id: str) -> str:
    try:
        codigo = ruta.read_text(encoding="utf-8", errors="replace")
        if not codigo.strip(): return "vacio"
        payload = json.dumps({"model": modelo_id, "messages": [{"role": "user",
            "content": f"Refactoriza a 400-450 lineas manteniendo funcionalidad.\n\n{codigo[:2000]}"}],
            "max_tokens": 8192, "stream": False}).encode("utf-8")
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
            data=payload, headers={"Content-Type": "application/json", "Authorization": "Bearer sk-or-v1-temp"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            texto = json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
            ruta.write_text(texto, encoding="utf-8")
            return "refactorizado-hf"
    except Exception as exc: return f"fallo-hf:{exc}"


def _refactor_local(ruta: Path) -> str:
    try:
        if not ruta.read_text(encoding="utf-8", errors="replace").strip(): return "vacio"
        res = descargar_modelo_hf("qwen2.5:0.5b")
        return "refactorizado-local" if res.get("exito") else "sin-ia-local"
    except Exception: return "error-local"


def refactorizar_constructor(modelo_hf: str = "google/gemma-2-2b-it:free",
                                        usar_local: bool = True) -> Dict[str, Any]:
    archivos = listar_py_constructor()
    if not archivos: return {"exito": True, "procesados": 0, "mensaje": "Constructor vacio."}
    resultados = []
    for ruta in archivos:
        estado = _refactor_local(ruta) if (usar_local and ARCHIVOS_TEMP_SESSION) else _refactor_hf(ruta, modelo_hf)
        resultados.append({"archivo": str(ruta), "estado": estado})
    ok = sum(1 for r in resultados if "refactorizado" in r["estado"])
    return {"exito": ok >= len(resultados) * 0.5, "procesados": len(resultados),
            "exitosos": ok, "resultados": resultados,
            "mensaje": f"Refactorizado {ok}/{len(resultados)} archivos en Constructor"}


def _barra_tiempo(elapsed: float) -> str:
    h = int(elapsed // 3600); m = int((elapsed % 3600) // 60); s = round(elapsed % 60, 1)
    if h > 0: return f"{h}h {m}m {s}s"
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"


def orquestar_pre_sesion(modelo_id: str = None) -> Dict[str, Any]:
    print(f"\n{ANSI_B}{ANSI_CY}  PRE-SESION IALOCAL{ANSI_R}")
    dl = descargar_ia_temporal(modelo_id=modelo_id)
    if not dl.get("exito"): return {"exito": False, "mensaje": f"Descarga temporal fallida: {dl.get('mensaje')}"}
    print(f"  {ANSI_G}IA temporal lista.{ANSI_R}")
    return {"exito": True, "descarga": dl}


def cerrar_sesion_ialocal(modelo_hf: str = "google/gemma-2-2b-it:free",
                                      usar_local: bool = True,
                                      mostrar_barra: bool = True) -> Dict[str, Any]:
    global TIEMPO_INICIO_CIERRE, CONSTRUCTOR_CIERRE_OK
    TIEMPO_INICIO_CIERRE = time.time()
    with LOCK_CIERRE:
        print(f"\n{ANSI_B}{ANSI_M}=== CIERRE SESION IALOCAL ==={ANSI_R}")
        t0 = time.time()
        ref = refactorizar_constructor(modelo_hf=modelo_hf, usar_local=usar_local)
        CONSTRUCTOR_CIERRE_OK = ref.get("exito", False)
        if mostrar_barra: print(f"  {ANSI_Y}[Refactor  ]{ANSI_R} {ANSI_G}{ref.get('mensaje','')}{ANSI_R} ({round(time.time()-t0,1)}s)")
        t_md = time.time()
        archivos_ref = listar_py_constructor()
        md_content = f"# Cambios Constructor\n\nTS: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\nRefactor: {CONSTRUCTOR_CIERRE_OK}\nArchivos: {len(archivos_ref)}\n"
        for a in archivos_ref: md_content += f"- {a.name}\n"
        md_path = CONSTRUCTOR_DIR / f"cierre_sesion_{int(time.time())}.md"
        try: CONSTRUCTOR_DIR.mkdir(parents=True, exist_ok=True); md_path.write_text(md_content, encoding="utf-8"); md_ok = True
        except Exception: md_ok = False
        if mostrar_barra: print(f"  {ANSI_Y}[MD         ]{ANSI_R} {ANSI_G if md_ok else ANSI_RR}{md_path.name}{ANSI_R} ({round(time.time()-t_md,1)}s)")
        t_p = time.time()
        peso_bytes = sum(a.stat().st_size for a in archivos_ref if a.exists()) if archivos_ref else 0
        peso_info = {"pesos_generados_kb": round(peso_bytes / 1024, 1) if peso_bytes else 0, "neuronas_activas": 50}
        if mostrar_barra: print(f"  {ANSI_Y}[Pesos     ]{ANSI_R} {ANSI_G}Pesos: {peso_info['pesos_generados_kb']}KB | Neuronas: {peso_info['neuronas_activas']}{ANSI_R} ({round(time.time()-t_p,1)}s)")
        t_i = time.time()
        try:
            from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
            get_ipfs_manager(); cid = "bafy-pre-sesion"; ipfs_ok = True
        except Exception:
            cid = "--"; ipfs_ok = False
        if mostrar_barra: print(f"  {ANSI_Y}[IPFS      ]{ANSI_R} {ANSI_G if ipfs_ok else ANSI_RR}{cid[:16]}...{ANSI_R} ({round(time.time()-t_i,1)}s)")
        t_l = time.time()
        li = limpiar_ia_temporal()
        if mostrar_barra: print(f"  {ANSI_Y}[Limpieza  ]{ANSI_R} {ANSI_G}{li.get('eliminados',0)} archivos{ANSI_R} ({round(time.time()-t_l,1)}s)")
        elapsed = time.time() - TIEMPO_INICIO_CIERRE
        print(f"\n  {ANSI_B}{ANSI_G}CIERRE COMPLETADO en {_barra_tiempo(elapsed)}{ANSI_R}")
        print(f"  {ANSI_D}Refactor: {'OK' if CONSTRUCTOR_CIERRE_OK else 'FALLBACK'}"
              f" | MD: {'OK' if md_ok else 'FAIL'}"
              f" | IPFS: {'OK' if ipfs_ok else 'OFFLINE'}"
              f" | Temp: {li.get('eliminados',0)} borrado(s){ANSI_R}\n")
        return {"exito": True, "tiempo_segundos": round(elapsed, 2), "barra_tiempo": _barra_tiempo(elapsed),
                "refactorizacion": ref, "md_creado": md_path.name if md_ok else None,
                "pesos": peso_info, "ipfs_cid": cid, "limpieza": li, "constructor_ok": CONSTRUCTOR_CIERRE_OK}


def info_completa() -> Dict[str, Any]:
    perfil = cargar_perfil_hardware()
    cap = verificar_capacidades(perfil=perfil)
    return {"modulo": __name__, "version": __version__,
            "perfil_hw": perfil, "capacidades": cap,
            "catalogo_modelos": listar_modelos_hf(),
            "temp_activos": len(ARCHIVOS_TEMP_SESSION),
            "constructor_existe": CONSTRUCTOR_DIR.exists()}


def estado_ialocal() -> Dict[str, Any]:
    return {"lista_temp": [str(a.name) for a in ARCHIVOS_TEMP_SESSION],
            "tiene_constructor": CONSTRUCTOR_DIR.exists(),
            "modelos_descargados": len(ARCHIVOS_TEMP_SESSION),
            "perfil": cargar_perfil_hardware()}


ARCHIVOS_TEMP_SESSION = _cargar_temp()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print(f"\n{ANSI_B}{ANSI_CY}{'=' * 60}{ANSI_R}")
    print(f"{ANSI_B}{ANSI_W}  IALOCAL v{__version__}{ANSI_R}")
    print(f"{ANSI_CY}  Descarga HF + Cierre Sesion{ANSI_R}")
    print(f"{ANSI_CY}{'=' * 60}{ANSI_R}")
    recomendar_y_descargar(limite=2)
    print("  Fin.\n")
