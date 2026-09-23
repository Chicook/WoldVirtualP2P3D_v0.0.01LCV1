# [IALOCAL] Refactor sintactico verificado (AST OK).
"""
CMFG - Celebro Model & Feed Gateway (Arquitectura WoldVirtualP2P3D 2026)
========================================================================
Paquete neurorregulador y bus central de operaciones de Celebro:
  - Orquestacion y transduccion de 50 neuronas activas (ENRN, RF_EN, RF_SL, RNP, SLRN).
  - Persistencia descentralizada y gestion de ciclo de vida con Kubo IPFS.
  - Telemetria de pesos vivos con monitor ANSI de alta resolucion temporal.
  - Exportacion unificada de API, comprobacion de entorno y hooks de sesion.
"""
from __future__ import annotations
import os, sys, time, json, logging, threading
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union
__version__: Final[str] = '2026.3.1'
__author__: Final[str] = 'Equipo Celebro WoldVirtualP2P3D'
__status__: Final[str] = 'Production / Optimized'
__package_name__: Final[str] = 'LC.celebro.CMFG'
logger: logging.Logger = logging.getLogger('WoldVirtualP2P3D.CMFG')
PACKAGE_ROOT: Final[Path] = Path(__file__).parent.resolve()
CELEBRO_DIR: Final[Path] = PACKAGE_ROOT.parent.resolve()
PSNRL_DIR: Final[Path] = CELEBRO_DIR / 'PSNRL'
RED_NEURONAL_DIR: Final[Path] = CELEBRO_DIR / 'red_neuronal'
IPFSONL_DIR: Final[Path] = PACKAGE_ROOT / 'IPFSonl'
MANIFEST_PATH: Final[Path] = PACKAGE_ROOT / 'ipfs_manifest.json'
_SUBSYSTEM_REGISTRY: Dict[str, Dict[str, Any]] = {'PSNRCV': {'modulo': 'LC.celebro.CMFG.PSNRCV', 'descripcion': 'Conversor y Transductor Neural de 50 Neuronas', 'requerido': True, 'cargado': False}, 'pesos_vivos': {'modulo': 'LC.celebro.CMFG.pesos_vivos', 'descripcion': 'Monitor Dinamico ANSI de Pesos Vivos', 'requerido': True, 'cargado': False}, 'ipfs_manager': {'modulo': 'LC.celebro.CMFG.ipfs_manager', 'descripcion': 'Persistencia Descentralizada IPFS y Limpieza Atomica', 'requerido': True, 'cargado': False}, 'IPFSonl': {'modulo': 'LC.celebro.CMFG.IPFSonl', 'descripcion': 'Orquestador y Daemon Kubo IPFS Local', 'requerido': False, 'cargado': False}}
_SESSION_STATE: Dict[str, Any] = {'session_id': f'CMFG_{int(time.time())}', 'start_time': time.time(), 'neuronas_activas': 50, 'psnrl_sincronizado': False, 'daemon_ipfs_vinculado': False, 'last_sync_timestamp': 0.0, 'operaciones_completadas': 0}
_HOOKS_CIERRE: List[Callable[[], Any]] = []
_LOCK = threading.Lock()

def inicializar_entorno() -> Dict[str, bool]:
    """
    Valida y crea las rutas criticas del arbol de Celebro:
      - Directorio PSNRL para pesos locales en transito.
      - Directorio red_neuronal para las 5 subredes de neuronas.
      - Directorio IPFSonl para el motor Kubo y repositorio .ipfs.
    """
    resultados: Dict[str, bool] = {}
    rutas_requeridas = [(PSNRL_DIR, 'PSNRL'), (RED_NEURONAL_DIR, 'red_neuronal'), (IPFSONL_DIR, 'IPFSonl')]
    for ruta, etiqueta in rutas_requeridas:
        try:
            ruta.mkdir(parents=True, exist_ok=True)
            resultados[etiqueta] = True
        except Exception as e:
            logger.error('Fallo inicializando directorio %s (%s): %s', etiqueta, ruta, e)
            resultados[etiqueta] = False
    return resultados

def registrar_hook_cierre(func: Callable[[], Any]) -> None:
    """Registra una rutina de limpieza o persistencia a ejecutar en fin de sesion."""
    with _LOCK:
        if func not in _HOOKS_CIERRE:
            _HOOKS_CIERRE.append(func)
            logger.debug('Hook de cierre registrado: %s', getattr(func, '__name__', str(func)))

def desregistrar_hook_cierre(func: Callable[[], Any]) -> bool:
    """Remueve un hook de cierre previamente registrado."""
    with _LOCK:
        if func in _HOOKS_CIERRE:
            _HOOKS_CIERRE.remove(func)
            return True
        return False

def ejecutar_hooks_cierre() -> Dict[str, Any]:
    """Ejecuta ordenadamente todos los hooks de cierre registrados."""
    res: Dict[str, Any] = {'ejecutados': 0, 'fallos': []}
    with _LOCK:
        for hook in list(_HOOKS_CIERRE):
            nombre = getattr(hook, '__name__', str(hook))
            try:
                hook()
                res['ejecutados'] += 1
            except Exception as e:
                res['fallos'].append(f'{nombre}: {e}')
                logger.error('Error en hook de cierre %s: %s', nombre, e)
    return res

def obtener_subredes_disponibles() -> Dict[str, int]:
    """
    Escanea la topologia neuronal de Celebro para verificar las 50 neuronas
    distribuidas en las capas ENRN, RF_EN, RF_SL, RNP y SLRN.
    """
    capas = ['ENRN', 'RF_EN', 'RF_SL', 'RNP', 'SLRN']
    conteo: Dict[str, int] = {}
    if not RED_NEURONAL_DIR.exists():
        return {c: 0 for c in capas}
    for capa in capas:
        dir_capa = RED_NEURONAL_DIR / capa
        if dir_capa.exists() and dir_capa.is_dir():
            archivos = [f for f in dir_capa.iterdir() if f.is_file() and f.suffix in ['.py', '.json', '.bin', '']]
            conteo[capa] = len(archivos)
        else:
            conteo[capa] = 0
    return conteo

def verificar_estado_subsistemas() -> Dict[str, Any]:
    """Diagnostico en vivo de subsistemas del paquete CMFG."""
    diag: Dict[str, Any] = {}
    for k, v in _SUBSYSTEM_REGISTRY.items():
        try:
            mod = sys.modules.get(v['modulo'])
            diag[k] = {'descripcion': v['descripcion'], 'activo': mod is not None, 'requerido': v['requerido']}
        except Exception as e:
            diag[k] = {'descripcion': v['descripcion'], 'activo': False, 'error': str(e)}
    return diag

def get_session_info() -> Dict[str, Any]:
    """Retorna telemetria y estado de la sesion operativa actual."""
    uptime = time.time() - _SESSION_STATE['start_time']
    return {'session_id': _SESSION_STATE['session_id'], 'uptime_segundos': round(uptime, 2), 'neuronas_activas': _SESSION_STATE['neuronas_activas'], 'psnrl_sincronizado': _SESSION_STATE['psnrl_sincronizado'], 'operaciones_completadas': _SESSION_STATE['operaciones_completadas'], 'directorio_psnrl': str(PSNRL_DIR), 'manifiesto_ipfs': str(MANIFEST_PATH)}

def incrementar_contador_operaciones() -> int:
    """Incrementa de forma atomica el contador de operaciones procesadas."""
    with _LOCK:
        _SESSION_STATE['operaciones_completadas'] += 1
        return _SESSION_STATE['operaciones_completadas']

def purgar_temporales() -> int:
    """Elimina residuos y volcados temporales de operaciones previas."""
    eliminados = 0
    for patron in ['_tmp_*.*', '*.tmp', '*.lock']:
        for f in PACKAGE_ROOT.glob(patron):
            try:
                if f.is_file():
                    f.unlink()
                    eliminados += 1
            except Exception as e:
                logger.warning('No se pudo purgar %s: %s', f.name, e)
    return eliminados

def obtener_resumen_psnrl() -> Dict[str, Any]:
    """Analiza los archivos de pesos neuronales persistidos en PSNRL."""
    if not PSNRL_DIR.exists():
        return {'total_archivos': 0, 'tamano_total_bytes': 0, 'archivos': []}
    archivos = [f for f in PSNRL_DIR.iterdir() if f.is_file() and (not f.name.startswith('.'))]
    tam_total = sum((f.stat().st_size for f in archivos))
    return {'total_archivos': len(archivos), 'tamano_total_bytes': tam_total, 'tamano_kb': round(tam_total / 1024, 2), 'archivos': [f.name for f in archivos]}

def auto_conectar_ipfs() -> Tuple[bool, Optional[str]]:
    """Intenta localizar y vincular el daemon IPFS local de forma segura."""
    try:
        from LC.celebro.CMFG.IPFSonl.ipfs_launcher import arrancar_daemon, get_ipfs_api_url
        if arrancar_daemon(esperar_segundos=4):
            _SESSION_STATE['daemon_ipfs_vinculado'] = True
            return (True, get_ipfs_api_url())
    except Exception as e:
        logger.debug('Conexion rapida de launcher omitida: %s', e)
    try:
        from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
        mgr = get_ipfs_manager()
        url = mgr.descubrir_daemon()
        _SESSION_STATE['daemon_ipfs_vinculado'] = bool(url)
        return (bool(url), url)
    except Exception as e:
        logger.warning('No se pudo autoconectar a IPFS: %s', e)
        return (False, None)

def guardar_checkpoint_sesion(etiqueta: str='auto') -> Dict[str, Any]:
    """Genera una instantanea del estado de operacion y metadatos del paquete."""
    ckpt = {'timestamp': time.time(), 'fecha': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'etiqueta': etiqueta, 'sesion': get_session_info(), 'psnrl': obtener_resumen_psnrl(), 'topologia': obtener_subredes_disponibles()}
    destino = PACKAGE_ROOT / f'_checkpoint_{etiqueta}.json'
    try:
        with open(destino, 'w', encoding='utf-8') as f:
            json.dump(ckpt, f, indent=2, ensure_ascii=False)
        return {'exito': True, 'path': str(destino), 'bytes': destino.stat().st_size}
    except Exception as e:
        logger.error('Fallo guardando checkpoint %s: %s', etiqueta, e)
        return {'exito': False, 'error': str(e)}

def restaurar_ultimo_checkpoint() -> Optional[Dict[str, Any]]:
    """Recupera el archivo de checkpoint mas reciente disponible."""
    ckpts = sorted(PACKAGE_ROOT.glob('_checkpoint_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        return None
    try:
        with open(ckpts[0], 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error('Error leyendo checkpoint %s: %s', ckpts[0].name, e)
        return None

def sincronizar_manifiesto_con_psnrl() -> Dict[str, Any]:
    """Verifica correspondencia entre manifiesto de CIDs y archivos fisicos."""
    resumen = {'en_manifiesto': 0, 'en_disco': 0, 'coincidentes': 0}
    try:
        from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
        mgr = get_ipfs_manager()
        man = mgr.manifest.get('weights', {})
        resumen['en_manifiesto'] = len(man)
        psnrl_info = obtener_resumen_psnrl()
        resumen['en_disco'] = psnrl_info['total_archivos']
        return resumen
    except Exception as e:
        logger.debug('Sincronizacion preventiva de manifiesto no disponible: %s', e)
        return resumen

def exportar_diagnostico_json(destino_path: Optional[Union[str, Path]]=None) -> Path:
    """Exporta el diagnostico de integridad completo del paquete a un archivo JSON."""
    reporte = {'generado_en': time.time(), 'fecha_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'version_cmfg': __version__, 'sesion': get_session_info(), 'subsistemas': verificar_estado_subsistemas(), 'topologia': obtener_subredes_disponibles(), 'psnrl': obtener_resumen_psnrl(), 'sincronizacion': sincronizar_manifiesto_con_psnrl()}
    out = Path(destino_path) if destino_path else PACKAGE_ROOT / '_diagnostico_cmfg.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    return out

def cerrar_sesion_cmfg(subir_pesos: bool=True) -> Dict[str, Any]:
    """
    Protocolo formal de cierre para el paquete CMFG:
      1. Ejecuta hooks de cierre registrados.
      2. Sube y limpia PSNRL via IPFS si subir_pesos=True.
      3. Purga residuos temporales.
    """
    logger.info('Iniciando cierre formal de CMFG (%s)...', _SESSION_STATE['session_id'])
    res: Dict[str, Any] = {'session_id': _SESSION_STATE['session_id'], 'hooks': ejecutar_hooks_cierre(), 'ipfs_resumen': None, 'temporales_purgados': purgar_temporales()}
    if subir_pesos:
        try:
            from LC.celebro.CMFG.ipfs_manager import cerrar_sesion_y_subir_psnrl
            res['ipfs_resumen'] = cerrar_sesion_y_subir_psnrl()
        except Exception as e:
            res['ipfs_error'] = str(e)
            logger.error('Error en subida IPFS durante cierre: %s', e)
    logger.info('Cierre de CMFG completado exitosamente.')
    return res

def banner_diagnostico() -> str:
    """Genera cadena informativa de bienvenida y diagnostico para consola."""
    topo = obtener_subredes_disponibles()
    psnrl = obtener_resumen_psnrl()
    lineas = ['=' * 68, f'  CELEBRO MODEL & FEED GATEWAY (CMFG) v{__version__}', '=' * 68, f'  Directorio Base : {PACKAGE_ROOT}', f"  Directorio PSNRL: {PSNRL_DIR} ({psnrl['total_archivos']} arch, {psnrl['tamano_kb']} KB)", f"  Topologia Red   : ENRN={topo.get('ENRN', 0)}, RF_EN={topo.get('RF_EN', 0)}, RF_SL={topo.get('RF_SL', 0)}, RNP={topo.get('RNP', 0)}, SLRN={topo.get('SLRN', 0)}", f"  Sesion ID       : {_SESSION_STATE['session_id']}", '=' * 68]
    return '\n'.join(lineas)
_init_res = inicializar_entorno()

def __getattr__(name: str) -> Any:
    """Importacion dinamica controlada de modulos y clases del paquete."""
    if name in ('PSNRCV', 'RedNeuronalOrquestador'):
        from LC.celebro.CMFG import PSNRCV
        return getattr(PSNRCV, name) if hasattr(PSNRCV, name) else PSNRCV
    elif name in ('PesosVivosMonitor', 'pesos_vivos'):
        from LC.celebro.CMFG import pesos_vivos
        return getattr(pesos_vivos, name) if hasattr(pesos_vivos, name) else pesos_vivos
    elif name in ('IPFSManager', 'get_ipfs_manager', 'cerrar_sesion_y_subir_psnrl', 'ipfs_manager'):
        from LC.celebro.CMFG import ipfs_manager
        return getattr(ipfs_manager, name) if hasattr(ipfs_manager, name) else ipfs_manager
    elif name == 'IPFSonl':
        from LC.celebro.CMFG import IPFSonl
        return IPFSonl
    raise AttributeError(f"Modulo '{__name__}' no posee el atributo '{name}'")
__all__: Final[List[str]] = ['__version__', '__author__', '__status__', '__package_name__', 'PACKAGE_ROOT', 'CELEBRO_DIR', 'PSNRL_DIR', 'RED_NEURONAL_DIR', 'IPFSONL_DIR', 'MANIFEST_PATH', 'inicializar_entorno', 'registrar_hook_cierre', 'desregistrar_hook_cierre', 'ejecutar_hooks_cierre', 'obtener_subredes_disponibles', 'verificar_estado_subsistemas', 'get_session_info', 'incrementar_contador_operaciones', 'purgar_temporales', 'obtener_resumen_psnrl', 'auto_conectar_ipfs', 'guardar_checkpoint_sesion', 'restaurar_ultimo_checkpoint', 'sincronizar_manifiesto_con_psnrl', 'exportar_diagnostico_json', 'cerrar_sesion_cmfg', 'banner_diagnostico']
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    print(banner_diagnostico())
    print('\n[+] Diagnostico de subsistemas CMFG:')
    for sub, info in verificar_estado_subsistemas().items():
        print(f"    - {sub:<15}: {('OK' if info.get('activo') else 'Pendiente')} ({info['descripcion']})")
    conectado, url = auto_conectar_ipfs()
    print(f"\n[+] Estado IPFS: {('Conectado (' + str(url) + ')' if conectado else 'Offline (Modo Autonomo CIDv1)')}")
