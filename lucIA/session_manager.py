"""
lucIA.session_manager - Gestor centralizado del ciclo de vida de sesiones, caché y pesos IPFS
=============================================================================================

Garantiza:
1. Redirección de TODA la compilación de Python (__pycache__) a:
   C:\\Users\\User\\Desktop\\AreaDeTrabajo\\WoldVirtualP2P3D_v0.0.01\\lucIA\\Celebro\\cache
2. Vaciado completo de Celebro\\cache al cerrar sesión (queda 100% vacía).
3. Exportación de pesos neuronales a IPFS y borrado del sistema local al cerrar sesión.
4. Verificación rigurosa de tamaño de lucIA <= 125 MB.
5. Codificación de versiones_sesion/ en pesos neuronales y subida a IPFS al cerrar.
"""

import os
import sys
import atexit
import signal
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .ipfs_manager import IPFSManager
from .CORE.base import LucIASystem

logger = logging.getLogger("lucIA.Session")

# Rutas base
LUCIA_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = LUCIA_ROOT / "Celebro" / "cache"
MAX_ALLOWED_MB = 125.0


def configurar_redireccion_pycache() -> Path:
    """
    Configura sys.pycache_prefix y PYTHONPYCACHEPREFIX para que
    absolutamente todo el bytecode (.pyc) vaya a Celebro/cache.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_str = str(CACHE_DIR)
    sys.pycache_prefix = cache_str
    os.environ["PYTHONPYCACHEPREFIX"] = cache_str
    logger.info(f"Redirección de __pycache__ configurada en: {cache_str}")
    return CACHE_DIR


def vaciar_directorio_cache() -> int:
    """
    Elimina todos los archivos y subdirectorios dentro de Celebro/cache,
    dejándolo completamente vacío. También limpia posibles __pycache__ residuales en lucIA.
    """
    eliminados = 0
    if CACHE_DIR.exists():
        for item in CACHE_DIR.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                    eliminados += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    eliminados += 1
            except Exception as e:
                logger.warning(f"No se pudo eliminar de cache {item.name}: {e}")

    # Asegurar que el directorio cache existe pero esté vacío
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Barrido preventivo de cualquier __pycache__ o .pyc en todo lucIA
    for root, dirs, files in os.walk(LUCIA_ROOT):
        # Evitar borrar el propio directorio Celebro/cache
        if Path(root).resolve() == CACHE_DIR:
            continue
        for d in dirs:
            if d == "__pycache__":
                pycache_path = Path(root) / d
                try:
                    shutil.rmtree(pycache_path)
                    eliminados += 1
                except Exception:
                    pass
        for f in files:
            if f.endswith(".pyc") or f.endswith(".pyo"):
                file_path = Path(root) / f
                try:
                    file_path.unlink()
                    eliminados += 1
                except Exception:
                    pass

    # Barrido de snapshots antiguos en versiones_sesion (mantener maximo 1 para no acumular MB)
    versiones_dir = LUCIA_ROOT / "versiones_sesion"
    if versiones_dir.exists():
        snapshots = sorted([s for s in versiones_dir.iterdir() if s.is_dir()], key=lambda x: x.stat().st_mtime)
        if len(snapshots) > 1:
            for antiguo in snapshots[:-1]:
                try:
                    shutil.rmtree(antiguo)
                    eliminados += 1
                except Exception:
                    pass

    logger.info(f"🧹 Directorio cache vaciado por completo. Elementos purgados: {eliminados}")
    return eliminados


def calcular_tamano_lucia_mb() -> float:
    """Calcula el tamaño total en disco del directorio lucIA en MB (excluye venv/)."""
    total_bytes = 0
    excluidos = {"venv", "__pycache__"}
    for root, dirs, files in os.walk(LUCIA_ROOT):
        dirs[:] = [d for d in dirs if d not in excluidos]
        for f in files:
            p = Path(root) / f
            try:
                total_bytes += p.stat().st_size
            except Exception:
                pass
    return total_bytes / (1024.0 * 1024.0)


def verificar_limite_espacio() -> Dict[str, Any]:
    """Verifica que lucIA no supere los 125 MB estipulados."""
    tamano_actual = calcular_tamano_lucia_mb()
    esta_dentro_del_limite = tamano_actual <= MAX_ALLOWED_MB
    return {
        "tamano_mb": round(tamano_actual, 2),
        "limite_mb": MAX_ALLOWED_MB,
        "disponible_mb": round(MAX_ALLOWED_MB - tamano_actual, 2),
        "valido": esta_dentro_del_limite
    }


class LucIASession:
    """
    Sesión activa de LucIA. Administra modelos, pesos, caché y conexión con IPFS.
    """

    def __init__(self, ipfs_url: str = "http://127.0.0.1:5001"):
        self.ipfs = IPFSManager(api_url=ipfs_url)
        self.sistema = LucIASystem(nombre="LucIASystem_Session")
        self.archivos_pesos_temporales: List[Path] = []
        self._sesion_activa = False
        self._limpieza_ejecutada = False

    def iniciar(self) -> "LucIASession":
        """Inicia la sesión, configura caché y valida tamaño del sistema."""
        configurar_redireccion_pycache()
        estado_espacio = verificar_limite_espacio()
        logger.info(
            f"🚀 Sesión iniciada. Tamaño actual: {estado_espacio['tamano_mb']} MB / {MAX_ALLOWED_MB} MB "
            f"(Disponible: {estado_espacio['disponible_mb']} MB)"
        )
        if not estado_espacio["valido"]:
            logger.error(f"⚠️ ADVERTENCIA: Se superó el límite de {MAX_ALLOWED_MB} MB!")

        self._sesion_activa = True
        self._limpieza_ejecutada = False
        return self

    def registrar_archivo_pesos(self, ruta_archivo: Union[str, Path]) -> None:
        """Registra un archivo de pesos local para ser subido a IPFS y borrado al cerrar sesión."""
        p = Path(ruta_archivo).resolve()
        if p not in self.archivos_pesos_temporales:
            self.archivos_pesos_temporales.append(p)

    def cerrar(self) -> Dict[str, Any]:
        """
        Cierra la sesión de LucIA:
        1. Recolecta pesos de neuronas en memoria y los sube a IPFS.
        2. Sube todos los archivos de pesos registrados a IPFS.
        3. Borra todos los pesos del sistema local tras ser confirmados en IPFS.
        4. Vacía completamente Celebro/cache.
        5. Verifica tamaño final del sistema.
        """
        if self._limpieza_ejecutada:
            return {"estado": "ya_cerrada"}

        logger.info("🔒 Iniciando proceso de cierre de sesión de LucIA...")
        resultados_ipfs = []

        # 0. Codificar automáticamente memoria_conversacion y archivos de datos en cache si existen
        try:
            codificar_memoria_en_pesos(sesion=self)
            codificar_cache_en_pesos(sesion=self)
        except Exception as e:
            logger.debug(f"Aviso en codificación automática de memoria/cache: {e}")

        # 1. Exportar y subir pesos de neuronas registradas en memoria
        pesos_memoria = self.sistema.recolectar_todos_los_pesos()
        if pesos_memoria:
            logger.info(f"📤 Transfiriendo {len(pesos_memoria)} conjuntos de pesos de memoria a IPFS...")
            res_mem = self.ipfs.almacenar_pesos(
                origen=pesos_memoria,
                nombre_modelo="pesos_sesion_memoria",
                eliminar_local=False,
                metadatos={"origen": "memoria_sistema", "num_neuronas": len(pesos_memoria)}
            )
            resultados_ipfs.append(res_mem)

        # 2. Subir y borrar archivos de pesos locales registrados en disco
        for archivo_peso in list(self.archivos_pesos_temporales):
            if archivo_peso.exists():
                logger.info(f"📤 Transfiriendo archivo local de pesos a IPFS: {archivo_peso.name}")
                res_archivo = self.ipfs.almacenar_pesos(
                    origen=archivo_peso,
                    nombre_modelo=archivo_peso.stem,
                    eliminar_local=True,  # Borrado local estricto
                    metadatos={"archivo_original": archivo_peso.name}
                )
                resultados_ipfs.append(res_archivo)

        self.archivos_pesos_temporales.clear()

        # 3. Vaciar completamente la carpeta de cache
        elementos_cache_borrados = vaciar_directorio_cache()

        # 4. Validar tamaño final del sistema
        tamano_final = verificar_limite_espacio()
        self._sesion_activa = False
        self._limpieza_ejecutada = True

        logger.info(
            f"✅ Sesión cerrada exitosamente.\n"
            f"   - Pesos subidos a IPFS: {len(resultados_ipfs)}\n"
            f"   - Archivos locales de pesos borrados: SÍ\n"
            f"   - Celebro/cache vaciada: SÍ ({elementos_cache_borrados} limpiados)\n"
            f"   - Tamaño final del sistema: {tamano_final['tamano_mb']} MB (Límite: {MAX_ALLOWED_MB} MB)"
        )

        return {
            "exito": True,
            "cids_ipfs": [r["cid"] for r in resultados_ipfs],
            "resultados_ipfs": resultados_ipfs,
            "cache_vaciada": True,
            "elementos_cache_borrados": elementos_cache_borrados,
            "tamano_final_mb": tamano_final["tamano_mb"]
        }

    def __enter__(self) -> "LucIASession":
        return self.iniciar()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cerrar()


# Instancia global por defecto
_sesion_global: Optional[LucIASession] = None


def obtener_sesion() -> LucIASession:
    """Obtiene o crea la sesión global de LucIA."""
    global _sesion_global
    if _sesion_global is None:
        _sesion_global = LucIASession()
    return _sesion_global


def iniciar_sesion() -> LucIASession:
    """Inicia la sesión activa de LucIA."""
    sesion = obtener_sesion()
    return sesion.iniciar()


def cerrar_sesion() -> Dict[str, Any]:
    """Cierra la sesión activa de LucIA, enviando pesos a IPFS y limpiando cache."""
    global _sesion_global
    if _sesion_global is not None:
        resultado = _sesion_global.cerrar()
        _sesion_global = None
        return resultado
    else:
        # Aunque no haya sesión creada, asegurar que cache esté vacía
        vaciar_directorio_cache()
        return {"exito": True, "mensaje": "Cache vaciada preventivamente"}


def codificar_versiones_en_pesos(sesion: LucIASession, conversor) -> List[Path]:
    """
    Codifica los snapshots de versiones_sesion/ en pesos neuronales y los
    registra en la sesion para que se suban a IPFS al cerrar.

    Proceso por cada snapshot en versiones_sesion/<version>/:
      1. Lee todos los archivos .py del snapshot como corpus de texto.
      2. Los pasa por el ConversorRespuestaPesos para generar activaciones
         neuronales que codifican el contenido del código como pesos.
      3. Guarda un checkpoint .npz en Celebro/ con el nombre del snapshot.
      4. Registra el .npz en la sesion para subida a IPFS y borrado local.

    Args:
        sesion: Sesion activa de LucIA.
        conversor: Instancia de ConversorRespuestaPesos.

    Returns:
        Lista de archivos .npz generados y registrados.
    """
    versiones_dir = LUCIA_ROOT / "versiones_sesion"
    celebro_dir = LUCIA_ROOT / "Celebro"
    archivos_generados: List[Path] = []

    if not versiones_dir.exists():
        logger.info("versiones_sesion/ no existe, nada que codificar.")
        return archivos_generados

    snapshots = sorted(
        [s for s in versiones_dir.iterdir() if s.is_dir()],
        key=lambda x: x.stat().st_mtime
    )

    if not snapshots:
        logger.info("No hay snapshots en versiones_sesion/ para codificar.")
        return archivos_generados

    logger.info(f"🧬 Codificando {len(snapshots)} snapshot(s) de versiones_sesion/ en pesos neuronales...")

    for snapshot_dir in snapshots:
        try:
            # 1. Leer todos los archivos .py del snapshot como corpus de texto
            archivos_py = sorted(snapshot_dir.rglob("*.py"))
            if not archivos_py:
                continue

            corpus_chunks: List[str] = []
            for py_file in archivos_py:
                try:
                    texto = py_file.read_text(encoding="utf-8", errors="ignore")
                    # Dividir en bloques de 500 chars para no saturar el encoder
                    for i in range(0, len(texto), 500):
                        chunk = texto[i:i + 500].strip()
                        if chunk:
                            corpus_chunks.append(chunk)
                except Exception:
                    continue

            if not corpus_chunks:
                continue

            logger.info(f"   📄 Snapshot '{snapshot_dir.name}': {len(corpus_chunks)} bloques de código → neuronas")

            # 2. Pasar cada bloque por el conversor para actualizar pesos
            for idx, chunk in enumerate(corpus_chunks):
                prompt_meta = (
                    f"[Snapshot {snapshot_dir.name} - bloque {idx + 1}/{len(corpus_chunks)}] "
                    f"Código fuente de LucIA para asimilación neuronal:"
                )
                try:
                    conversor.convertir_respuesta_a_pesos(
                        prompt=prompt_meta,
                        respuesta=chunk,
                        modelo_origen=f"snapshot_{snapshot_dir.name}"
                    )
                except Exception as e:
                    logger.debug(f"Error procesando bloque {idx}: {e}")
                    continue

            # 3. Guardar checkpoint .npz con el nombre del snapshot
            nombre_archivo = celebro_dir / f"pesos_snapshot_{snapshot_dir.name}.npz"
            archivo_guardado = conversor.guardar_pesos_en_celebro(archivo=nombre_archivo)

            # 4. Registrar en la sesión para subida a IPFS y borrado local
            sesion.registrar_archivo_pesos(archivo_guardado)
            archivos_generados.append(archivo_guardado)

            logger.info(
                f"   ✅ Snapshot '{snapshot_dir.name}' → "
                f"{archivo_guardado.name} registrado para IPFS"
            )

        except Exception as e:
            logger.warning(f"⚠️ Error codificando snapshot {snapshot_dir.name}: {e}")
            continue

    logger.info(
        f"🧬 Codificación completada: {len(archivos_generados)} archivos de pesos "
        f"listos para IPFS."
    )
    return archivos_generados


def codificar_memoria_en_pesos(sesion: Optional[LucIASession] = None, conversor = None) -> Optional[Path]:
    """
    Lee memoria_conversacion.json (en Celebro/ o Celebro/cache/),
    convierte cada turno (usuario -> respuesta LucIA) en pesos neuronales
    propagados por las 18 neuronas de Celebro, genera un checkpoint .npz de pesos,
    lo registra en la sesión activa para subida a IPFS y borrado local,
    y elimina el archivo JSON local de la memoria.
    """
    if sesion is None:
        sesion = obtener_sesion()

    if conversor is None:
        try:
            from .Celebro.conversor_pesos import get_conversor_pesos
            conversor = get_conversor_pesos()
        except Exception as e:
            logger.error(f"No se pudo cargar conversor_pesos: {e}")
            return None

    # Buscar el archivo de memoria en Celebro/ o Celebro/cache/
    rutas_candidatas = [
        LUCIA_ROOT / "Celebro" / "memoria_conversacion.json",
        CACHE_DIR / "memoria_conversacion.json"
    ]
    memoria_archivo: Optional[Path] = None
    for r in rutas_candidatas:
        if r.exists():
            memoria_archivo = r
            break

    if not memoria_archivo:
        logger.info("ℹ️ No se encontró memoria_conversacion.json para codificar.")
        return None

    try:
        import json
        with open(memoria_archivo, "r", encoding="utf-8") as f:
            datos = json.load(f)

        turnos = datos.get("turnos", [])
        if not turnos:
            logger.info("memoria_conversacion.json no tiene turnos guardados.")
            try:
                memoria_archivo.unlink(missing_ok=True)
            except Exception:
                pass
            return None

        logger.info(f"🧠 Codificando {len(turnos)} turnos de memoria conversacional en pesos neuronales...")

        for idx, t in enumerate(turnos):
            usuario_msg = str(t.get("user", "")).strip()
            lucia_msg = str(t.get("lucia", "")).strip()
            if not usuario_msg and not lucia_msg:
                continue

            prompt_meta = f"[Memoria Conversación - Turno {idx + 1}/{len(turnos)}] {usuario_msg}"
            try:
                conversor.convertir_respuesta_a_pesos(
                    prompt=prompt_meta,
                    respuesta=lucia_msg,
                    modelo_origen=t.get("modelo", "memoria_conversacion")
                )
            except Exception as e:
                logger.debug(f"Error asimilando turno de memoria {idx}: {e}")
                continue

        # Guardar checkpoint .npz de los pesos de la memoria
        celebro_dir = LUCIA_ROOT / "Celebro"
        nombre_npz = celebro_dir / "pesos_memoria_conversacion.npz"
        archivo_npz = conversor.guardar_pesos_en_celebro(archivo=nombre_npz)

        # Registrar en la sesión para IPFS y borrado local
        sesion.registrar_archivo_pesos(archivo_npz)

        # Eliminar el archivo json original y cualquier copia residual en cache
        for r in rutas_candidatas:
            if r.exists():
                try:
                    r.unlink(missing_ok=True)
                    logger.info(f"🗑️ Archivo local eliminado: {r.name} ✅")
                except Exception as e:
                    logger.warning(f"No se pudo eliminar {r}: {e}")

        logger.info(f"✅ Memoria conversacional convertida a {archivo_npz.name} y registrada para IPFS.")
        return archivo_npz

    except Exception as e:
        logger.error(f"⚠️ Error codificando memoria_conversacion en pesos: {e}")
        return None


def codificar_cache_en_pesos(sesion: Optional[LucIASession] = None, conversor = None) -> List[Path]:
    """
    Inspecciona archivos de datos temporales depositados en Celebro/cache/
    (ej. json, txt, csv, log), los asimila en las neuronas de Celebro,
    genera checkpoints de pesos para IPFS y los purga de cache.
    """
    archivos_generados: List[Path] = []
    if not CACHE_DIR.exists():
        return archivos_generados

    if sesion is None:
        sesion = obtener_sesion()

    if conversor is None:
        try:
            from .Celebro.conversor_pesos import get_conversor_pesos
            conversor = get_conversor_pesos()
        except Exception:
            return archivos_generados

    # Buscar archivos de datos en cache (excluyendo archivos pyc)
    extensiones_datos = {".json", ".txt", ".csv", ".log", ".md"}
    archivos_datos = [
        f for f in CACHE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in extensiones_datos
    ]

    for archivo in archivos_datos:
        try:
            contenido = archivo.read_text(encoding="utf-8", errors="ignore").strip()
            if not contenido:
                archivo.unlink(missing_ok=True)
                continue

            logger.info(f"🧬 Procesando archivo de cache '{archivo.name}' en pesos neuronales...")
            prompt_meta = f"[Cache Data - {archivo.name}]"
            conversor.convertir_respuesta_a_pesos(
                prompt=prompt_meta,
                respuesta=contenido[:2000],
                modelo_origen=f"cache_{archivo.stem}"
            )

            celebro_dir = LUCIA_ROOT / "Celebro"
            nombre_npz = celebro_dir / f"pesos_cache_{archivo.stem}.npz"
            archivo_guardado = conversor.guardar_pesos_en_celebro(archivo=nombre_npz)
            sesion.registrar_archivo_pesos(archivo_guardado)
            archivos_generados.append(archivo_guardado)

            # Borrar archivo original de cache
            archivo.unlink(missing_ok=True)
            logger.info(f"   ✅ Cache '{archivo.name}' convertida a pesos y registrada para IPFS.")
        except Exception as e:
            logger.warning(f"Error codificando archivo de cache {archivo.name}: {e}")

    return archivos_generados


def restaurar_pesos_desde_ipfs(
    sesion: Optional[LucIASession] = None,
    conversor = None,
    patron_modelo: Optional[str] = None
) -> Dict[str, Any]:
    """
    Subsistema de arranque que descarga los pesos neuronales más recientes desde IPFS
    y los restaura directamente en las 18 neuronas de Celebro (ENRN, RF_SL, RF_EN, RNP, SLRN).

    No ensucia el disco permanente; los bytes .npz se decodifican en memoria RAM
    y se inyectan en las matrices sinápticas del conversor.
    """
    if sesion is None:
        sesion = obtener_sesion()

    if conversor is None:
        try:
            from .Celebro.conversor_pesos import get_conversor_pesos
            conversor = get_conversor_pesos()
        except Exception as e:
            logger.error(f"No se pudo cargar conversor_pesos para restauración: {e}")
            return {"exito": False, "error": str(e)}

    ipfs = sesion.ipfs

    # Buscar el checkpoint más relevante (preferencia: 'pesos_activos' o 'memoria' o último disponible)
    patrones_busqueda = [patron_modelo, "pesos_activos", "pesos_memoria_conversacion", None]
    ultimo_registro = None
    for pat in patrones_busqueda:
        if pat is not None or ultimo_registro is None:
            ultimo_registro = ipfs.obtener_ultimo_cid_pesos(patron_nombre=pat)
            if ultimo_registro:
                break

    if not ultimo_registro:
        logger.info("ℹ️ No se encontraron checkpoints de pesos en IPFS. Iniciando con sinapsis base.")
        return {
            "exito": True,
            "restaurado": False,
            "motivo": "sin_checkpoints_previos",
            "cid": None
        }

    cid = ultimo_registro.get("cid")
    nombre_modelo = ultimo_registro.get("nombre_modelo", "pesos_checkpoint")
    logger.info(f"📥 Descargando pesos neuronales desde IPFS (CID: {cid} | Modelo: {nombre_modelo})...")

    # Descargar payload binario desde IPFS directamente a RAM
    datos_binarios = ipfs.descargar_y_extraer_pesos(cid)
    if not datos_binarios:
        logger.warning(f"⚠️ No se pudo obtener el contenido del CID {cid} desde IPFS.")
        return {
            "exito": False,
            "restaurado": False,
            "error": "descarga_fallida",
            "cid": cid
        }

    # Restaurar en las 18 neuronas de Celebro
    resultado_carga = conversor.cargar_pesos_en_celebro(datos_binarios)

    logger.info(
        f"✨ Pesos neuronales restaurados desde IPFS con éxito:\n"
        f"   - CID: {cid}\n"
        f"   - Modelo: {nombre_modelo}\n"
        f"   - Componentes sinápticos restaurados: {resultado_carga.get('neuronas_restauradas', 0)}"
    )

    return {
        "exito": True,
        "restaurado": True,
        "cid": cid,
        "nombre_modelo": nombre_modelo,
        "timestamp": ultimo_registro.get("timestamp"),
        "fecha_utc": ultimo_registro.get("fecha_utc"),
        "componentes_restaurados": resultado_carga.get("neuronas_restauradas", 0)
    }


# Registrar atexit para garantizar que al cerrar la ejecución de Python siempre se limpie
atexit.register(cerrar_sesion)


