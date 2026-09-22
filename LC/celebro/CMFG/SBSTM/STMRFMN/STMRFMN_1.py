"""STMRFMN_1.py - Fachada eficiente para el monitor y refactorizador HRCNTR.

Conserva la API historica y centraliza carga perezosa, singleton, caches y
telemetria; la logica real sigue delegada en HRCNTR.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from threading import Lock, RLock
from time import perf_counter
from typing import Any, Dict, Final, List, Optional, TypeVar, cast

__version__: Final[str] = "2026.3.1"
MIN_LINEAS: Final[int] = 400
MAX_LINEAS: Final[int] = 450
_HRCNTR_MODULE: Final[str] = "LC.celebro.CMFG.SBSTM.HRCNTR"
_ESTADO_TTL: Final[float] = 1.0
_MONITOR_TTL: Final[float] = 5.0
_T = TypeVar("_T")

# ─── EXCEPCIONES PUBLICAS ─────────────────────────────────────────────────
class STMRFMNError(RuntimeError):
    """Error de coordinacion de STMRFMN."""

class HRCNTRNoDisponibleError(STMRFMNError):
    """HRCNTR no esta disponible en el entorno actual."""

# ─── MODELOS INTERNOS ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class OperacionSTMRFMN:
    """Telemetria de una operacion delegada."""
    nombre: str
    segundos: float
    exito: bool
    detalle: Mapping[str, Any]

@dataclass(frozen=True)
class _BindingSet:
    """Referencias validadas a los simbolos de HRCNTR."""
    gestor_class: type
    get_gestor: Callable[[], Any]
    ejecutar: Callable[..., Dict[str, Any]]
    estado: Callable[[], Dict[str, Any]]
    actualizar: Callable[..., Dict[str, Any]]
    cierre: Callable[[], Dict[str, Any]]
    confirmar: Callable[[], str]

# ─── IMPORTACION PEREZOSA Y VALIDACION ────────────────────────────────────
def _normalizar_overlay(overlay: str = "") -> str:
    """Normaliza una ruta sin cambiar la semantica de HRCNTR."""
    if not overlay:
        return ""
    ruta = Path(overlay).expanduser()
    return str(ruta.resolve()) if ruta.exists() else str(ruta)

def _crear_bindings(modulo: Any) -> _BindingSet:
    """Valida los simbolos minimos exportados por HRCNTR."""
    nombres = (
        "GestorHRCNTR", "get_gestor_hrctnr", "ejecutar_hrctnr",
        "estado_hrctnr", "actualizar_sistema_hrctnr",
        "ciclo_cierre_hrctnr", "confirmar_actualizacion_hrctnr",
    )
    faltantes = [nombre for nombre in nombres if not hasattr(modulo, nombre)]
    if faltantes:
        raise HRCNTRNoDisponibleError(
            f"HRCNTR no exporta: {', '.join(faltantes)}"
        )
    return _BindingSet(
        gestor_class=getattr(modulo, "GestorHRCNTR"),
        get_gestor=getattr(modulo, "get_gestor_hrctnr"),
        ejecutar=getattr(modulo, "ejecutar_hrctnr"),
        estado=getattr(modulo, "estado_hrctnr"),
        actualizar=getattr(modulo, "actualizar_sistema_hrctnr"),
        cierre=getattr(modulo, "ciclo_cierre_hrctnr"),
        confirmar=getattr(modulo, "confirmar_actualizacion_hrctnr"),
    )

def _importar_hrctnr() -> Dict[str, Any]:
    """Importa HRCNTR una sola vez y devuelve una API compatible."""
    try:
        modulo = import_module(_HRCNTR_MODULE)
    except Exception as exc:
        raise HRCNTRNoDisponibleError(f"No se pudo cargar HRCNTR: {exc}") from exc
    bindings = _crear_bindings(modulo)
    return {
        "GestorHRCNTR": bindings.gestor_class,
        "get_gestor_hrctnr": bindings.get_gestor,
        "ejecutar_hrctnr": bindings.ejecutar,
        "estado_hrctnr": bindings.estado,
        "actualizar_sistema_hrctnr": bindings.actualizar,
        "ciclo_cierre_hrctnr": bindings.cierre,
        "confirmar_actualizacion_hrctnr": bindings.confirmar,
        "_bindings": bindings,
    }

# ─── SINGLETON Y CACHE GLOBAL ─────────────────────────────────────────────
_HRCNTR_API: Optional[Dict[str, Any]] = None
_HRCNTR_API_LOCK: Final[Lock] = Lock()
_GESTOR_STMRFMN: Optional["GestorSTMRFMN"] = None
_GESTOR_STMRFMN_LOCK: Final[Lock] = Lock()
GestorHRCNTR: Optional[type] = None

def _get_hrctnr_api() -> Dict[str, Any]:
    """Devuelve la API HRCNTR cargada de forma segura entre hilos."""
    global _HRCNTR_API, GestorHRCNTR
    with _HRCNTR_API_LOCK:
        if _HRCNTR_API is None:
            _HRCNTR_API = _importar_hrctnr()
            GestorHRCNTR = cast(Optional[type], _HRCNTR_API["GestorHRCNTR"])
        return _HRCNTR_API

def _bindings() -> _BindingSet:
    """Obtiene el conjunto de funciones HRCNTR validado."""
    api = _get_hrctnr_api()
    conjunto = api.get("_bindings")
    if isinstance(conjunto, _BindingSet):
        return conjunto
    return _crear_bindings(type("_HRCNTRModulo", (), api))

def _resultado_dict(resultado: Any, operacion: str) -> Dict[str, Any]:
    """Normaliza respuestas sin ocultar su contenido."""
    if isinstance(resultado, Mapping):
        return dict(resultado)
    if resultado is None:
        return {"exito": True, "operacion": operacion}
    return {"exito": True, "operacion": operacion, "resultado": resultado}

def _medir(operacion: str, funcion: Callable[..., _T], *args: Any,
           **kwargs: Any) -> tuple[_T, OperacionSTMRFMN]:
    """Ejecuta una llamada y calcula su duracion."""
    inicio = perf_counter()
    try:
        resultado = funcion(*args, **kwargs)
        return resultado, OperacionSTMRFMN(
            nombre=operacion,
            segundos=round(perf_counter() - inicio, 6),
            exito=True,
            detalle={},
        )
    except Exception as exc:
        raise STMRFMNError(f"{operacion} fallo: {exc}") from exc

# ─── CLASE DE COORDINACION ────────────────────────────────────────────────
class GestorSTMRFMN:
    """Coordina HRCNTR con singleton, cache y metricas.

    La clase no duplica el refactorizador: delega las operaciones reales y
    evita importaciones, escaneos y lecturas repetidas innecesarias.
    """
    def __init__(self, bindings: Optional[_BindingSet] = None,
                 gestor: Optional[Any] = None) -> None:
        self._bindings = bindings
        self._gestor = gestor
        self._lock = RLock()
        self._estado_cache: Optional[Dict[str, Any]] = None
        self._estado_ts = 0.0
        self._monitor_cache: Optional[Dict[str, Any]] = None
        self._monitor_ts = 0.0
        self._ultimo_refactor: Optional[Dict[str, Any]] = None
        self._ultima_operacion: Optional[OperacionSTMRFMN] = None

    @property
    def disponible(self) -> bool:
        """Indica si el singleton de HRCNTR puede obtenerse."""
        try:
            return self.gestor is not None
        except Exception:
            return False

    @property
    def gestor(self) -> Any:
        """Devuelve el singleton real de HRCNTR."""
        if self._gestor is None:
            with self._lock:
                if self._gestor is None:
                    conjunto = self._bindings or _bindings()
                    self._gestor = conjunto.get_gestor()
        if self._gestor is None:
            raise HRCNTRNoDisponibleError("El gestor HRCNTR no esta inicializado.")
        return self._gestor

    @property
    def ultima_operacion(self) -> Optional[OperacionSTMRFMN]:
        """Ultima operacion medida por esta fachada."""
        return self._ultima_operacion

    def _invocar(self, nombre: str, *args: Any, **kwargs: Any) -> Any:
        conjunto = self._bindings or _bindings()
        funcion = getattr(conjunto, nombre)
        resultado, medida = _medir(nombre, funcion, *args, **kwargs)
        self._ultima_operacion = medida
        return resultado

    def _invalidar_caches(self) -> None:
        with self._lock:
            self._estado_cache = None
            self._estado_ts = 0.0
            self._monitor_cache = None
            self._monitor_ts = 0.0

    def limpiar_cache(self) -> None:
        """Liberar los caches locales sin tocar el sistema."""
        self._invalidar_caches()
        self._ultimo_refactor = None

    def estado(self, forzar: bool = False,
               usar_cache: bool = True) -> Dict[str, Any]:
        """Obtiene el estado y reutiliza una lectura reciente."""
        ahora = perf_counter()
        if (usar_cache and not forzar and self._estado_cache is not None
                and ahora - self._estado_ts <= _ESTADO_TTL):
            return dict(self._estado_cache)
        resultado = _resultado_dict(self._invocar("estado"), "estado")
        with self._lock:
            self._estado_cache = resultado
            self._estado_ts = perf_counter()
        return dict(resultado)

    def monitor_sistema(self, forzar: bool = False,
                        usar_cache: bool = True) -> Dict[str, Any]:
        """Monitoriza LC y cachea el escaneo durante cinco segundos."""
        ahora = perf_counter()
        if (usar_cache and not forzar and self._monitor_cache is not None
                and ahora - self._monitor_ts <= _MONITOR_TTL):
            return dict(self._monitor_cache)
        metodo = getattr(self.gestor, "monitor_sistema", None)
        if not callable(metodo):
            raise STMRFMNError("HRCNTR no expone monitor_sistema().")
        inicio = perf_counter()
        try:
            resultado = _resultado_dict(metodo(), "monitor_sistema")
            exito = True
        except Exception as exc:
            resultado = {"exito": False, "error": str(exc)}
            exito = False
        self._ultima_operacion = OperacionSTMRFMN(
            nombre="monitor_sistema",
            segundos=round(perf_counter() - inicio, 6),
            exito=exito,
            detalle={} if exito else {"error": resultado.get("error", "")},
        )
        with self._lock:
            self._monitor_cache = resultado
            self._monitor_ts = perf_counter()
        return dict(resultado)

    def refactorizar(self, overlay: str = "", mostrar_barra: bool = True,
                     usar_cache: bool = False) -> Dict[str, Any]:
        """Ejecuta el refactor y conserva su ultimo reporte exitoso."""
        if usar_cache and not overlay and self._ultimo_refactor is not None:
            return dict(self._ultimo_refactor)
        ruta = _normalizar_overlay(overlay)
        resultado = _resultado_dict(
            self._invocar("ejecutar", overlay=ruta, mostrar_barra=mostrar_barra),
            "refactorizar",
        )
        if resultado.get("exito", False):
            with self._lock:
                self._ultimo_refactor = dict(resultado)
        self._invalidar_caches()
        return dict(resultado)

    def actualizar(self, confirmar: bool = False) -> Dict[str, Any]:
        """Aplica una actualizacion confirmada y actualiza los caches."""
        resultado = _resultado_dict(
            self._invocar("actualizar", confirmar=confirmar), "actualizar"
        )
        if resultado.get("exito", False) or confirmar:
            self._invalidar_caches()
        return dict(resultado)

    def ciclo_cierre(self) -> Dict[str, Any]:
        """Ejecuta refactor, confirmacion y actualizacion en orden oficial."""
        resultado = _resultado_dict(self._invocar("cierre"), "ciclo_cierre")
        self._invalidar_caches()
        return dict(resultado)

    def confirmar_actualizacion(self) -> str:
        """Devuelve el mensaje de confirmacion de HRCNTR."""
        return str(self._invocar("confirmar"))

    def get_gestor(self) -> Any:
        """Alias para codigo que espera un accessor de gestor."""
        return self.gestor

    def listar_modulos(self) -> List[Dict[str, Any]]:
        """Lista modulos usando el inventario de HRCNTR."""
        metodo = getattr(self.gestor, "listar_modulos", None)
        if callable(metodo):
            return list(cast(Iterable[Any], metodo()))
        return list(self.monitor_sistema().get("archivos", []))

    def listar_neuronas(self) -> List[Dict[str, Any]]:
        """Lista neuronas sin repetir el escaneo completo."""
        metodo = getattr(self.gestor, "listar_neuronas", None)
        if callable(metodo):
            return list(cast(Iterable[Any], metodo()))
        return [
            item for item in self.monitor_sistema().get("archivos", [])
            if any(p in str(item.get("ruta", ""))
                   for p in ("red_neuronal", "RF_EN", "RF_SL", "ENRN"))
        ]

    def seleccionar_clases(self) -> List[Dict[str, Any]]:
        """Selecciona clases oversized mediante el gestor HRCNTR."""
        metodo = getattr(self.gestor, "seleccionar_clases", None)
        if callable(metodo):
            return list(cast(Iterable[Any], metodo()))
        seleccion: List[Dict[str, Any]] = []
        for archivo in self.monitor_sistema().get("archivos", []):
            if not archivo.get("sobreescrito"):
                continue
            for clase in archivo.get("clases", []):
                seleccion.append({
                    "modulo_origen": archivo.get("ruta", ""),
                    "clase": clase.get("nombre", "") if isinstance(clase, Mapping)
                    else str(clase),
                    "lineas_modulo": archivo.get("lineas", 0),
                    "ruta_absoluta": archivo.get("absoluta", ""),
                })
        return seleccion

    def listar_subsistemas(self) -> List[Dict[str, Any]]:
        """Devuelve los subsistemas generados en la sesion activa."""
        metodo = getattr(self.gestor, "listar_subsistemas", None)
        return list(cast(Iterable[Any], metodo())) if callable(metodo) else []

    def resumen_sesion(self) -> str:
        """Resume la sesion sin imprimir ni escribir en disco."""
        metodo = getattr(self.gestor, "resumen_sesion", None)
        return str(metodo()) if callable(metodo) else ""

    def diagnostico(self) -> Dict[str, Any]:
        """Reporte rapido de disponibilidad, estado y caches."""
        inicio = perf_counter()
        try:
            estado = self.estado(forzar=True, usar_cache=False)
            return {
                "disponible": True, "version": __version__, "estado": estado,
                "cache_estado": self._estado_cache is not None,
                "cache_monitor": self._monitor_cache is not None,
                "segundos": round(perf_counter() - inicio, 6),
            }
        except Exception as exc:
            return {
                "disponible": False, "version": __version__, "error": str(exc),
                "segundos": round(perf_counter() - inicio, 6),
            }

    def __enter__(self) -> "GestorSTMRFMN":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any,
                 exc_tb: Any) -> bool:
        self.limpiar_cache()
        return False

# ─── ALIAS Y SINGLETON PUBLICO ────────────────────────────────────────────
STMRFMNManager = GestorSTMRFMN

def get_gestor_stmrfmn() -> GestorSTMRFMN:
    """Singleton de la fachada STMRFMN."""
    global _GESTOR_STMRFMN
    with _GESTOR_STMRFMN_LOCK:
        if _GESTOR_STMRFMN is None:
            _GESTOR_STMRFMN = GestorSTMRFMN()
        return _GESTOR_STMRFMN

# ─── API DE COMPATIBILIDAD CON HRCNTR ─────────────────────────────────────
def get_gestor_hrctnr() -> Any:
    """Compatibilidad con la API publica original de HRCNTR."""
    return get_gestor_stmrfmn().gestor

def ejecutar_hrctnr(overlay: str = "", mostrar_barra: bool = True) -> Dict[str, Any]:
    """Compatibilidad con la funcion historica de refactor neural."""
    return get_gestor_stmrfmn().refactorizar(
        overlay=overlay, mostrar_barra=mostrar_barra
    )

def estado_hrctnr() -> Dict[str, Any]:
    """Compatibilidad con la funcion historica de estado."""
    return get_gestor_stmrfmn().estado()

def actualizar_sistema_hrctnr(confirmar: bool = False) -> Dict[str, Any]:
    """Compatibilidad con la funcion historica de actualizacion."""
    return get_gestor_stmrfmn().actualizar(confirmar=confirmar)

def ciclo_cierre_hrctnr() -> Dict[str, Any]:
    """Compatibilidad con la funcion historica de cierre."""
    return get_gestor_stmrfmn().ciclo_cierre()

def confirmar_actualizacion_hrctnr() -> str:
    """Compatibilidad con la funcion historica de confirmacion."""
    return get_gestor_stmrfmn().confirmar_actualizacion()

# ─── FUNCIONES DE USO DIRECTO ─────────────────────────────────────────────
def refactorizar_sistema(overlay: str = "", mostrar_barra: bool = True) -> Dict[str, Any]:
    """Nombre descriptivo para la operacion publica de refactor."""
    return ejecutar_hrctnr(overlay=overlay, mostrar_barra=mostrar_barra)

def monitor_sistema(forzar: bool = False) -> Dict[str, Any]:
    """Acceso directo al monitor cacheado del sistema."""
    return get_gestor_stmrfmn().monitor_sistema(forzar=forzar)

def listar_modulos() -> List[Dict[str, Any]]:
    """Acceso directo al inventario de modulos."""
    return get_gestor_stmrfmn().listar_modulos()

def listar_neuronas() -> List[Dict[str, Any]]:
    """Acceso directo al inventario de neuronas."""
    return get_gestor_stmrfmn().listar_neuronas()

def seleccionar_clases() -> List[Dict[str, Any]]:
    """Acceso directo a la seleccion de clases oversized."""
    return get_gestor_stmrfmn().seleccionar_clases()

def diagnostico_stmrfmn() -> Dict[str, Any]:
    """Diagnostico ligero de la fachada y de su subsistema delegado."""
    return get_gestor_stmrfmn().diagnostico()

def limpiar_cache_stmrfmn() -> None:
    """Liberar los caches de la instancia singleton."""
    get_gestor_stmrfmn().limpiar_cache()

# ─── EXPORTACION PUBLICA ──────────────────────────────────────────────────
__all__: Final[List[str]] = [
    "GestorHRCNTR", "GestorSTMRFMN", "STMRFMNManager", "STMRFMNError",
    "HRCNTRNoDisponibleError", "OperacionSTMRFMN", "get_gestor_stmrfmn",
    "get_gestor_hrctnr", "ejecutar_hrctnr", "estado_hrctnr",
    "actualizar_sistema_hrctnr", "ciclo_cierre_hrctnr",
    "confirmar_actualizacion_hrctnr", "refactorizar_sistema",
    "monitor_sistema", "listar_modulos", "listar_neuronas",
    "seleccionar_clases", "diagnostico_stmrfmn", "limpiar_cache_stmrfmn",
    "_get_hrctnr_api",
]