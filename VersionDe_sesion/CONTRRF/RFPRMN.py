"""CLI para el ciclo de vida del código de WoldVirtualP2P3D: gestión de
carpetas/archivos del proyecto, sincronización PRY <-> RFC y diagnóstico
de refactorización asistido por LucIA (modelo local vía Ollama)."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

__version__ = "2.0.0"

# --- Estilo de terminal ------------------------------------------------------

_COLOR_ACTIVO = sys.stdout.isatty() and "NO_COLOR" not in os.environ


class Estilo:
    """Códigos ANSI usados para dar formato a la salida en terminal.

    Se desactivan automáticamente si la salida no es una terminal interactiva
    o si la variable de entorno ``NO_COLOR`` está definida.
    """

    _CODIGOS = {
        "reset": "0", "negrita": "1", "tenue": "2",
        "cian": "36", "verde": "32", "amarillo": "33",
        "rojo": "31", "magenta": "35", "azul": "34",
    }

    def __getattr__(self, nombre: str) -> str:
        if not _COLOR_ACTIVO:
            return ""
        return f"\033[{self._CODIGOS[nombre]}m"


E = Estilo()
ANCHO_TERMINAL = shutil.get_terminal_size(fallback=(80, 24)).columns


def _caja(titulo: str, filas: list[tuple[str, str]], color: str = "cian") -> str:
    """Dibuja una caja con bordes Unicode a partir de pares (etiqueta, texto)."""
    ancho_etiqueta = max((len(etiqueta) for etiqueta, _ in filas), default=0)
    ancho_interior = min(
        max(len(titulo), *(ancho_etiqueta + 3 + len(texto) for etiqueta, texto in filas)) + 2,
        ANCHO_TERMINAL - 2,
    )
    col = getattr(E, color)
    linea_top = f"{col}╭{'─' * ancho_interior}╮{E.reset}"
    linea_titulo = f"{col}│{E.reset} {E.negrita}{titulo.ljust(ancho_interior - 1)}{E.reset}{col}│{E.reset}"
    linea_sep = f"{col}├{'─' * ancho_interior}┤{E.reset}"
    linea_bottom = f"{col}╰{'─' * ancho_interior}╯{E.reset}"

    cuerpo = []
    for etiqueta, texto in filas:
        contenido = f" {E.negrita}{etiqueta.ljust(ancho_etiqueta)}{E.reset}  {texto}"
        relleno = ancho_interior - (ancho_etiqueta + len(texto) + 3)
        cuerpo.append(f"{col}│{E.reset}{contenido}{' ' * max(relleno, 0)}{col}│{E.reset}")

    return "\n".join([linea_top, linea_titulo, linea_sep, *cuerpo, linea_bottom])


def _panel(titulo: str, texto: str, color: str = "magenta") -> str:
    """Dibuja un panel de ancho fijo con ``texto`` envuelto en párrafos."""
    ancho = min(ANCHO_TERMINAL - 4, 78)
    col = getattr(E, color)
    encabezado = f"{col}┌─ {E.negrita}{titulo}{E.reset}{col} {'─' * max(ancho - len(titulo) - 3, 0)}{E.reset}"
    cuerpo = "\n".join(
        f"{col}│{E.reset} {linea}" for linea in textwrap.wrap(texto, width=ancho) or [""]
    )
    pie = f"{col}└{'─' * (ancho + 2)}{E.reset}"
    return f"{encabezado}\n{cuerpo}\n{pie}"


# --- Constantes y configuración --------------------------------------------

#: Respuestas que se interpretan como una confirmación afirmativa.
RESPUESTAS_AFIRMATIVAS = frozenset({"s", "si", "sí", "y", "yes"})

#: Tamaño máximo (en bytes) admitido para el análisis local de un archivo.
LIMITE_TAMANO_ANALISIS = 2_000_000

#: Extensiones de archivo admitidas por ``comando_crear_archivo``.
EXTENSIONES_POR_LENGUAJE: dict[str, str] = {
    "python": ".py", "py": ".py",
    "csharp": ".cs", "cs": ".cs",
    "html": ".html",
    "css": ".css",
    "javascript": ".js", "js": ".js",
    "java": ".java",
    "cpp": ".cpp", "c++": ".cpp", "c": ".c",
    "php": ".php",
    "ruby": ".rb",
    "go": ".go",
    "rust": ".rs",
    "typescript": ".ts", "ts": ".ts",
    "sh": ".sh",
    "bat": ".bat",
}

#: Patrón usado para detectar marcadores TODO/FIXME en comentarios.
PATRON_MARCADOR_PENDIENTE = re.compile(r"(?i)(?:#|//|/\*)\s*(?:todo|fixme)\b")


class ErrorProyecto(Exception):
    """Error base para fallos controlados de esta herramienta."""


class RaizProyectoNoEncontrada(ErrorProyecto):
    """Se lanza cuando no puede localizarse la raíz del proyecto."""


def _imprimir_seguro(mensaje: str) -> None:
    """Print tolerante a consolas cp1252 (sustituye símbolos no representables)."""
    try:
        print(mensaje)
    except UnicodeEncodeError:
        print(mensaje.encode("ascii", "replace").decode("ascii"))


def informar(mensaje: str) -> None:
    """Muestra un mensaje informativo estándar."""
    _imprimir_seguro(f"{E.cian}›{E.reset} {mensaje}")


def exito(mensaje: str) -> None:
    """Muestra un mensaje de operación completada correctamente."""
    _imprimir_seguro(f"{E.verde}✔{E.reset} {mensaje}")


def advertir(mensaje: str) -> None:
    """Muestra una advertencia no bloqueante."""
    _imprimir_seguro(f"{E.amarillo}⚠{E.reset} {mensaje}")


def reportar_error(mensaje: str) -> None:
    """Muestra un error de operación al usuario."""
    _imprimir_seguro(f"{E.rojo}✖{E.reset} {mensaje}")


def pedir_confirmacion(mensaje: str) -> bool:
    """Pide al usuario una confirmación s/n y devuelve ``True`` si aceptó."""
    aviso = f"{E.tenue}(s/n){E.reset}"
    respuesta = input(f"{E.negrita}?{E.reset} {mensaje} {aviso}: ").strip().lower()
    return respuesta in RESPUESTAS_AFIRMATIVAS


# --- Localización de la raíz del proyecto -----------------------------------

def encontrar_raiz(ruta: str | Path) -> str:
    """Localiza la raíz del proyecto, soportando el layout moderno (RFC +
    Sistema_Principal/PRY) y el heredado (Construccion/CPRZ/PRY)."""
    actual = Path(ruta).resolve()
    if actual.is_file():
        actual = actual.parent
    if (actual / "RFC").is_dir() and (actual.parent / "Sistema_Principal" / "PRY").is_dir():
        return str(actual.parent)
    for candidato in (actual, *actual.parents):
        if (candidato / "Construccion" / "CPRZ" / "PRY").is_dir():
            return str(candidato)
    raise RaizProyectoNoEncontrada("No se encontró la raíz del proyecto.")


RAIZ = Path(encontrar_raiz(__file__))
if (RAIZ / "Sistema_Principal" / "PRY").is_dir():
    ORIGEN = RAIZ / "Sistema_Principal" / "PRY"
    DESTINO = RAIZ / "CONTRRF" / "RFC"
else:
    ORIGEN = RAIZ / "Construccion" / "CPRZ" / "PRY"
    DESTINO = RAIZ / "Construccion" / "CONTRRF" / "RFC"

CONFIG_IA_LOCAL = DESTINO / "LC" / "LC" / "modelosIAlocal" / "IAlocal.json"

#: Rutas de carpetas creadas durante la sesión, relativas a la raíz.
carpetas_creadas: list[str] = []


# --- Utilidades de sistema de archivos --------------------------------------

def hacer_escribible(ruta: str) -> None:
    """Otorga permiso de escritura a ``ruta`` de forma recursiva."""
    if os.path.islink(ruta):
        return
    os.chmod(ruta, os.stat(ruta).st_mode | stat.S_IWRITE)
    if os.path.isdir(ruta):
        with os.scandir(ruta) as entradas:
            for entrada in entradas:
                hacer_escribible(entrada.path)


def eliminar_ruta(ruta: str) -> None:
    """Elimina un archivo o un árbol de directorios completo."""
    if os.path.isdir(ruta) and not os.path.islink(ruta):
        shutil.rmtree(ruta)
    else:
        os.remove(ruta)


def borrar_contenido(carpeta: str) -> None:
    """Vacía ``carpeta``, reintentando tras corregir permisos si es necesario."""
    for item in os.listdir(carpeta):
        ruta = os.path.join(carpeta, item)
        try:
            eliminar_ruta(ruta)
        except PermissionError:
            hacer_escribible(ruta)
            eliminar_ruta(ruta)


def copiar_contenido(origen: str, destino: str) -> None:
    """Copia recursivamente el contenido de ``origen`` dentro de ``destino``."""
    os.makedirs(destino, exist_ok=True)
    for item in os.listdir(origen):
        origen_item = os.path.join(origen, item)
        destino_item = os.path.join(destino, item)
        if os.path.isdir(origen_item) and not os.path.islink(origen_item):
            shutil.copytree(origen_item, destino_item, dirs_exist_ok=True)
        else:
            shutil.copy2(origen_item, destino_item)


def encontrar_contenido_actual(carpeta: str) -> str:
    """Desciende por carpetas ``actualizacion_*`` anidadas hasta el contenido real."""
    actual = carpeta
    while True:
        contenido = os.listdir(actual)
        if len(contenido) != 1:
            return actual
        hijo = contenido[0]
        ruta = os.path.join(actual, hijo)
        if not hijo.startswith("actualizacion_") or not os.path.isdir(ruta):
            return actual
        actual = ruta


# --- Comandos de gestión del proyecto ---------------------------------------

def comando_refactorizar() -> None:
    """Copia el contenido actual de PRY hacia RFC, previa confirmación."""
    if not ORIGEN.is_dir():
        reportar_error(f"Origen no encontrado: {ORIGEN}")
        return

    advertir("esta acción reemplazará todo el contenido de RFC.")
    if not pedir_confirmacion("¿Desea continuar?"):
        informar("Operación cancelada.")
        return

    try:
        os.makedirs(DESTINO, exist_ok=True)
        origen = encontrar_contenido_actual(str(ORIGEN))
        borrar_contenido(str(DESTINO))
        copiar_contenido(origen, str(DESTINO))
        exito(f"Contenido copiado de {origen} a {DESTINO}")
    except OSError as error:
        reportar_error(f"Error al copiar el contenido: {error}")


def crear_carpeta(ruta: str, nombre: str) -> str:
    """Crea (si no existe) ``nombre`` dentro de ``ruta`` y devuelve la ruta final."""
    carpeta = os.path.join(ruta, nombre)
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
    return carpeta


def comando_crear_carpeta() -> None:
    """Solicita interactivamente una ruta y un nombre, y crea la carpeta."""
    ruta = input("Ruta donde crear: ")
    nombre = input("Nombre de la carpeta: ")
    carpeta = crear_carpeta(ruta, nombre)
    exito(f"Carpeta creada en: {carpeta}")
    relativa = os.path.relpath(carpeta, RAIZ)
    if relativa not in carpetas_creadas:
        carpetas_creadas.append(relativa)


def comando_crear_archivo() -> None:
    """Solicita interactivamente ruta, nombre y lenguaje, y crea un archivo vacío."""
    ruta = input("Ruta donde crear: ").strip()
    nombre = input("Nombre del archivo (sin extensión): ").strip()
    lenguaje = input("Lenguaje de programación (ej: python, csharp, html): ").strip().lower()

    extension = EXTENSIONES_POR_LENGUAJE.get(lenguaje, f".{lenguaje}")
    if not nombre.endswith(extension):
        nombre += extension

    archivo = os.path.join(ruta, nombre)
    os.makedirs(os.path.dirname(archivo), exist_ok=True)
    if not os.path.exists(archivo):
        open(archivo, "w", encoding="utf-8").close()
    exito(f"Archivo creado en: {archivo}")


def actualizar() -> None:
    """Promueve el contenido de RFC a PRY como una nueva versión histórica."""
    if not ORIGEN.is_dir():
        reportar_error(f"Origen no encontrado: {ORIGEN}")
        return
    if not DESTINO.is_dir() or not os.listdir(DESTINO):
        reportar_error("No hay contenido refactorizado para actualizar.")
        return

    advertir("PRY será reemplazado por una nueva versión de RFC.")
    if not pedir_confirmacion("¿Seguro que deseas continuar?"):
        informar("Operación cancelada.")
        return

    try:
        origen = encontrar_contenido_actual(str(DESTINO))
        borrar_contenido(str(ORIGEN))
        numero = uuid.uuid4().hex[:8]
        destino = os.path.join(ORIGEN, f"actualizacion_{numero}")
        copiar_contenido(origen, destino)
        borrar_contenido(str(DESTINO))
        exito(f"Actualización completada en: {destino}")
        informar(f"Contenido eliminado de: {DESTINO}")
    except OSError as error:
        reportar_error(f"Error al actualizar: {error}")


# --- Análisis estático de código --------------------------------------------

@dataclass
class AnalisisArchivo:
    """Resultado del análisis estático de un archivo local."""

    ruta: str
    extension: str
    lineas: int = 0
    lineas_vacias: int = 0
    lineas_largas: int = 0
    marcadores_pendientes: int = 0
    funciones: int = 0
    clases: int = 0
    importaciones: int = 0
    funciones_sin_docstring: int = 0
    bloques_except: int = 0
    except_genericos: int = 0
    argumentos_mutables: int = 0
    puntos_decision: int = 0
    sintaxis_valida: Optional[bool] = None
    detalle_sintaxis: str = ""

    def a_dict(self) -> dict:
        """Convierte el análisis en un diccionario serializable a JSON."""
        return asdict(self)


def analizar_archivo_local(ruta: Path) -> AnalisisArchivo:
    """Analiza el archivo: métricas textuales para cualquier extensión y,
    para ``.py``, un recorrido AST (funciones, clases, docstrings ausentes,
    excepciones genéricas, argumentos mutables) útil para planear el cambio."""
    ruta = ruta.resolve()
    contenido = ruta.read_text(encoding="utf-8-sig", errors="replace")
    lineas = contenido.splitlines()

    analisis = AnalisisArchivo(
        ruta=str(ruta),
        extension=ruta.suffix.lower(),
        lineas=len(lineas),
        lineas_vacias=sum(not linea.strip() for linea in lineas),
        lineas_largas=sum(len(linea) > 100 for linea in lineas),
        marcadores_pendientes=sum(
            len(PATRON_MARCADOR_PENDIENTE.findall(linea)) for linea in lineas
        ),
    )
    if analisis.extension != ".py":
        return analisis

    try:
        arbol = ast.parse(contenido, filename=str(ruta))
        analisis.sintaxis_valida = True
    except SyntaxError as error:
        analisis.sintaxis_valida = False
        analisis.detalle_sintaxis = f"línea {error.lineno}, columna {error.offset}: {error.msg}"
        return analisis

    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
            analisis.funciones += 1
            if ast.get_docstring(nodo) is None:
                analisis.funciones_sin_docstring += 1
            valores_por_defecto = list(nodo.args.defaults) + [
                valor for valor in nodo.args.kw_defaults if valor is not None
            ]
            analisis.argumentos_mutables += sum(
                isinstance(valor, (ast.List, ast.Dict, ast.Set)) for valor in valores_por_defecto
            )
        elif isinstance(nodo, ast.ClassDef):
            analisis.clases += 1
        elif isinstance(nodo, (ast.Import, ast.ImportFrom)):
            analisis.importaciones += 1
        elif isinstance(nodo, ast.ExceptHandler):
            analisis.bloques_except += 1
            if nodo.type is None:
                analisis.except_genericos += 1
        elif isinstance(nodo, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.BoolOp)):
            analisis.puntos_decision += 1

    return analisis


def crear_plan_refactorizacion(analisis: AnalisisArchivo) -> list[str]:
    """Deriva un plan de pasos concretos a partir de un :class:`AnalisisArchivo`."""
    plan: list[str] = []

    if analisis.sintaxis_valida is False:
        plan.append("Corregir primero el error de sintaxis y agregar una prueba que lo detecte.")
    if analisis.funciones_sin_docstring:
        plan.append(
            f"Documentar las {analisis.funciones_sin_docstring} funciones que no declaran su propósito."
        )
    if analisis.argumentos_mutables:
        plan.append("Eliminar argumentos mutables predeterminados para evitar estado compartido.")
    if analisis.except_genericos or analisis.bloques_except:
        plan.append("Reemplazar capturas generales por errores específicos y contexto de recuperación.")
    if analisis.puntos_decision > 20 or analisis.lineas > 400:
        plan.append("Dividir funciones largas y separar responsabilidades en componentes cohesivos.")
    if analisis.lineas_largas:
        plan.append("Reorganizar las líneas extensas para legibilidad sin cambiar el comportamiento.")
    if analisis.marcadores_pendientes:
        plan.append("Resolver o documentar los marcadores TODO y FIXME antes del cambio.")
    if analisis.extension != ".py":
        plan.append("Definir pruebas de contrato para el lenguaje y conservar la API existente.")

    plan.append("Establecer una línea base con pruebas, lint y métricas antes de modificar el archivo.")
    plan.append("Aplicar cambios pequeños y reversibles, uno por responsabilidad.")
    plan.append("Ejecutar pruebas, lint y typecheck; comparar el resultado con la línea base.")
    return plan


# --- Diagnóstico asistido por LucIA (IA local) ------------------------------

def _crear_diagnostico_respaldo(analisis: AnalisisArchivo) -> str:
    """Genera un diagnóstico determinista cuando el motor de IA local no responde."""
    nombre = Path(analisis.ruta).name
    if analisis.sintaxis_valida is False:
        detalle = f"detecté un error de sintaxis en {analisis.detalle_sintaxis}"
    else:
        detalles = [
            f"{analisis.lineas} líneas",
            f"{analisis.funciones} funciones",
            f"{analisis.clases} clases",
        ]
        if analisis.funciones_sin_docstring:
            detalles.append(f"{analisis.funciones_sin_docstring} funciones sin documentar")
        if analisis.argumentos_mutables:
            detalles.append("argumentos mutables riesgosos")
        if analisis.except_genericos:
            detalles.append("capturas de excepciones demasiado generales")
        detalle = "identifiqué " + ", ".join(detalles)

    return (
        f"Yo, LucIA, revisé {nombre} y {detalle}. "
        "El archivo es un punto válido para refactorizar si primero fijamos su comportamiento "
        "actual y verificamos cada cambio con pruebas."
    )


def _cargar_configuracion_ia_local() -> tuple[str, str]:
    """Lee ``IAlocal.json`` y valida que el endpoint apunte a este mismo equipo."""
    with open(CONFIG_IA_LOCAL, "r", encoding="utf-8") as archivo_config:
        configuracion = json.load(archivo_config)

    endpoint = str(configuracion.get("endpoint", "http://localhost:11434")).rstrip("/")
    uri_local = urllib.parse.urlparse(endpoint)
    if uri_local.scheme not in ("http", "https") or uri_local.hostname not in (
        "localhost", "127.0.0.1", "::1",
    ):
        raise ValueError("El endpoint de IA local debe estar en este equipo.")

    modelo = str(configuracion.get("default_model", "cogito:3b"))
    return endpoint, modelo


def consultar_ia_local(contenido: str, analisis: AnalisisArchivo) -> str:
    """Pide un diagnóstico a LucIA vía un modelo local (Ollama); si el motor
    no responde o la configuración es inválida, cae en el respaldo local."""
    prompt_sistema = (
        "Eres LucIA, una programadora que diagnostica código. Responde en español, "
        "en primera persona y en máximo 90 palabras. Describe solo hallazgos verificables "
        "del archivo, sus riesgos y el primer paso seguro. No generes código."
    )
    try:
        endpoint, modelo = _cargar_configuracion_ia_local()
        contexto = (
            f"Archivo: {Path(analisis.ruta).name}\n"
            f"Métricas: {json.dumps(analisis.a_dict(), ensure_ascii=False)}\n"
            f"Código:\n{contenido[:12000]}"
        )
        carga = {
            "model": modelo,
            "messages": [
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": contexto},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 300},
        }
        peticion = urllib.request.Request(
            f"{endpoint}/api/chat",
            data=json.dumps(carga).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(peticion, timeout=30) as respuesta:
            resultado = json.loads(respuesta.read().decode("utf-8"))

        if not isinstance(resultado, dict):
            raise ValueError("La respuesta de IA local no es válida.")
        mensaje = resultado.get("message")
        if not isinstance(mensaje, dict):
            raise ValueError("La respuesta de IA local no contiene un mensaje.")

        diagnostico = " ".join(str(mensaje.get("content", "")).split()[:100]).strip()
        return diagnostico or _crear_diagnostico_respaldo(analisis)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError) as error:
        advertir(f"Motor IA local no disponible ({error}); se usa el diagnóstico local de LucIA.")
        return _crear_diagnostico_respaldo(analisis)


ConsultorIA = Callable[[str, AnalisisArchivo], str]


def comando_ia_local(consultor: Optional[ConsultorIA] = None) -> bool:
    """Analiza el archivo indicado por el usuario y muestra el diagnóstico de
    LucIA; ``consultor`` permite inyectar otra implementación (p. ej. pruebas)."""
    entrada = input("Ruta del archivo a refactorizar: ").strip().strip("\"'")
    if not entrada:
        reportar_error("No se adjuntó ninguna ruta.")
        return False

    ruta = Path(os.path.expandvars(os.path.expanduser(entrada))).resolve()
    if not ruta.exists():
        reportar_error(f"El archivo no existe: {ruta}")
        return False
    if not ruta.is_file():
        reportar_error(f"La ruta no corresponde a un archivo: {ruta}")
        return False

    try:
        if ruta.stat().st_size > LIMITE_TAMANO_ANALISIS:
            reportar_error("El archivo supera el límite local de 2 MB.")
            return False
        contenido = ruta.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as error:
        reportar_error(f"No se pudo leer el archivo: {error}")
        return False

    informar(f"Archivo adjunto: {ruta}")
    informar(f"Tamaño: {ruta.stat().st_size} bytes | Líneas: {len(contenido.splitlines())}")
    if not pedir_confirmacion("¿Desea que se analice el código y se cree un plan de refactorización?"):
        informar("Análisis cancelado.")
        return False

    try:
        analisis = analizar_archivo_local(ruta)
        generador = consultor or consultar_ia_local
        diagnostico = generador(contenido, analisis)
        plan = crear_plan_refactorizacion(analisis)
    except (OSError, ValueError, TypeError) as error:
        reportar_error(f"El análisis local falló: {error}")
        return False

    print()
    print(_panel("LucIA · Diagnóstico", diagnostico.strip()))
    print(f"\n{E.negrita}Plan de refactorización{E.reset}")
    for indice, paso in enumerate(plan, start=1):
        print(f"  {E.cian}{indice}.{E.reset} {paso}")
    return True


# --- Refactorización con modelo local de código ("ds ialocal") ------------------
# Este bloque implementa el comando "ds ialocal": al dar Intro tras pedir el archivo
# descarga de inmediato un modelo de IA para código (no repetido, según el .md de
# registro), refactoriza, genera una respuesta interna a LucIA, LucIA lo apunta en el
# .md con sus propias palabras junto al nombre del modelo, y después se borra el
# modelo. Así cada archivo usa un modelo distinto sin repetir IAlocal.

#: Candidatos de código ligeros (HuggingFace/Ollama) aptos para 8 VRAM / 8 RAM.
#: Carpeta donde se descargan los pesos GGUF desde HuggingFace.
DIR_MODELOS_LOCAL = DESTINO / "LC" / "LC" / "modelosIAlocal" / "IAlocalDESCARGADA"
#: Repo/archivo GGUF pequeño de código (Qwen2.5-Coder 1.5B Q4, ~1 GB, apto 8 RAM).
HF_REPO_CODIGO = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
HF_ARCHIVO_CODIGO = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

#: Candidatos de código ligeros (HuggingFace/Ollama) aptos para 8 VRAM / 8 RAM.
MODELOS_CODIGO_LIGEROS = (
    "qwen2.5-coder:7b",
    "starcoder2:7b",
    "codellama:7b",
    "deepseek-coder:6.7b",
)

#: Límite de líneas por archivo refactorizado (regla 400/450).
LIMITE_LINEAS_REFACTOR = 450

REGISTRO_REFACTOR_MD = DESTINO / "LC" / "LC" / "modelosIAlocal" / "registro_refactor.md"


def _descargar_peso_hf(destino: Path | None = None) -> Path:
    """Descarga el GGUF de código desde HuggingFace DIRECTO en modelosIAlocal/IAlocalDESCARGADA.

    Stream con progreso visible (MB / %): el archivo crece en la carpeta durante
    la descarga para que se vea el modelo y su proceso en el explorador/terminal.
    """
    from huggingface_hub import hf_hub_url
    dest = Path(destino) if destino else DIR_MODELOS_LOCAL
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / HF_ARCHIVO_CODIGO
    parcial = dest / (HF_ARCHIVO_CODIGO + ".part")
    if final.exists() and final.stat().st_size > 0:
        informar(f"Peso ya presente (sin descarga): {final} ({final.stat().st_size} bytes)")
        return final
    # Si quedó un .part completo de una descarga anterior, adoptarlo.
    if parcial.exists() and parcial.stat().st_size > 1000 * 1024 * 1024:
        try:
            parcial.replace(final)
            exito(f"Modelo HF recuperado: {final} ({final.stat().st_size} bytes)")
            return final
        except OSError:
            informar(f"Peso visible en: {parcial} ({parcial.stat().st_size} bytes)")
            return parcial
    url = hf_hub_url(repo_id=HF_REPO_CODIGO, filename=HF_ARCHIVO_CODIGO)
    _imprimir_seguro(f"[DESCARGA] {HF_REPO_CODIGO}/{HF_ARCHIVO_CODIGO}")
    _imprimir_seguro(f"[DESCARGA] URL: {url}")
    _imprimir_seguro(f"[DESCARGA] Destino visible: {final}")
    req = urllib.request.Request(url, headers={"User-Agent": "WoldVirtualP2P3D"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", "0") or 0)
        _imprimir_seguro(f"[DESCARGA] Tamano total: {total / 1048576:.1f} MB")
        bajados = 0
        ultimo_pct = -1
        with open(final, "wb") as f:
            while True:
                bloque = resp.read(4 * 1024 * 1024)
                if not bloque:
                    break
                f.write(bloque)
                bajados += len(bloque)
                if total > 0:
                    pct = int(bajados * 100 / total)
                    if pct != ultimo_pct and (pct % 5 == 0 or pct == 100):
                        ultimo_pct = pct
                        _imprimir_seguro(
                            f"[DESCARGA] {bajados / 1048576:.1f}/{total / 1048576:.1f} MB ({pct}%)")
                elif bajados % (100 * 1048576) < 4 * 1024 * 1024:
                    _imprimir_seguro(f"[DESCARGA] {bajados / 1048576:.1f} MB...")
    # Limpia resto .part antiguo si ya no está bloqueado.
    try:
        if parcial.exists() and final.exists() and parcial.stat().st_size == final.stat().st_size:
            parcial.unlink()
    except OSError:
        pass
    exito(f"Modelo HF descargado: {final} ({final.stat().st_size} bytes)")
    return final


def _modelos_ollama_instalados() -> set[str]:
    """Devuelve los modelos ya descargados en Ollama (vacío si no responde)."""
    try:
        endpoint, _ = _cargar_configuracion_ia_local()
        with urllib.request.urlopen(f"{endpoint}/api/tags", timeout=10) as resp:
            datos = json.loads(resp.read().decode("utf-8"))
        instalados = set()
        for m in datos.get("models", []):
            if not isinstance(m, dict):
                continue
            for clave in ("name", "model"):
                nombre = str(m.get(clave, "") or "").strip().lower()
                if nombre:
                    instalados.add(nombre)
                    # "qwen2.5-coder:7b" y "qwen2.5-coder:latest" deben matchear por base.
                    instalados.add(nombre.split(":")[0])
        return instalados
    except Exception as error:
        advertir(f"No se pudo listar modelos Ollama ({error}).")
        return set()


def _leer_candidatos_codigo() -> list[str]:
    """Lee los candidatos de IAlocal.json (models_available) + lista de respaldo."""
    candidatos: list[str] = []
    try:
        with open(CONFIG_IA_LOCAL, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for m in cfg.get("models_available", []):
            nombre = str(m).strip()
            # Solo modelos de código o pequeños aptos para 8 VRAM / 8 RAM.
            if nombre and nombre not in candidatos:
                candidatos.append(nombre)
    except (OSError, ValueError) as error:
        advertir(f"No se pudo leer candidatos de IAlocal.json ({error}).")
    for respaldo in MODELOS_CODIGO_LIGEROS:
        if respaldo not in candidatos:
            candidatos.append(respaldo)
    # Prioriza modelos de código primero.
    candidatos.sort(key=lambda n: 0 if "coder" in n.lower() or "code" in n.lower() else 1)
    return candidatos


def _descargar_modelo_ollama(endpoint: str, modelo: str) -> bool:
    """Descarga ``modelo`` vía /api/pull leyendo el stream NDJSON; True si OK."""
    carga = json.dumps({"model": modelo, "stream": True}).encode("utf-8")
    pet = urllib.request.Request(
        f"{endpoint}/api/pull", data=carga,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(pet, timeout=1800) as resp:
            # La API devuelve NDJSON por líneas: hay que consumirlo hasta "completed".
            while True:
                linea = resp.readline()
                if not linea:
                    break
                try:
                    evento = json.loads(linea.decode("utf-8"))
                except ValueError:
                    continue
                estado = str(evento.get("status", ""))
                if estado:
                    print(f"\r{E.tenue}{modelo}: {estado[:70]}{E.reset}      ", end="", flush=True)
                if evento.get("status") == "success" or "success" in estado.lower():
                    print()
                    return True
                if "error" in evento:
                    print()
                    reportar_error(f"Error al descargar {modelo}: {evento['error']}")
                    return False
        print()
        # Si el stream terminó sin error, verificar con /api/tags.
        return modelo.lower() in _modelos_ollama_instalados()
    except Exception as error:
        print()
        advertir(f"Fallo /api/pull para {modelo} ({error}); se prueba 'ollama pull'.")
    # Respaldo: CLI de Ollama (muestra progreso nativo).
    try:
        proc = subprocess.run(["ollama", "pull", modelo], timeout=1800)
        return proc.returncode == 0 and modelo.lower() in _modelos_ollama_instalados()
    except Exception as error:
        reportar_error(f"No se pudo descargar {modelo}: {error}")
        return False


def _modelos_usados_registro() -> set[str]:
    """Lee el .md de registro y devuelve los modelos ya usados (para no repetir)."""
    try:
        if REGISTRO_REFACTOR_MD.exists():
            return set(re.findall(r"[A-Za-z0-9._:-]+:[A-Za-z0-9._-]+",
                                  REGISTRO_REFACTOR_MD.read_text(encoding="utf-8")))
    except OSError:
        pass
    return set()


def _asegurar_modelo_codigo() -> str:
    """Descarga de inmediato un modelo de código NO usado antes (sin repetir IAlocal)."""
    # 1) Peso HF en modelosIAlocal/IAlocalDESCARGADA (corrige "no se descarga el modelo").
    try:
        _descargar_peso_hf()
    except Exception as error:
        raise ErrorProyecto(f"No se pudo descargar el peso HF ({error}).")
    usados = {u.lower() for u in _modelos_usados_registro()}
    candidatos = [c for c in _leer_candidatos_codigo() if c.lower() not in usados]
    if not candidatos:  # todos usados: se permite reutilizar desde el primero
        candidatos = _leer_candidatos_codigo()
    endpoint, _ = _cargar_configuracion_ia_local()
    for elegido in candidatos:
        informar(f"Descargando modelo de IA para código: {elegido}...")
        if _descargar_modelo_ollama(endpoint, elegido):
            exito(f"Modelo descargado: {elegido}")
            return elegido
        advertir(f"Se probará con el siguiente candidato tras fallar {elegido}.")
    raise ErrorProyecto("Ningún modelo de código pudo descargarse; revisa Ollama y conexión.")


def _borrar_modelo_ollama(endpoint: str, modelo: str) -> None:
    """Borra el modelo local tras registrar, para no acumular IAlocal repetidas."""
    try:
        carga = json.dumps({"model": modelo}).encode("utf-8")
        pet = urllib.request.Request(
            f"{endpoint}/api/delete", data=carga,
            headers={"Content-Type": "application/json"}, method="DELETE")
        with urllib.request.urlopen(pet, timeout=120):
            pass
    except Exception:
        try:
            subprocess.run(["ollama", "rm", modelo], timeout=300)
        except Exception as error:
            advertir(f"No se pudo borrar {modelo} ({error}).")
    informar(f"Modelo local borrado tras el registro: {modelo}")


def _refactorizar_con_modelo(contenido: str, analisis: AnalisisArchivo, modelo: str) -> str:
    """Pide al modelo local el código refactorizado respetando 400/450 líneas."""
    endpoint, _ = _cargar_configuracion_ia_local()
    prompt = (
        "Eres LucIA. Refactoriza el código dado sin cambiar su comportamiento. "
        f"Regla estricta: el archivo resultante debe tener como máximo {LIMITE_LINEAS_REFACTOR} "
        "líneas (ideal 400). Si lo supera, divide en módulos y devuelve solo el archivo "
        "principal refactorizado. Responde SOLO con código, sin explicaciones ni cercas."
    )
    carga = {"model": modelo, "messages": [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Archivo: {Path(analisis.ruta).name}\n{contenido[:15000]}"}],
        "stream": False, "options": {"temperature": 0.2, "num_predict": 4000}}
    pet = urllib.request.Request(
        f"{endpoint}/api/chat", data=json.dumps(carga).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(pet, timeout=600) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    texto = str(res.get("message", {}).get("content", "")).strip()
    # Limpia cercas de código si el modelo las añade.
    texto = re.sub(r"^```[a-zA-Z]*\n|```$", "", texto, flags=re.MULTILINE).strip()
    lineas = texto.splitlines()
    if len(lineas) > LIMITE_LINEAS_REFACTOR:
        advertir(f"El resultado tiene {len(lineas)} líneas; se recorta a {LIMITE_LINEAS_REFACTOR}.")
        texto = "\n".join(lineas[:LIMITE_LINEAS_REFACTOR])
    return texto


def _registrar_refactor_md(ruta: Path, modelo: str, descripcion: str) -> None:
    """Añade al .md de registro qué hizo el modelo y su nombre (evita re-descargas)."""
    REGISTRO_REFACTOR_MD.parent.mkdir(parents=True, exist_ok=True)
    entrada = (f"\n## {ruta.name} — {modelo}\n"
               f"Fecha: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}\n"
               f"Archivo: {ruta}\nDescripción del modelo: {descripcion.strip()}\n")
    with open(REGISTRO_REFACTOR_MD, "a", encoding="utf-8") as f:
        f.write(entrada)
    exito(f"Registro guardado en: {REGISTRO_REFACTOR_MD}")


def _pedir_extra_o_diagnostico(contenido: str, analisis: AnalisisArchivo) -> None:
    """Tras un 'no', ofrece: 1 = el usuario añade algo, 2 = diagnóstico completo."""
    print(f"{E.negrita}¿Quieres tú añadir algo? ¿Necesitas un diagnóstico más completo?{E.reset}")
    print("  1 · Añadir algo propio al plan")
    print("  2 · Diagnóstico más completo")
    opcion = input("Selecciona 1 o 2: ").strip()
    if opcion == "1":
        extra = input("Escribe lo que quieres añadir al plan: ").strip()
        if extra:
            informar(f"Añadido a tu plan: {extra}")
    elif opcion == "2":
        print(_panel("LucIA · Diagnóstico completo",
                     f"Archivo {Path(analisis.ruta).name}: {analisis.lineas} líneas, "
                     f"{analisis.funciones} funciones, {analisis.clases} clases, "
                     f"{analisis.importaciones} importaciones, {analisis.puntos_decision} "
                     f"puntos de decisión, {analisis.lineas_largas} líneas largas, "
                     f"{analisis.marcadores_pendientes} pendientes, sintaxis válida: "
                     f"{analisis.sintaxis_valida}. Recomiendo dividir por responsabilidad, "
                     f"documentar, tipar y cubrir con pruebas antes de tocar nada."))
    else:
        informar("Opción no válida; no se hizo nada más.")


def comando_ds_ialocal(consultor: Optional[ConsultorIA] = None) -> bool:
    """Flujo 'ds ialocal': Intro -> descarga modelo -> refactor -> nota de LucIA en .md -> borra modelo."""
    entrada = input("Archivo al cual hay que refactorizar: ").strip().strip("\"'")
    if not entrada:
        reportar_error("No se indicó ningún archivo.")
        return False
    ruta = Path(os.path.expandvars(os.path.expanduser(entrada))).resolve()
    if not ruta.is_file():
        reportar_error(f"Archivo no válido: {ruta}")
        return False
    try:
        contenido = ruta.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as error:
        reportar_error(f"No se pudo leer: {error}")
        return False
    # 1) Al dar Intro se descarga de inmediato un modelo de IA para código.
    try:
        endpoint, _ = _cargar_configuracion_ia_local()
        modelo = _asegurar_modelo_codigo()
    except (ErrorProyecto, ValueError) as error:
        reportar_error(str(error))
        return False
    # 2) Refactorización con ese modelo (regla 400/450).
    analisis = analizar_archivo_local(ruta)
    try:
        nuevo = _refactorizar_con_modelo(contenido, analisis, modelo)
    except Exception as error:
        reportar_error(f"El modelo local falló: {error}")
        _borrar_modelo_ollama(endpoint, modelo)
        return False
    ruta.write_text(nuevo + "\n", encoding="utf-8")
    exito(f"Archivo refactorizado ({len(nuevo.splitlines())} líneas): {ruta}")
    # 3) Respuesta interna a LucIA: ella describe con sus palabras lo hecho.
    generador = consultor or consultar_ia_local
    analisis_nuevo = analizar_archivo_local(ruta)
    nota_lucia = generador(nuevo, analisis_nuevo)
    # 4) LucIA lo apunta en el .md junto con el nombre de la IA local.
    _registrar_refactor_md(ruta, modelo, nota_lucia)
    # 5) Se borra el modelo para no repetir IAlocal entre archivos.
    _borrar_modelo_ollama(endpoint, modelo)
    return True


# --- Menú interactivo --------------------------------------------------------

def cerrar() -> None:
    """Finaliza la sesión interactiva."""
    informar("Cerrando el sistema...")


def mostrar_menu() -> None:
    """Imprime la lista de comandos disponibles dentro de una caja con bordes."""
    filas = [
        ("crear carpeta", "Crear una carpeta nueva"),
        ("crear archivo", "Crear un archivo nuevo"),
        ("refactorizar", "Copiar el contenido actual de PRY a RFC"),
        ("actualizar", "Reemplazar PRY con la versión de RFC"),
        ("ia local", "Analizar un archivo con LucIA y crear un plan"),
        ("ds ialocal", "Plan de LucIA + refactor con modelo local y registro .md"),
        ("cerrar", "Cerrar el sistema"),
    ]
    print(_caja(f"WoldVirtualP2P3D · Gestor de Proyecto  v{__version__}", filas))
    print()


def ejecutar_comandos() -> None:
    """Bucle principal de la shell interactiva."""
    despachador: dict[str, Callable[[], None]] = {
        "crear carpeta": comando_crear_carpeta,
        "crear archivo": comando_crear_archivo,
        "refactorizar": comando_refactorizar,
        "actualizar": actualizar,
        "ia local": comando_ia_local,
        "ds ialocal": comando_ds_ialocal,
    }

    mostrar_menu()
    while True:
        entrada = input("> ").strip().lower()
        if entrada == "cerrar":
            cerrar()
            break
        accion = despachador.get(entrada)
        if accion is None:
            reportar_error(f"Comando no reconocido: '{entrada}'")
            continue
        accion()


if __name__ == "__main__":
    ejecutar_comandos()