import ast
import json
import os
import re
import shutil
import stat
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

def encontrar_raiz(ruta):
    actual = Path(ruta).resolve()
    if actual.is_file():
        actual = actual.parent
    if (actual / "RFC").is_dir() and (actual.parent / "Sistema_Principal" / "PRY").is_dir():
        return str(actual.parent)
    for candidato in (actual, *actual.parents):
        if (candidato / "Construccion" / "CPRZ" / "PRY").is_dir():
            return str(candidato)
    raise RuntimeError("No se encontro la raiz del proyecto.")


ROOT = encontrar_raiz(__file__)
RAIZ = Path(ROOT)
if (RAIZ / "Sistema_Principal" / "PRY").is_dir():
    ORIGEN = str(RAIZ / "Sistema_Principal" / "PRY")
    DESTINO = str(RAIZ / "CONTRRF" / "RFC")
else:
    ORIGEN = str(RAIZ / "Construccion" / "CPRZ" / "PRY")
    DESTINO = str(RAIZ / "Construccion" / "CONTRRF" / "RFC")
CONFIG_IA_LOCAL = os.path.join(
    DESTINO, "LC", "LC", "modelosIAlocal", "IAlocal.json"
)
creadas = []

def hacer_escribible(ruta):
    if os.path.islink(ruta):
        return
    os.chmod(ruta, os.stat(ruta).st_mode | stat.S_IWRITE)
    if os.path.isdir(ruta):
        with os.scandir(ruta) as entradas:
            for entrada in entradas:
                hacer_escribible(entrada.path)


def eliminar_ruta(ruta):
    if os.path.isdir(ruta) and not os.path.islink(ruta):
        shutil.rmtree(ruta)
    else:
        os.remove(ruta)

def borrar_contenido(carpeta):
    for item in os.listdir(carpeta):
        ruta = os.path.join(carpeta, item)
        try:
            eliminar_ruta(ruta)
        except PermissionError:
            hacer_escribible(ruta)
            eliminar_ruta(ruta)

def copiar_contenido(origen, destino):
    os.makedirs(destino, exist_ok=True)
    for item in os.listdir(origen):
        origen_item = os.path.join(origen, item)
        destino_item = os.path.join(destino, item)
        if os.path.isdir(origen_item) and not os.path.islink(origen_item):
            shutil.copytree(origen_item, destino_item, dirs_exist_ok=True)
        else:
            shutil.copy2(origen_item, destino_item)

def encontrar_contenido_actual(carpeta):
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

def comando_refactorizar():
    if not os.path.isdir(ORIGEN):
        print(f"Origen no encontrado: {ORIGEN}")
        return

    print("ADVERTENCIA: esta accion reemplazara todo el contenido de RFC.")
    confirmacion = input("¿Desea continuar? (s/n): ").strip().lower()
    if confirmacion not in ("s", "si", "sí", "y", "yes"):
        print("Operacion cancelada.")
        return

    try:
        os.makedirs(DESTINO, exist_ok=True)
        origen = encontrar_contenido_actual(ORIGEN)
        borrar_contenido(DESTINO)
        copiar_contenido(origen, DESTINO)
        print(f"Contenido copiado de {origen} a {DESTINO}")
    except OSError as error:
        print(f"Error al copiar el contenido: {error}")

def crear_carpeta(ruta, nombre):
    carpeta = os.path.join(ruta, nombre)
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
    return carpeta

def comando_crear_carpeta():
    ruta = input("Ruta donde crear: ")
    nombre = input("Nombre de la carpeta: ")
    carpeta = crear_carpeta(ruta, nombre)
    print(f"Carpeta creada en: {carpeta}")
    relativa = os.path.relpath(carpeta, ROOT)
    if relativa not in creadas:
        creadas.append(relativa)

def comando_crear_archivo():
    ruta = input("Ruta donde crear: ").strip()
    nombre = input("Nombre del archivo (sin extension): ").strip()
    lenguaje = input("Lenguaje de programacion (ej: python, csharp, html): ").strip().lower()
    extensiones = {
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
    extension = extensiones.get(lenguaje, f".{lenguaje}")
    nombre = nombre + extension if not nombre.endswith(extension) else nombre
    archivo = os.path.join(ruta, nombre)
    os.makedirs(os.path.dirname(archivo), exist_ok=True)
    if not os.path.exists(archivo):
        open(archivo, "w").close()
    print(f"Archivo creado en: {archivo}")


def actualizar():
    if not os.path.isdir(ORIGEN):
        print(f"Origen no encontrado: {ORIGEN}")
        return
    if not os.path.isdir(DESTINO) or not os.listdir(DESTINO):
        print("No hay contenido refactorizado para actualizar.")
        return

    print("ADVERTENCIA: PRY sera reemplazado por una nueva version de RFC.")
    confirmacion = input("¿Seguro que deseas continuar? (s/n): ").strip().lower()
    if confirmacion not in ("s", "si", "sí", "y", "yes"):
        print("Operacion cancelada.")
        return

    try:
        origen = encontrar_contenido_actual(DESTINO)
        borrar_contenido(ORIGEN)
        numero = uuid.uuid4().hex[:8]
        destino = os.path.join(ORIGEN, f"actualizacion_{numero}")
        copiar_contenido(origen, destino)
        borrar_contenido(DESTINO)
        print(f"Actualización completada en: {destino}")
        print(f"Contenido eliminado de: {DESTINO}")
    except OSError as error:
        print(f"Error al actualizar: {error}")


def analizar_archivo_local(ruta):
    ruta = Path(ruta).resolve()
    contenido = ruta.read_text(encoding="utf-8-sig", errors="replace")
    lineas = contenido.splitlines()
    analisis = {
        "ruta": str(ruta),
        "extension": ruta.suffix.lower(),
        "lineas": len(lineas),
        "lineas_vacias": sum(not linea.strip() for linea in lineas),
        "lineas_largas": sum(len(linea) > 100 for linea in lineas),
        "marcadores_pendientes": sum(
            len(re.findall(r"(?i)(?:#|//|/\*)\s*(?:todo|fixme)\b", linea))
            for linea in lineas
        ),
        "funciones": 0,
        "clases": 0,
        "importaciones": 0,
        "funciones_sin_docstring": 0,
        "bloques_except": 0,
        "except_genericos": 0,
        "argumentos_mutables": 0,
        "puntos_decision": 0,
        "sintaxis_valida": None,
        "detalle_sintaxis": "",
    }
    if analisis["extension"] != ".py":
        return analisis

    try:
        arbol = ast.parse(contenido, filename=str(ruta))
        analisis["sintaxis_valida"] = True
    except SyntaxError as error:
        analisis["sintaxis_valida"] = False
        analisis["detalle_sintaxis"] = (
            f"linea {error.lineno}, columna {error.offset}: {error.msg}"
        )
        return analisis

    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
            analisis["funciones"] += 1
            if ast.get_docstring(nodo) is None:
                analisis["funciones_sin_docstring"] += 1
            argumentos = nodo.args
            valores = list(argumentos.defaults)
            valores.extend(valor for valor in argumentos.kw_defaults if valor is not None)
            analisis["argumentos_mutables"] += sum(
                isinstance(valor, (ast.List, ast.Dict, ast.Set)) for valor in valores
            )
        elif isinstance(nodo, ast.ClassDef):
            analisis["clases"] += 1
        elif isinstance(nodo, (ast.Import, ast.ImportFrom)):
            analisis["importaciones"] += 1
        elif isinstance(nodo, ast.ExceptHandler):
            analisis["bloques_except"] += 1
            if nodo.type is None:
                analisis["except_genericos"] += 1
        elif isinstance(nodo, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.BoolOp)):
            analisis["puntos_decision"] += 1
    return analisis


def crear_plan_refactorizacion(analisis):
    plan = []
    if analisis["sintaxis_valida"] is False:
        plan.append("Corregir primero el error de sintaxis y agregar una prueba que lo detecte.")
    if analisis["funciones_sin_docstring"]:
        plan.append(
            f"Documentar las {analisis['funciones_sin_docstring']} funciones que no declaran su proposito."
        )
    if analisis["argumentos_mutables"]:
        plan.append("Eliminar argumentos mutables predeterminados para evitar estado compartido.")
    if analisis["except_genericos"] or analisis["bloques_except"]:
        plan.append("Reemplazar capturas generales por errores especificos y contexto de recuperacion.")
    if analisis["puntos_decision"] > 20 or analisis["lineas"] > 400:
        plan.append("Dividir funciones largas y separar responsabilidades en componentes cohesivos.")
    if analisis["lineas_largas"]:
        plan.append("Reorganizar las lineas extensas para legibilidad sin cambiar el comportamiento.")
    if analisis["marcadores_pendientes"]:
        plan.append("Resolver o documentar los marcadores TODO y FIXME antes del cambio.")
    if analisis["extension"] != ".py":
        plan.append("Definir pruebas de contrato para el lenguaje y conservar la API existente.")
    plan.append("Establecer una linea base con pruebas, lint y metricas antes de modificar el archivo.")
    plan.append("Aplicar cambios pequenos y reversibles, uno por responsabilidad.")
    plan.append("Ejecutar pruebas, lint y typecheck; comparar el resultado con la linea base.")
    return plan


def _crear_diagnostico_respaldo(analisis):
    nombre = Path(analisis["ruta"]).name
    if analisis["sintaxis_valida"] is False:
        detalle = f"detecté un error de sintaxis en {analisis['detalle_sintaxis']}"
    else:
        detalles = [
            f"{analisis['lineas']} lineas",
            f"{analisis['funciones']} funciones",
            f"{analisis['clases']} clases",
        ]
        if analisis["funciones_sin_docstring"]:
            detalles.append(f"{analisis['funciones_sin_docstring']} funciones sin documentar")
        if analisis["argumentos_mutables"]:
            detalles.append("argumentos mutables riesgosos")
        if analisis["except_genericos"]:
            detalles.append("capturas de excepciones demasiado generales")
        detalle = "identifiqué " + ", ".join(detalles)
    return (
        f"Yo, LucIA, revisé {nombre} y {detalle}. "
        "El archivo es un punto válido para refactorizar si primero fijamos su comportamiento "
        "actual y verificamos cada cambio con pruebas."
    )


def consultar_ia_local(contenido, analisis):
    try:
        with open(CONFIG_IA_LOCAL, "r", encoding="utf-8") as archivo_config:
            configuracion = json.load(archivo_config)
        endpoint = str(configuracion.get("endpoint", "http://localhost:11434")).rstrip("/")
        uri_local = urllib.parse.urlparse(endpoint)
        if uri_local.scheme not in ("http", "https") or uri_local.hostname not in (
            "localhost", "127.0.0.1", "::1"
        ):
            raise ValueError("El endpoint de IA local debe estar en este equipo.")
        modelo = str(configuracion.get("default_model", "cogito:3b"))
        prompt_sistema = (
            "Eres LucIA, una programadora que diagnostica codigo. Responde en espanol, "
            "en primera persona y en maximo 90 palabras. Describe solo hallazgos verificables "
            "del archivo, sus riesgos y el primer paso seguro. No generes codigo."
        )
        contexto = (
            f"Archivo: {Path(analisis['ruta']).name}\n"
            f"Metricas: {json.dumps(analisis, ensure_ascii=False)}\n"
            f"Codigo:\n{contenido[:12000]}"
        )
        carga = {
            "model": modelo,
            "messages": [
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": contexto},
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 300,
            },
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
            raise ValueError("La respuesta de IA local no es valida.")
        mensaje = resultado.get("message")
        if not isinstance(mensaje, dict):
            raise ValueError("La respuesta de IA local no contiene un mensaje.")
        diagnostico = str(mensaje.get("content", "")).strip()
        diagnostico = " ".join(diagnostico.split()[:100])
        return diagnostico or _crear_diagnostico_respaldo(analisis)
    except (OSError, ValueError, KeyError, urllib.error.URLError):
        print("  Motor IA local no disponible; se usa el diagnostico local de LucIA.")
        return _crear_diagnostico_respaldo(analisis)


def comando_ia_local(consultor=None):
    entrada = input("Ruta del archivo a refactorizar: ").strip().strip("\"'")
    if not entrada:
        print("No se adjunto ninguna ruta.")
        return False

    ruta = Path(os.path.expandvars(os.path.expanduser(entrada))).resolve()
    if not ruta.exists():
        print(f"El archivo no existe: {ruta}")
        return False
    if not ruta.is_file():
        print(f"La ruta no corresponde a un archivo: {ruta}")
        return False

    try:
        if ruta.stat().st_size > 2_000_000:
            print("El archivo supera el limite local de 2 MB.")
            return False
        contenido = ruta.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as error:
        print(f"No se pudo leer el archivo: {error}")
        return False

    print(f"Archivo adjunto: {ruta}")
    print(f"Tamano: {ruta.stat().st_size} bytes | Lineas: {len(contenido.splitlines())}")
    confirmacion = input(
        "¿Desea que se analice el codigo y se cree un plan de refactorizacion? (s/n): "
    ).strip().lower()
    if confirmacion not in ("s", "si", "sí", "y", "yes"):
        print("Analisis cancelado.")
        return False

    try:
        analisis = analizar_archivo_local(ruta)
        generador = consultor or consultar_ia_local
        diagnostico = generador(contenido, analisis)
        plan = crear_plan_refactorizacion(analisis)
    except (OSError, ValueError, TypeError) as error:
        print(f"El analisis local fallo: {error}")
        return False

    print("\nDIAGNOSTICO DE LUCIA")
    print(diagnostico.strip())
    print("\nPLAN DE REFACTORIZACION")
    for indice, paso in enumerate(plan, start=1):
        print(f"{indice}. {paso}")
    return True


def cerrar():
    print("Cerrando el sistema...")


def mostrar_menu():
    print("=== COMANDOS ===")
    print("crear carpeta  - Crear una carpeta nueva")
    print("crear archivo  - Crear un archivo nuevo")
    print("refactorizar   - Copiar el contenido actual de PRY a RFC")
    print("actualizar     - Reemplazar PRY con la versión de RFC")
    print("IA local       - Analisar un archivo y crear un plan de refactorización")
    print("cerrar         - Cerrar el sistema")
    print("================")
    print()


def ejecutar_comandos():
    mostrar_menu()
    while True:
        entrada = input("> ").strip().lower()
        if entrada == "crear carpeta":
            comando_crear_carpeta()
        elif entrada == "crear archivo":
            comando_crear_archivo()
        elif entrada == "refactorizar":
            comando_refactorizar()
        elif entrada == "actualizar":
            actualizar()
        elif entrada == "ia local":
            comando_ia_local()
        elif entrada == "cerrar":
            cerrar()
            break
        else:
            print(f"Comando no reconocido: '{entrada}'")


if __name__ == "__main__":
    ejecutar_comandos()
