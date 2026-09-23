import os
import shutil
import stat
import sys
import uuid


def encontrar_raiz(ruta):
    actual = os.path.abspath(ruta)
    while True:
        if os.path.isdir(os.path.join(actual, "Construccion", "CPRZ", "PRY")):
            return actual
        padre = os.path.dirname(actual)
        if padre == actual:
            raise RuntimeError("No se encontro la raiz del proyecto.")
        actual = padre


ROOT = encontrar_raiz(__file__)
ORIGEN = os.path.join(ROOT, "Construccion", "CPRZ", "PRY")
DESTINO = os.path.join(ROOT, "Construccion", "CONTRRF", "RFC")
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

    confirmacion = input("¿Seguro que deseas continuar? (s/n): ").strip().lower()
    if confirmacion != "s":
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


def cerrar():
    print("Cerrando el sistema...")
    sys.exit()


def mostrar_menu():
    print("=== COMANDOS ===")
    print("crear carpeta  - Crear una carpeta nueva")
    print("crear archivo  - Crear un archivo nuevo")
    print("refactorizar   - Copiar el contenido actual de PRY a RFC")
    print("actualizar     - Reemplazar PRY con la versión de RFC")
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
        elif entrada == "cerrar":
            cerrar()
        else:
            print(f"Comando no reconocido: '{entrada}'")


if __name__ == "__main__":
    ejecutar_comandos()
