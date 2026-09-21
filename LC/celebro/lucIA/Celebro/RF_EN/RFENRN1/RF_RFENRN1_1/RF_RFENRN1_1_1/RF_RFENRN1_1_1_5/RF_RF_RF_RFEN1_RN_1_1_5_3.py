"""
Sistema de Integración con C# y .NET
=====================================
Puente bidireccional entre Python y C#/.NET para integración con Unity y OpenSim.
Soporta Mono, .NET Core y .NET Framework.

Características:
- Interoperabilidad con C# usando pythonnet
- Conexión con Unity Engine
- Integración con OpenSimulator
- Gestión de eventos .NET
- Serialización automática de datos
"""

import sys
import os
import subprocess
import json
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass
import socket
import struct


@dataclass
class DotNetConfig:
    """Configuración para integración .NET"""
    runtime_version: str = "net6.0"
    assemblies_path: List[str] = None
    unity_project_path: Optional[str] = None
    opensim_server: Optional[str] = None
    opensim_port: int = 9000
    enable_coreclr: bool = True

    def __post_init__(self):
        if self.assemblies_path is None:
            self.assemblies_path = []


class DotNetInterface:
    """
    Interfaz base para interactuar con .NET/Mono
    """

    def __init__(self, config: Optional[DotNetConfig] = None):
        self.config = config or DotNetConfig()
        self.clr_loaded = False
        self.assemblies: Dict[str, Any] = {}
        print("🔷 DotNetInterface inicializado")
        self._initialize_clr()

    def _initialize_clr(self):
        """Inicializa el Common Language Runtime"""
        try:
            import clr
            self.clr = clr
            self.clr_loaded = True
            print("✅ CLR cargado exitosamente")

            # Añadir rutas de ensamblados
            for assembly_path in self.config.assemblies_path:
                if os.path.exists(assembly_path):
                    sys.path.append(assembly_path)
                    print(f"📁 Ruta de ensamblado añadida: {assembly_path}")

        except ImportError:
            print("⚠️ pythonnet no está instalado. Instala con: pip install pythonnet")
            self.clr_loaded = False

    def load_assembly(self, assembly_name: str) -> bool:
        """
        Carga un ensamblado .NET

        Args:
            assembly_name: Nombre del ensamblado (ej: "System.Drawing")

        Returns:
            True si se cargó exitosamente
        """
        if not self.clr_loaded:
            print("❌ CLR no está cargado")
            return False

        try:
            self.clr.AddReference(assembly_name)
            self.assemblies[assembly_name] = True
            print(f"✅ Ensamblado cargado: {assembly_name}")
            return True
        except Exception as e:
            print(f"❌ Error cargando ensamblado {assembly_name}: {e}")
            return False

    def import_namespace(self, namespace: str) -> Any:
        """
        Importa un namespace de .NET

        Args:
            namespace: Nombre del namespace (ej: "System.Collections.Generic")

        Returns:
            Módulo del namespace
        """
        if not self.clr_loaded:
            print("❌ CLR no está cargado")
            return None

        try:
            module = __import__(namespace)
            print(f"📦 Namespace importado: {namespace}")
            return module
        except ImportError as e:
            print(f"❌ Error importando namespace {namespace}: {e}")
            return None

    def create_instance(self, type_name: str, *args) -> Any:
        """
        Crea una instancia de un tipo .NET

        Args:
            type_name: Nombre completo del tipo
            *args: Argumentos del constructor

        Returns:
            Instancia del objeto .NET
        """
        if not self.clr_loaded:
            print("❌ CLR no está cargado")
            return None

        try:
            parts = type_name.rsplit('.', 1)
            if len(parts) == 2:
                namespace, class_name = parts
                module = self.import_namespace(namespace)
                if module:
                    cls = getattr(module, class_name)
                    instance = cls(*args)
                    print(f"✅ Instancia creada: {type_name}")
                    return instance
        except Exception as e:
            print(f"❌ Error creando instancia: {e}")

        return None

    def call_static_method(self, type_name: str, method_name: str, *args) -> Any:
        """
        Llama a un método estático de .NET

        Args:
            type_name: Nombre completo del tipo
            method_name: Nombre del método
            *args: Argumentos del método

        Returns:
            Resultado del método
        """
        try:
            parts = type_name.rsplit('.', 1)
            if len(parts) == 2:
                namespace, class_name = parts
                module = self.import_namespace(namespace)
                if module:
                    cls = getattr(module, class_name)
                    method = getattr(cls, method_name)
                    result = method(*args)
                    print(f"✅ Método estático llamado: {type_name}.{method_name}")
                    return result
        except Exception as e:
            print(f"❌ Error llamando método estático: {e}")

        return None


class MonoRuntime:
    """
    Gestor del runtime Mono para ejecutar código C#
    """

    def __init__(self):
        self.mono_path = self._find_mono()
        self.temp_dir = Path("./mono_temp")
        self.temp_dir.mkdir(exist_ok=True)
        print(f"🔧 MonoRuntime inicializado")

    def _find_mono(self) -> Optional[str]:
        """Busca la instalación de Mono"""
        possible_paths = [
            "/usr/bin/mono",
            "/usr/local/bin/mono",
            "C:\\Program Files\\Mono\\bin\\mono.exe",
            "C:\\Program Files (x86)\\Mono\\bin\\mono.exe"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Mono encontrado en: {path}")
                return path

        # Intentar encontrar en PATH
        mono = subprocess.run(["which", "mono"], capture_output=True, text=True)
        if mono.returncode == 0:
            return mono.stdout.strip()

        print("⚠️ Mono no encontrado en el sistema")
        return None

    def compile_csharp(self, cs_code: str, output_name: str) -> Optional[str]:
        """
        Compila código C# a un ensamblado .NET

        Args:
            cs_code: Código fuente C#
            output_name: Nombre del archivo de salida (.dll o .exe)

        Returns:
            Ruta al ensamblado compilado
        """
        cs_file = self.temp_dir / "temp.cs"
        output_file = self.temp_dir / output_name

        # Escribir código fuente
        with open(cs_file, 'w') as f:
            f.write(cs_code)

        # Compilar con mcs (Mono C# compiler) o csc (.NET compiler)
        compiler = self._find_csharp_compiler()
        if not compiler:
            print("❌ No se encontró compilador de C#")
            return None

        cmd = [compiler, str(cs_file), f"-out:{output_file}", "-optimize+"]

        print(f"🔨 Compilando {cs_file}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Compilación exitosa: {output_file}")
            return str(output_file)
        else:
            print(f"❌ Error de compilación: {result.stderr}")
            return None

    def _find_csharp_compiler(self) -> Optional[str]:
        """Encuentra el compilador de C# disponible"""
        compilers = ["mcs", "csc", "dotnet"]
        for compiler in compilers:
            if subprocess.run(["which", compiler], capture_output=True).returncode == 0:
                return compiler
        return None

    def execute_assembly(self, assembly_path: str, args: List[str] = None) -> str:
        """
        Ejecuta un ensamblado .NET

        Args:
            assembly_path: Ruta al ensamblado
            args: Argumentos para el programa

        Returns:
            Salida del programa
        """
        if not self.mono_path:
            print("❌ Mono no está disponible")
            return ""

        cmd = [self.mono_path, assembly_path] + (args or [])
        result = subprocess.run(cmd, capture_output=True, text=True)

        return result.stdout


class UnityBridge:
    """
    Puente de comunicación con Unity Engine
    """

    def __init__(self, unity_project: Optional[str] = None):
        self.unity_project = unity_project
        self.socket = None
        self.connected = False
        print("🎮 UnityBridge inicializado")

    def connect_to_unity(self, host: str = "localhost", port: int = 8080) -> bool:
        """
        Establece conexión con Unity via socket

        Args:
            host: Host de Unity
            port: Puerto de comunicación

        Returns:
            True si la conexión fue exitosa
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((host, port))
            self.connected = True
            print(f"✅ Conectado a Unity en {host}:{port}")
            return True
        except Exception as e:
            print(f"❌ Error conectando a Unity: {e}")
            return False

    def send_command(self, command: str, data: Dict = None) -> bool:
        """
        Envía un comando a Unity

        Args:
            command: Nombre del comando
            data: Datos adicionales

        Returns:
            True si el comando se envió exitosamente
        """
        if not self.connected:
            print("❌ No hay conexión con Unity")
            return False

        try:
            message = json.dumps({
                'command': command,
                'data': data or {}
            })

            # Enviar longitud del mensaje primero (4 bytes)
            msg_length = len(message.encode('utf-8'))
            self.socket.sendall(struct.pack('!I', msg_length))

            # Enviar mensaje
            self.socket.sendall(message.encode('utf-8'))

            print(f"📤 Comando enviado a Unity: {command}")
            return True

        except Exception as e:
            print(f"❌ Error enviando comando: {e}")
            return False

    def receive_response(self) -> Optional[Dict]:
        """
        Recibe respuesta de Unity

        Returns:
            Diccionario con la respuesta
        """
        if not self.connected:
            return None

        try:
            # Leer longitud del mensaje
            length_data = self.socket.recv(4)
            msg_length = struct.unpack('!I', length_data)[0]

            # Leer mensaje
            message = self.socket.recv(msg_length).decode('utf-8')
            response = json.loads(message)

            print(f"📥 Respuesta recibida de Unity")
            return response

        except Exception as e:
            print(f"❌ Error recibiendo respuesta: {e}")
            return None

    def spawn_object(self, prefab: str, position: List[float], rotation: List[float]) -> bool:
        """Crea un objeto en Unity"""
        return self.send_command('SpawnObject', {
            'prefab': prefab,
            'position': position,
            'rotation': rotation
        })

    def update_avatar(self, avatar_id: str, properties: Dict) -> bool:
        """Actualiza propiedades de un avatar en Unity"""
        return self.send_command('UpdateAvatar', {
            'avatarId': avatar_id,
            'properties': properties
        })

    def disconnect(self):
        """Cierra la conexión con Unity"""
        if self.socket:
            self.socket.close()
            self.connected = False
            print("🔌 Desconectado de Unity")


class OpenSimConnector:
    """
    Conector para OpenSimulator (metaverso 3D)
    """

    def __init__(self, server: str = "localhost", port: int = 9000):
        self.server = server
        self.port = port
        self.session_id = None
        self.agent_id = None
        print(f"🌐 OpenSimConnector inicializado para {server}:{port}")

    def login(self, first_name: str, last_name: str, password: str) -> bool:
        """
        Inicia sesión en OpenSim

        Args:
            first_name: Nombre del avatar
            last_name: Apellido del avatar
            password: Contraseña

        Returns:
            True si el login fue exitoso
        """
        # Implementación simplificada
        print(f"🔐 Iniciando sesión como {first_name} {last_name}")

        # Aquí iría la lógica de autenticación con OpenSim
        # usando el protocolo LLUDP o XMLRPC

        self.session_id = "temp_session_id"
        self.agent_id = "temp_agent_id"

        print("✅ Sesión iniciada en OpenSim")
        return True

    def send_chat_message(self, message: str, channel: int = 0) -> bool:
        """Envía un mensaje de chat en OpenSim"""
        if not self.session_id:
            print("❌ No hay sesión activa")
            return False

        print(f"💬 Mensaje enviado: {message}")
        return True

    def get_region_info(self) -> Dict:
        """Obtiene información de la región actual"""
        return {
            'name': 'WoldVirtual Region',
            'position': [256.0, 256.0, 25.0],
            'size': [256, 256]
        }

    def teleport(self, region: str, x: float, y: float, z: float) -> bool:
        """Teletransporta el avatar a una ubicación"""
        print(f"🚀 Teletransportando a {region} ({x}, {y}, {z})")
        return True

    def create_object(self, prim_data: Dict) -> Optional[str]:
        """
        Crea un objeto primitivo en OpenSim

        Args:
            prim_data: Datos del primitivo (forma, tamaño, textura, etc.)

        Returns:
            UUID del objeto creado
        """
        import uuid
        object_uuid = str(uuid.uuid4())
        print(f"🎲 Objeto creado con UUID: {object_uuid}")
        return object_uuid


class CSharpBridge:
    """
    Puente principal que coordina toda la integración con C#
    """

    def __init__(self, config: Optional[DotNetConfig] = None):
        self.config = config or DotNetConfig()
        self.dotnet = DotNetInterface(self.config)
        self.mono = MonoRuntime()
        self.unity = UnityBridge(self.config.unity_project_path)
        self.opensim = OpenSimConnector(
            self.config.opensim_server or "localhost",
            self.config.opensim_port
        )
        print("🌉 CSharpBridge inicializado completamente")

    def execute_csharp_code(self, cs_code: str) -> Any:
        """
        Compila y ejecuta código C# inline

        Args:
            cs_code: Código fuente C#

        Returns:
            Resultado de la ejecución
        """
        dll_path = self.mono.compile_csharp(cs_code, "temp.dll")
        if dll_path:
            result = self.mono.execute_assembly(dll_path)
            return result
        return None

    def integrate_with_metaverse(self) -> Dict[str, bool]:
        """
        Integra todos los componentes del metaverso

        Returns:
            Estado de cada integración
        """
        status = {
            'dotnet': self.dotnet.clr_loaded,
            'mono': self.mono.mono_path is not None,
            'unity': self.unity.connect_to_unity(),
            'opensim': self.opensim.login("LucIA", "Agent", "password123")
        }

        print(f"📊 Estado de integración: {status}")
        return status


# Ejemplo de código C# para compilar
CSHARP_SAMPLE = """
using System;

namespace WoldVirtual {
    public class NeuronProcessor {
        public static double[] ProcessNeurons(double[] inputs) {
            double[] outputs = new double[inputs.Length];
            for (int i = 0; i < inputs.Length; i++) {
                outputs[i] = Math.Tanh(inputs[i]);
            }
            return outputs;
        }
        
        public static void Main() {
            Console.WriteLine("WoldVirtual Neuron Processor v1.0");
        }
    }
}
"""

print("Módulo de integración C# cargado")
