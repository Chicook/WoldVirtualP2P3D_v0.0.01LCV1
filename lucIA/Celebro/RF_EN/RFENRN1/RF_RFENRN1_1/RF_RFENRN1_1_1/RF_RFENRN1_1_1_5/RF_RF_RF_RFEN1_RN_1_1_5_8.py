"""
Sistema de Compilación y Optimización Cross-Platform
====================================================
Gestión avanzada de compilación, optimización JIT y build cross-platform.
Soporta C++, C#, Python y generación de binarios optimizados.

Características:
- Compilación cross-platform (Windows, Linux, macOS)
- Optimización JIT con Numba y PyPy
- Resolución automática de dependencias
- Generación de binarios standalone
- Perfilado y análisis de rendimiento
"""

import os
import sys
import platform
import subprocess
import shutil
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import json
import hashlib
import tempfile


@dataclass
class BuildConfig:
    """Configuración de compilación"""
    target_platform: str = platform.system().lower()  # windows, linux, darwin
    target_arch: str = platform.machine().lower()  # x86_64, arm64, etc.
    optimization_level: int = 2  # 0-3
    enable_lto: bool = True  # Link Time Optimization
    enable_pgo: bool = False  # Profile Guided Optimization
    enable_parallel_build: bool = True
    num_jobs: int = os.cpu_count() or 4
    output_dir: str = "./build"
    intermediate_dir: str = "./build/obj"
    debug_symbols: bool = False
    strip_binary: bool = True
    static_linking: bool = False


class DependencyResolver:
    """
    Resuelve y gestiona dependencias del proyecto
    """

    def __init__(self):
        self.dependencies: Dict[str, Dict] = {}
        self.resolved: Dict[str, str] = {}
        print("📦 DependencyResolver inicializado")

    def add_dependency(self, name: str, version: str, repo: Optional[str] = None):
        """Añade una dependencia"""
        self.dependencies[name] = {
            'version': version,
            'repo': repo,
            'installed': False
        }
        print(f"➕ Dependencia añadida: {name} v{version}")

    def resolve_dependencies(self) -> bool:
        """Resuelve todas las dependencias"""
        print("🔍 Resolviendo dependencias...")

        for name, info in self.dependencies.items():
            if self._is_dependency_installed(name, info['version']):
                self.resolved[name] = self._get_dependency_path(name)
                self.dependencies[name]['installed'] = True
                print(f"✅ {name} ya está instalado")
            else:
                print(f"📥 Instalando {name}...")
                if self._install_dependency(name, info):
                    self.resolved[name] = self._get_dependency_path(name)
                    self.dependencies[name]['installed'] = True
                else:
                    print(f"❌ Error instalando {name}")
                    return False

        print("✅ Todas las dependencias resueltas")
        return True

    def _is_dependency_installed(self, name: str, version: str) -> bool:
        """Verifica si una dependencia está instalada"""
        # Implementación simplificada
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    def _install_dependency(self, name: str, info: Dict) -> bool:
        """Instala una dependencia"""
        try:
            cmd = [sys.executable, "-m", "pip", "install", f"{name}=={info['version']}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Error: {e}")
            return False

    def _get_dependency_path(self, name: str) -> str:
        """Obtiene la ruta de instalación de una dependencia"""
        try:
            module = __import__(name)
            return os.path.dirname(module.__file__)
        except:
            return ""

    def generate_requirements_txt(self, output_file: str = "requirements.txt"):
        """Genera archivo requirements.txt"""
        with open(output_file, 'w') as f:
            for name, info in self.dependencies.items():
                f.write(f"{name}=={info['version']}\n")
        print(f"📝 Generado {output_file}")


class JITOptimizer:
    """
    Optimizador Just-In-Time usando Numba y otros
    """

    def __init__(self):
        self.numba_available = False
        self._check_numba()
        print("⚡ JITOptimizer inicializado")

    def _check_numba(self):
        """Verifica si Numba está disponible"""
        try:
            import numba
            self.numba = numba
            self.numba_available = True
            print("✅ Numba disponible")
        except ImportError:
            print("⚠️ Numba no está instalado. Instala con: pip install numba")

    def optimize_function(self, func, use_cuda: bool = False):
        """
        Optimiza una función con JIT

        Args:
            func: Función a optimizar
            use_cuda: Si True, compila para CUDA

        Returns:
            Función optimizada
        """
        if not self.numba_available:
            print("⚠️ Numba no disponible, retornando función original")
            return func

        if use_cuda:
            try:
                optimized = self.numba.cuda.jit(func)
                print(f"✅ Función optimizada para CUDA: {func.__name__}")
                return optimized
            except:
                print("⚠️ CUDA no disponible, usando CPU")

        optimized = self.numba.jit(nopython=True, cache=True)(func)
        print(f"✅ Función optimizada con JIT: {func.__name__}")
        return optimized

    def vectorize_function(self, func):
        """Vectoriza una función para operaciones paralelas"""
        if not self.numba_available:
            return func

        vectorized = self.numba.vectorize(func)
        print(f"🔢 Función vectorizada: {func.__name__}")
        return vectorized

    def benchmark_function(self, func, *args, iterations: int = 1000) -> float:
        """
        Mide el rendimiento de una función

        Returns:
            Tiempo promedio de ejecución en segundos
        """
        import time

        # Calentar la función
        for _ in range(10):
            func(*args)

        # Medir
        start = time.perf_counter()
        for _ in range(iterations):
            func(*args)
        end = time.perf_counter()

        avg_time = (end - start) / iterations
        print(f"📊 Benchmark {func.__name__}: {avg_time*1000:.4f}ms promedio")
        return avg_time


class CrossPlatformBuilder:
    """
    Constructor cross-platform para múltiples arquitecturas
    """

    def __init__(self, config: Optional[BuildConfig] = None):
        self.config = config or BuildConfig()
        self.compilers = self._detect_compilers()
        print(f"🔧 CrossPlatformBuilder inicializado para {self.config.target_platform}")

    def _detect_compilers(self) -> Dict[str, str]:
        """Detecta los compiladores disponibles"""
        compilers = {}

        # Compiladores C++
        for compiler in ['g++', 'clang++', 'cl', 'icpc']:
            if shutil.which(compiler):
                compilers['cpp'] = compiler
                break

        # Compiladores C#
        for compiler in ['mcs', 'csc', 'dotnet']:
            if shutil.which(compiler):
                compilers['csharp'] = compiler
                break

        # Python
        compilers['python'] = sys.executable

        print(f"🔍 Compiladores detectados: {list(compilers.keys())}")
        return compilers

    def build_cpp_project(self, source_files: List[str], output_name: str,
                          include_dirs: Optional[List[str]] = None,
                          library_dirs: Optional[List[str]] = None,
                          libraries: Optional[List[str]] = None) -> bool:
        """
        Compila un proyecto C++

        Args:
            source_files: Lista de archivos fuente
            output_name: Nombre del ejecutable de salida
            include_dirs: Directorios de includes
            library_dirs: Directorios de bibliotecas
            libraries: Bibliotecas a enlazar

        Returns:
            True si la compilación fue exitosa
        """
        if 'cpp' not in self.compilers:
            print("❌ No se encontró compilador C++")
            return False

        compiler = self.compilers['cpp']

        # Construir comando
        cmd = [compiler]

        # Archivos fuente
        cmd.extend(source_files)

        # Optimization flags
        opt_level = f"-O{self.config.optimization_level}"
        cmd.append(opt_level)

        # Standard
        cmd.extend(["-std=c++17", "-Wall"])

        # Include directories
        if include_dirs:
            for inc_dir in include_dirs:
                cmd.append(f"-I{inc_dir}")

        # Library directories
        if library_dirs:
            for lib_dir in library_dirs:
                cmd.append(f"-L{lib_dir}")

        # Libraries
        if libraries:
            for lib in libraries:
                cmd.append(f"-l{lib}")

        # Output
        output_path = Path(self.config.output_dir) / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["-o", str(output_path)])

        # LTO
        if self.config.enable_lto:
            cmd.append("-flto")

        # Parallel jobs
        if self.config.enable_parallel_build and 'make' in compiler.lower():
            cmd.append(f"-j{self.config.num_jobs}")

        # OpenMP
        cmd.append("-fopenmp")

        # SSE/AVX
        cmd.extend(["-msse4.2", "-mavx2"])

        print(f"🔨 Compilando proyecto C++...")
        print(f"Comando: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Compilación exitosa: {output_path}")

            # Strip binary si está habilitado
            if self.config.strip_binary and self.config.target_platform != 'windows':
                subprocess.run(['strip', str(output_path)])
                print("🗜️ Binary stripped")

            return True
        else:
            print(f"❌ Error de compilación:\n{result.stderr}")
            return False

    def build_csharp_project(self, source_files: List[str], output_name: str) -> bool:
        """Compila un proyecto C#"""
        if 'csharp' not in self.compilers:
            print("❌ No se encontró compilador C#")
            return False

        compiler = self.compilers['csharp']
        output_path = Path(self.config.output_dir) / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if compiler == 'dotnet':
            # Usar dotnet build
            cmd = ['dotnet', 'build', '-c', 'Release', '-o', str(output_path.parent)]
        else:
            # Usar mcs o csc
            cmd = [compiler] + source_files + [f'-out:{output_path}', '-optimize+']

        print(f"🔨 Compilando proyecto C#...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Compilación exitosa: {output_path}")
            return True
        else:
            print(f"❌ Error de compilación:\n{result.stderr}")
            return False

    def create_universal_binary(self, binaries: Dict[str, str], output_name: str) -> bool:
        """
        Crea un binario universal (macOS) o multi-arch

        Args:
            binaries: Diccionario de {arch: binary_path}
            output_name: Nombre del binario de salida
        """
        if self.config.target_platform != 'darwin':
            print("⚠️ Los binarios universales solo están disponibles en macOS")
            return False

        output_path = Path(self.config.output_dir) / output_name

        # Usar lipo para combinar binarios
        cmd = ['lipo', '-create'] + list(binaries.values()) + ['-output', str(output_path)]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Binario universal creado: {output_path}")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False


class BinaryGenerator:
    """
    Generador de binarios standalone
    """

    def __init__(self):
        print("📦 BinaryGenerator inicializado")

    def create_standalone_python(self, entry_point: str, output_name: str,
                                 icon: Optional[str] = None) -> bool:
        """
        Crea un ejecutable standalone de Python usando PyInstaller

        Args:
            entry_point: Script Python principal
            output_name: Nombre del ejecutable
            icon: Ruta al icono (opcional)

        Returns:
            True si la generación fue exitosa
        """
        if not shutil.which('pyinstaller'):
            print("⚠️ PyInstaller no está instalado. Instala con: pip install pyinstaller")
            return False

        cmd = [
            'pyinstaller',
            '--onefile',
            '--clean',
            f'--name={output_name}',
            '--strip'
        ]

        if icon:
            cmd.append(f'--icon={icon}')

        # Añadir opciones de optimización
        cmd.extend([
            '--optimize=2',
            '--noupx',  # No usar UPX por defecto
        ])

        cmd.append(entry_point)

        print(f"📦 Creando ejecutable standalone...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            dist_path = Path('dist') / output_name
            print(f"✅ Ejecutable creado: {dist_path}")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False

    def compress_binary(self, binary_path: str) -> bool:
        """Comprime un binario usando UPX"""
        if not shutil.which('upx'):
            print("⚠️ UPX no está instalado")
            return False

        cmd = ['upx', '--best', '--lzma', binary_path]

        print(f"🗜️ Comprimiendo binario...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Binario comprimido")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False


class CodeCompiler:
    """
    Compilador principal que coordina todos los componentes
    """

    def __init__(self, config: Optional[BuildConfig] = None):
        self.config = config or BuildConfig()
        self.dependency_resolver = DependencyResolver()
        self.jit_optimizer = JITOptimizer()
        self.builder = CrossPlatformBuilder(self.config)
        self.binary_generator = BinaryGenerator()
        print("🔧 CodeCompiler inicializado")

    def compile_project(self, project_config: Dict) -> bool:
        """
        Compila un proyecto completo

        Args:
            project_config: Configuración del proyecto

        Returns:
            True si la compilación fue exitosa
        """
        print(f"🚀 Compilando proyecto: {project_config.get('name', 'Unknown')}")

        # Resolver dependencias
        if not self.dependency_resolver.resolve_dependencies():
            print("❌ Error resolviendo dependencias")
            return False

        # Compilar según el tipo de proyecto
        project_type = project_config.get('type', 'python')

        if project_type == 'cpp':
            return self.builder.build_cpp_project(
                source_files=project_config.get('sources', []),
                output_name=project_config.get('output', 'app'),
                include_dirs=project_config.get('include_dirs'),
                library_dirs=project_config.get('library_dirs'),
                libraries=project_config.get('libraries')
            )

        elif project_type == 'csharp':
            return self.builder.build_csharp_project(
                source_files=project_config.get('sources', []),
                output_name=project_config.get('output', 'app.exe')
            )

        elif project_type == 'python':
            return self.binary_generator.create_standalone_python(
                entry_point=project_config.get('entry_point', 'main.py'),
                output_name=project_config.get('output', 'app'),
                icon=project_config.get('icon')
            )

        else:
            print(f"❌ Tipo de proyecto no soportado: {project_type}")
            return False

    def profile_code(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Perfila una función y retorna estadísticas"""
        import cProfile
        import pstats
        import io

        profiler = cProfile.Profile()
        profiler.enable()

        result = func(*args, **kwargs)

        profiler.disable()

        # Capturar estadísticas
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)

        profile_data = {
            'result': result,
            'stats': stream.getvalue()
        }

        print("📊 Perfilado completado")
        return profile_data


# Ejemplo de uso
EXAMPLE_PROJECT_CONFIG = {
    'name': 'WoldVirtual3D',
    'type': 'cpp',
    'sources': ['main.cpp', 'engine.cpp'],
    'output': 'woldvirtual',
    'include_dirs': ['./include'],
    'library_dirs': ['./lib'],
    'libraries': ['pthread', 'dl']
}

print("✅ Módulo de compilación cargado")
