"""
Sistema de Integración con C++
===============================
Puente bidireccional entre Python y C++ usando Pybind11, Nuitka y Shed Skin.
Permite compilar código Python a C++ y ejecutar bibliotecas C++ desde Python.

Características:
- Binding automático con Pybind11
- Compilación optimizada con Nuitka
- Traducción Python → C++ con Shed Skin
- Gestión de memoria compartida
- Interfaz de alto rendimiento
"""

import os
import sys
import subprocess
import ctypes
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass
import json
import tempfile
import shutil


@dataclass
class CompilationConfig:
    """Configuración para compilación de C++"""
    optimization_level: int = 3  # -O3
    std_version: str = "c++17"
    include_dirs: List[str] = None
    library_dirs: List[str] = None
    libraries: List[str] = None
    extra_compile_args: List[str] = None
    extra_link_args: List[str] = None
    parallel_jobs: int = 4
    enable_openmp: bool = True
    enable_sse: bool = True
    enable_avx: bool = True

    def __post_init__(self):
        if self.include_dirs is None:
            self.include_dirs = []
        if self.library_dirs is None:
            self.library_dirs = []
        if self.libraries is None:
            self.libraries = []
        if self.extra_compile_args is None:
            self.extra_compile_args = []
        if self.extra_link_args is None:
            self.extra_link_args = []


class Pybind11Interface:
    """
    Interfaz para crear bindings de C++ usando Pybind11
    """

    def __init__(self, config: Optional[CompilationConfig] = None):
        self.config = config or CompilationConfig()
        self.bindings: Dict[str, Any] = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix="pybind11_"))
        print(f"🔗 Pybind11Interface inicializado en {self.temp_dir}")

    def create_binding(self, cpp_code: str, module_name: str) -> bool:
        """
        Crea un binding de Pybind11 desde código C++

        Args:
            cpp_code: Código fuente C++
            module_name: Nombre del módulo Python resultante

        Returns:
            True si la compilación fue exitosa
        """
        try:
            # Crear archivo temporal con el código C++
            cpp_file = self.temp_dir / f"{module_name}.cpp"
            with open(cpp_file, 'w') as f:
                f.write(self._wrap_with_pybind11(cpp_code, module_name))

            # Compilar con g++ o clang++
            compiler = self._detect_compiler()
            compile_cmd = self._build_compile_command(cpp_file, module_name, compiler)

            print(f"🔨 Compilando {module_name}...")
            result = subprocess.run(compile_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Binding {module_name} compilado exitosamente")
                return True
            else:
                print(f"❌ Error de compilación: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error creando binding: {e}")
            return False

    def _wrap_with_pybind11(self, cpp_code: str, module_name: str) -> str:
        """Envuelve el código C++ con la interfaz de Pybind11"""
        wrapper = f"""
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

{cpp_code}

PYBIND11_MODULE({module_name}, m) {{
    m.doc() = "Módulo C++ compilado con Pybind11 para WoldVirtual3D";
    
    // Las funciones y clases se exportarán aquí
    // Esto se generará automáticamente basado en el código
}}
"""
        return wrapper

    def _detect_compiler(self) -> str:
        """Detecta el compilador C++ disponible"""
        compilers = ['g++', 'clang++', 'cl']
        for compiler in compilers:
            if shutil.which(compiler):
                print(f"🔍 Compilador detectado: {compiler}")
                return compiler
        raise RuntimeError("No se encontró ningún compilador C++")

    def _build_compile_command(self, source_file: Path, module_name: str, compiler: str) -> List[str]:
        """Construye el comando de compilación"""
        python_include = subprocess.check_output(
            [sys.executable, '-c', 'import sysconfig; print(sysconfig.get_path("include"))'],
            text=True
        ).strip()

        output_file = self.temp_dir / f"{module_name}.so"

        cmd = [
            compiler,
            str(source_file),
            f"-O{self.config.optimization_level}",
            f"-std={self.config.std_version}",
            "-shared",
            "-fPIC",
            f"-I{python_include}",
            f"-o{output_file}"
        ]

        # Añadir flags de optimización
        if self.config.enable_openmp:
            cmd.append("-fopenmp")
        if self.config.enable_sse:
            cmd.append("-msse4.2")
        if self.config.enable_avx:
            cmd.append("-mavx2")

        # Añadir directorios de include
        for inc_dir in self.config.include_dirs:
            cmd.append(f"-I{inc_dir}")

        # Añadir directorios de bibliotecas
        for lib_dir in self.config.library_dirs:
            cmd.append(f"-L{lib_dir}")

        # Añadir bibliotecas
        for lib in self.config.libraries:
            cmd.append(f"-l{lib}")

        cmd.extend(self.config.extra_compile_args)

        return cmd

    def load_module(self, module_name: str) -> Any:
        """Carga el módulo compilado"""
        sys.path.insert(0, str(self.temp_dir))
        try:
            module = __import__(module_name)
            self.bindings[module_name] = module
            print(f"📦 Módulo {module_name} cargado")
            return module
        except ImportError as e:
            print(f"❌ Error cargando módulo: {e}")
            return None

    def cleanup(self):
        """Limpia archivos temporales"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        print("🧹 Archivos temporales eliminados")


class NuitkaOptimizer:
    """
    Optimizador que usa Nuitka para compilar Python a C
    """

    def __init__(self):
        self.output_dir = Path("./nuitka_build")
        self.output_dir.mkdir(exist_ok=True)
        print("⚡ NuitkaOptimizer inicializado")

    def compile_python_to_c(self, python_file: str, output_name: str = None) -> bool:
        """
        Compila un archivo Python a código C usando Nuitka

        Args:
            python_file: Ruta al archivo Python
            output_name: Nombre del ejecutable resultante

        Returns:
            True si la compilación fue exitosa
        """
        if not shutil.which("nuitka"):
            print("❌ Nuitka no está instalado. Instala con: pip install nuitka")
            return False

        output_name = output_name or Path(python_file).stem

        cmd = [
            "nuitka",
            "--standalone",
            "--onefile",
            f"--output-dir={self.output_dir}",
            f"--output-filename={output_name}",
            "--enable-plugin=numpy",
            "--enable-plugin=torch",
            "--follow-imports",
            "--remove-output",
            python_file
        ]

        print(f"🔄 Compilando {python_file} con Nuitka...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Compilación exitosa: {self.output_dir / output_name}")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False

    def optimize_module(self, module_path: str) -> bool:
        """Optimiza un módulo Python completo"""
        cmd = [
            "nuitka",
            "--module",
            f"--output-dir={self.output_dir}",
            "--enable-plugin=numpy",
            "--remove-output",
            module_path
        ]

        print(f"⚙️ Optimizando módulo {module_path}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        return result.returncode == 0


class ShedSkinTranslator:
    """
    Traductor de Python a C++ usando Shed Skin
    """

    def __init__(self):
        self.output_dir = Path("./shedskin_build")
        self.output_dir.mkdir(exist_ok=True)
        print("🔄 ShedSkinTranslator inicializado")

    def translate_to_cpp(self, python_file: str) -> Optional[str]:
        """
        Traduce código Python a C++

        Args:
            python_file: Archivo Python a traducir

        Returns:
            Ruta al archivo C++ generado o None si falla
        """
        if not shutil.which("shedskin"):
            print("❌ Shed Skin no está instalado")
            return None

        cmd = [
            "shedskin",
            "-e",  # Generate extension module
            f"-d{self.output_dir}",
            python_file
        ]

        print(f"🔄 Traduciendo {python_file} a C++...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            cpp_file = self.output_dir / f"{Path(python_file).stem}.cpp"
            print(f"✅ Traducción exitosa: {cpp_file}")
            return str(cpp_file)
        else:
            print(f"❌ Error: {result.stderr}")
            return None


class CppBridge:
    """
    Puente principal que coordina todas las herramientas de integración C++
    """

    def __init__(self, config: Optional[CompilationConfig] = None):
        self.config = config or CompilationConfig()
        self.pybind11 = Pybind11Interface(self.config)
        self.nuitka = NuitkaOptimizer()
        self.shedskin = ShedSkinTranslator()
        self.loaded_modules: Dict[str, Any] = {}
        print("🌉 CppBridge inicializado")

    def compile_cpp_function(self, cpp_code: str, function_name: str) -> Callable:
        """
        Compila una función C++ y la hace disponible en Python

        Args:
            cpp_code: Código fuente C++
            function_name: Nombre de la función

        Returns:
            Función callable en Python
        """
        module_name = f"cpp_func_{function_name}"

        if self.pybind11.create_binding(cpp_code, module_name):
            module = self.pybind11.load_module(module_name)
            if module and hasattr(module, function_name):
                self.loaded_modules[function_name] = module
                return getattr(module, function_name)

        return None

    def optimize_python_code(self, python_file: str, method: str = "nuitka") -> bool:
        """
        Optimiza código Python usando el método especificado

        Args:
            python_file: Archivo Python a optimizar
            method: 'nuitka' o 'shedskin'

        Returns:
            True si la optimización fue exitosa
        """
        if method == "nuitka":
            return self.nuitka.compile_python_to_c(python_file)
        elif method == "shedskin":
            cpp_file = self.shedskin.translate_to_cpp(python_file)
            return cpp_file is not None
        else:
            print(f"❌ Método desconocido: {method}")
            return False

    def benchmark_performance(self, python_func: Callable, cpp_func: Callable,
                              iterations: int = 10000) -> Dict[str, float]:
        """
        Compara el rendimiento entre Python y C++

        Returns:
            Diccionario con tiempos de ejecución
        """
        import time

        # Benchmark Python
        start = time.perf_counter()
        for _ in range(iterations):
            python_func()
        python_time = time.perf_counter() - start

        # Benchmark C++
        start = time.perf_counter()
        for _ in range(iterations):
            cpp_func()
        cpp_time = time.perf_counter() - start

        speedup = python_time / cpp_time if cpp_time > 0 else 0

        results = {
            'python_time': python_time,
            'cpp_time': cpp_time,
            'speedup': speedup,
            'iterations': iterations
        }

        print(f"📊 Benchmark: Python={python_time:.4f}s, C++={cpp_time:.4f}s, Speedup={speedup:.2f}x")
        return results

    def cleanup(self):
        """Limpia todos los recursos temporales"""
        self.pybind11.cleanup()
        print("✅ CppBridge recursos liberados")


class CppCompiler:
    """Compilador directo de código C++"""

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="cpp_compiler_"))
        print(f"🔧 CppCompiler inicializado en {self.temp_dir}")

    def compile_and_run(self, cpp_code: str, args: List[str] = None) -> str:
        """Compila y ejecuta código C++"""
        source_file = self.temp_dir / "temp.cpp"
        exe_file = self.temp_dir / "temp.exe"

        with open(source_file, 'w') as f:
            f.write(cpp_code)

        # Compilar
        compile_cmd = ["g++", str(source_file), "-o", str(exe_file), "-O3", "-std=c++17"]
        subprocess.run(compile_cmd, check=True)

        # Ejecutar
        run_cmd = [str(exe_file)] + (args or [])
        result = subprocess.run(run_cmd, capture_output=True, text=True)

        return result.stdout


# Funciones de utilidad para código C++ optimizado
CPP_MATRIX_MULTIPLY = """
#include <vector>
#include <iostream>

std::vector<std::vector<double>> matrix_multiply(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B
) {
    size_t m = A.size();
    size_t n = B[0].size();
    size_t p = B.size();
    
    std::vector<std::vector<double>> C(m, std::vector<double>(n, 0.0));
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < p; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return C;
}
"""

print("Módulo de integración C++ cargado")
