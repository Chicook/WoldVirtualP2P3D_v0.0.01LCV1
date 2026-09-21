"""
Sistema de Monitoreo, Métricas y Debugging
==========================================
Recolección avanzada de métricas, logging y debugging para sistemas de RL.
Integración con TensorBoard, Weights & Biases y herramientas de profiling.

Características:
- Recolección de métricas en tiempo real
- Monitoreo de rendimiento (CPU, GPU, Memoria)
- Tracking de recursos del sistema
- Logging estructurado
- Debugging avanzado con breakpoints condicionales
- Visualización de métricas
"""

import os
import sys
import time
import psutil
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
import json
import threading
from datetime import datetime


@dataclass
class MetricPoint:
    """Punto de datos de métrica"""
    timestamp: float
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'name': self.name,
            'value': self.value,
            'tags': self.tags
        }


@dataclass
class SystemStats:
    """Estadísticas del sistema"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    disk_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_temperature: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0


class MetricsCollector:
    """
    Recolector de métricas para entrenamiento de RL
    """

    def __init__(self, buffer_size: int = 10000):
        self.metrics: Dict[str, deque] = {}
        self.buffer_size = buffer_size
        self.start_time = time.time()

        # Contadores
        self.episode_count = 0
        self.step_count = 0

        print("📊 MetricsCollector inicializado")

    def add_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Añade una métrica

        Args:
            name: Nombre de la métrica
            value: Valor de la métrica
            tags: Tags adicionales (opcional)
        """
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.buffer_size)

        point = MetricPoint(
            timestamp=time.time() - self.start_time,
            name=name,
            value=value,
            tags=tags or {}
        )

        self.metrics[name].append(point)

    def add_scalar(self, name: str, value: float, step: int):
        """Añade una métrica escalar (compatible con TensorBoard)"""
        self.add_metric(name, value, tags={'step': str(step)})

    def add_episode_metrics(self, episode: int, reward: float, length: int,
                            success: bool = False, **extra_metrics):
        """
        Añade métricas de un episodio completo

        Args:
            episode: Número de episodio
            reward: Recompensa total del episodio
            length: Longitud del episodio (pasos)
            success: Si el episodio fue exitoso
            **extra_metrics: Métricas adicionales
        """
        self.episode_count += 1
        tags = {'episode': str(episode)}

        self.add_metric('episode/reward', reward, tags)
        self.add_metric('episode/length', length, tags)
        self.add_metric('episode/success', 1.0 if success else 0.0, tags)

        for metric_name, metric_value in extra_metrics.items():
            self.add_metric(f'episode/{metric_name}', metric_value, tags)

    def add_training_metrics(self, step: int, loss: float, **extra_metrics):
        """Añade métricas de entrenamiento"""
        self.step_count += 1
        tags = {'step': str(step)}

        self.add_metric('train/loss', loss, tags)

        for metric_name, metric_value in extra_metrics.items():
            self.add_metric(f'train/{metric_name}', metric_value, tags)

    def get_metric(self, name: str) -> List[MetricPoint]:
        """Obtiene todas las muestras de una métrica"""
        return list(self.metrics.get(name, []))

    def get_latest(self, name: str) -> Optional[float]:
        """Obtiene el último valor de una métrica"""
        if name in self.metrics and len(self.metrics[name]) > 0:
            return self.metrics[name][-1].value
        return None

    def get_statistics(self, name: str) -> Dict[str, float]:
        """Calcula estadísticas de una métrica"""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return {}

        values = [point.value for point in self.metrics[name]]

        import numpy as np

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'count': len(values)
        }

    def get_moving_average(self, name: str, window: int = 100) -> List[float]:
        """Calcula media móvil de una métrica"""
        if name not in self.metrics:
            return []

        values = [point.value for point in self.metrics[name]]

        if len(values) < window:
            return values

        import numpy as np
        moving_avg = []

        for i in range(len(values) - window + 1):
            window_values = values[i:i + window]
            moving_avg.append(np.mean(window_values))

        return moving_avg

    def export_to_json(self, filename: str):
        """Exporta todas las métricas a JSON"""
        data = {}

        for name, points in self.metrics.items():
            data[name] = [point.to_dict() for point in points]

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Métricas exportadas a {filename}")

    def export_to_csv(self, filename: str):
        """Exporta métricas a CSV"""
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'timestamp', 'value', 'tags'])

            for name, points in self.metrics.items():
                for point in points:
                    tags_str = json.dumps(point.tags)
                    writer.writerow([name, point.timestamp, point.value, tags_str])

        print(f"💾 Métricas exportadas a {filename}")

    def clear(self):
        """Limpia todas las métricas"""
        self.metrics.clear()
        self.episode_count = 0
        self.step_count = 0
        print("🗑️ Métricas limpiadas")


class PerformanceMonitor:
    """
    Monitor de rendimiento del sistema en tiempo real
    """

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stats_history: deque = deque(maxlen=1000)

        # GPU support
        self.gpu_available = False
        self._check_gpu()

        print("📈 PerformanceMonitor inicializado")

    def _check_gpu(self):
        """Verifica si hay GPU disponible"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_available = True
            self.pynvml = pynvml
            print("✅ GPU detectada")
        except:
            print("ℹ️ GPU no disponible o pynvml no instalado")

    def get_current_stats(self) -> SystemStats:
        """Obtiene estadísticas actuales del sistema"""
        stats = SystemStats()

        # CPU
        stats.cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memoria
        memory = psutil.virtual_memory()
        stats.memory_percent = memory.percent
        stats.memory_used_gb = memory.used / (1024**3)
        stats.memory_total_gb = memory.total / (1024**3)

        # Disco
        disk = psutil.disk_usage('/')
        stats.disk_percent = disk.percent

        # Red
        net_io = psutil.net_io_counters()
        stats.network_sent_mb = net_io.bytes_sent / (1024**2)
        stats.network_recv_mb = net_io.bytes_recv / (1024**2)

        # GPU
        if self.gpu_available:
            try:
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)

                # Utilización
                utilization = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats.gpu_percent = utilization.gpu

                # Memoria
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                stats.gpu_memory_used_gb = mem_info.used / (1024**3)
                stats.gpu_memory_total_gb = mem_info.total / (1024**3)

                # Temperatura
                stats.gpu_temperature = self.pynvml.nvmlDeviceGetTemperature(
                    handle, self.pynvml.NVML_TEMPERATURE_GPU
                )
            except:
                pass

        return stats

    def start_monitoring(self):
        """Inicia el monitoreo en segundo plano"""
        if self.monitoring:
            print("⚠️ Ya está monitoreando")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        print("▶️ Monitoreo iniciado")

    def stop_monitoring(self):
        """Detiene el monitoreo"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        print("⏹️ Monitoreo detenido")

    def _monitor_loop(self):
        """Loop de monitoreo en segundo plano"""
        while self.monitoring:
            stats = self.get_current_stats()
            self.stats_history.append({
                'timestamp': time.time(),
                'stats': stats
            })
            time.sleep(self.update_interval)

    def get_stats_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de las estadísticas"""
        if not self.stats_history:
            return {}

        import numpy as np

        cpu_values = [entry['stats'].cpu_percent for entry in self.stats_history]
        mem_values = [entry['stats'].memory_percent for entry in self.stats_history]

        summary = {
            'cpu': {
                'current': cpu_values[-1] if cpu_values else 0,
                'avg': np.mean(cpu_values),
                'max': np.max(cpu_values)
            },
            'memory': {
                'current': mem_values[-1] if mem_values else 0,
                'avg': np.mean(mem_values),
                'max': np.max(mem_values)
            }
        }

        if self.gpu_available:
            gpu_values = [entry['stats'].gpu_percent for entry in self.stats_history]
            summary['gpu'] = {
                'current': gpu_values[-1] if gpu_values else 0,
                'avg': np.mean(gpu_values),
                'max': np.max(gpu_values)
            }

        return summary

    def print_current_stats(self):
        """Imprime las estadísticas actuales"""
        stats = self.get_current_stats()

        print("=" * 50)
        print("📊 ESTADÍSTICAS DEL SISTEMA")
        print("=" * 50)
        print(f"CPU:     {stats.cpu_percent:.1f}%")
        print(f"Memoria: {stats.memory_percent:.1f}% ({stats.memory_used_gb:.2f}/{stats.memory_total_gb:.2f} GB)")
        print(f"Disco:   {stats.disk_percent:.1f}%")

        if self.gpu_available and stats.gpu_percent > 0:
            print(f"\nGPU:     {stats.gpu_percent:.1f}%")
            print(f"GPU Mem: {stats.gpu_memory_used_gb:.2f}/{stats.gpu_memory_total_gb:.2f} GB")
            print(f"GPU Temp: {stats.gpu_temperature:.1f}°C")

        print("=" * 50)


class ResourceTracker:
    """Rastreador de recursos para optimización"""

    def __init__(self):
        self.tracked_resources: Dict[str, Any] = {}
        print("🔍 ResourceTracker inicializado")

    def track_memory_usage(self, func: Callable) -> Callable:
        """Decorator para rastrear uso de memoria de una función"""
        def wrapper(*args, **kwargs):
            import tracemalloc

            tracemalloc.start()
            result = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"🔍 {func.__name__}: Memoria actual={current/1024**2:.2f}MB, Pico={peak/1024**2:.2f}MB")

            return result

        return wrapper

    def track_execution_time(self, func: Callable) -> Callable:
        """Decorator para rastrear tiempo de ejecución"""
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()

            elapsed = end - start
            print(f"⏱️ {func.__name__}: {elapsed*1000:.4f}ms")

            return result

        return wrapper


class Logger:
    """
    Sistema de logging estructurado
    """

    def __init__(self, name: str = "WoldVirtual", log_dir: str = "./logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configurar logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Handler para archivo
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        print(f"📝 Logger inicializado: {log_file}")

    def debug(self, message: str, **kwargs):
        """Log de debug"""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        """Log de info"""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log de warning"""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        """Log de error"""
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs):
        """Log crítico"""
        self.logger.critical(message, extra=kwargs)


class Debugger:
    """
    Sistema de debugging avanzado
    """

    def __init__(self):
        self.breakpoints: Dict[str, Callable] = {}
        self.watch_variables: Dict[str, Any] = {}
        print("🐛 Debugger inicializado")

    def set_breakpoint(self, name: str, condition: Callable[[Any], bool]):
        """
        Establece un breakpoint condicional

        Args:
            name: Nombre del breakpoint
            condition: Función que retorna True cuando se debe pausar
        """
        self.breakpoints[name] = condition
        print(f"🔴 Breakpoint establecido: {name}")

    def check_breakpoint(self, name: str, context: Any) -> bool:
        """Verifica si se debe activar un breakpoint"""
        if name in self.breakpoints:
            if self.breakpoints[name](context):
                print(f"⏸️ Breakpoint alcanzado: {name}")
                return True
        return False

    def watch(self, name: str, value: Any):
        """Observa el valor de una variable"""
        if name in self.watch_variables:
            old_value = self.watch_variables[name]
            if old_value != value:
                print(f"👁️ Variable cambiada: {name} = {old_value} → {value}")

        self.watch_variables[name] = value

    def print_stack_trace(self):
        """Imprime el stack trace actual"""
        import traceback
        traceback.print_stack()

    def inspect_object(self, obj: Any, max_depth: int = 2):
        """Inspecciona un objeto y muestra sus propiedades"""
        import inspect

        print(f"\n🔍 Inspección de objeto: {type(obj).__name__}")
        print("=" * 50)

        for name, value in inspect.getmembers(obj):
            if not name.startswith('_'):
                print(f"{name}: {value}")

        print("=" * 50)


print("✅ Módulo de monitoreo y debugging cargado")
