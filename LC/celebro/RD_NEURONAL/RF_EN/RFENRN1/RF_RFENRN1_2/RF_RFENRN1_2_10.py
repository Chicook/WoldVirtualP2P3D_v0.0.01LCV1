"""
RF_RFENRN1_2_10.py - Gestor de Monitorización, Profiling y Automatización
==========================================================================

Implementa un sistema completo de monitorización, profiling y automatización
para redes neuronales de aprendizaje por refuerzo. Incluye PyTorch Profiler,
métricas en tiempo real, logging avanzado, alertas automáticas y dashboards
para optimizar el rendimiento y detectar problemas.

Características:
- PyTorch Profiler para análisis detallado de rendimiento
- Sistema de métricas en tiempo real con Weights & Biases
- Logging estructurado y alertas automáticas
- Dashboards interactivos para monitoreo
- Automatización de checkpoints y validación
- Detección automática de anomalías y problemas
- Sistema de reportes automáticos

Autor: LucIA Development Team
Versión: 2.0.0
"""

try:
    import torch
    import torch.nn as nn
    import torch.profiler as profiler
except ImportError:
    pass  # dependencia pesada opcional
try:
    torch
except NameError:
    import types as _t
    torch = _t.SimpleNamespace(
        no_grad=lambda *a, **k: (lambda f: f) if a and callable(a[0]) else (lambda f: f),
        optim=_t.SimpleNamespace(Optimizer=object),
        Tensor=object,
    )
try:
    nn
except NameError:
    import types as _t2
    nn = _t2.SimpleNamespace(Module=object)
import numpy as np
import time
import json
import logging
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psutil
import os
import warnings
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger('RFENRN1.RF_RFENRN1_2_10')


@dataclass
class MonitoringConfig:
    """Configuración para monitorización y profiling"""
    use_profiler: bool = True
    profiler_schedule: str = 'wait_arm_warmup'
    profiler_activities: List[str] = field(default_factory=lambda: ['cpu', 'cuda'])
    use_wandb: bool = True
    wandb_project: str = 'rl_optimization'
    wandb_entity: str = None
    use_tensorboard: bool = True
    log_dir: str = './logs'
    metrics_update_frequency: int = 10  # segundos
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'memory_usage': 0.9,
        'gpu_utilization': 0.95,
        'loss_spike': 2.0,
        'gradient_norm': 10.0
    })
    checkpoint_frequency: int = 100  # pasos
    validation_frequency: int = 50  # pasos
    use_anomaly_detection: bool = True
    anomaly_window: int = 100
    dashboard_port: int = 8080


class MonitoringManager:
    """
    Gestor de monitorización, profiling y automatización.

    Implementa un sistema completo para monitorear, analizar y optimizar
    el rendimiento de redes de aprendizaje por refuerzo.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        """
        Inicializa el gestor de monitorización.

        Args:
            config: Configuración de monitorización (opcional)
        """
        self.config = config or MonitoringConfig()
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = []
        self.checkpoints = []
        self.profiler_results = {}
        self.monitoring_stats = {
            'total_training_time': 0.0,
            'peak_memory_usage': 0.0,
            'average_gpu_utilization': 0.0,
            'total_loss': 0.0,
            'anomalies_detected': 0,
            'checkpoints_saved': 0
        }
        self.monitoring_active = False
        self.monitoring_thread = None
        self.anomaly_detector = AnomalyDetector(self.config.anomaly_window)

        # Configurar logging
        self._setup_logging()

        # Inicializar herramientas de monitoreo
        if self.config.use_wandb:
            self._init_wandb()

        if self.config.use_tensorboard:
            self._init_tensorboard()

        logger.info("MonitoringManager inicializado")

    def _setup_logging(self) -> None:
        """Configura el sistema de logging."""
        # Crear directorio de logs si no existe
        os.makedirs(self.config.log_dir, exist_ok=True)

        # Configurar logger principal
        log_file = os.path.join(self.config.log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    def _init_wandb(self) -> None:
        """Inicializa Weights & Biases."""
        try:
            import wandb

            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.__dict__
            )

            logger.info("Weights & Biases inicializado")

        except ImportError:
            logger.warning("Weights & Biases no disponible")
        except Exception as e:
            logger.error(f"Error inicializando W&B: {e}")

    def _init_tensorboard(self) -> None:
        """Inicializa TensorBoard."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter(
                log_dir=os.path.join(self.config.log_dir, 'tensorboard')
            )

            logger.info("TensorBoard inicializado")

        except ImportError:
            logger.warning("TensorBoard no disponible")
        except Exception as e:
            logger.error(f"Error inicializando TensorBoard: {e}")

    def start_profiling(self, model: nn.Module, optimizer, train_loader) -> profiler.profile:
        """
        Inicia el profiling del modelo.

        Args:
            model: Modelo PyTorch
            optimizer: Optimizador
            train_loader: DataLoader de entrenamiento

        Returns:
            Profiler configurado
        """
        if not self.config.use_profiler:
            return None

        try:
            profiler_schedule = profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=2
            )

            profiler_instance = profiler.profile(
                activities=self.config.profiler_activities,
                schedule=profiler_schedule,
                on_trace_ready=self._on_trace_ready,
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )

            logger.info("Profiling iniciado")
            return profiler_instance

        except Exception as e:
            logger.error(f"Error iniciando profiling: {e}")
            return None

    def _on_trace_ready(self, prof) -> None:
        """
        Callback cuando el profiling está listo.

        Args:
            prof: Instancia del profiler
        """
        try:
            # Guardar resultados del profiling
            trace_file = os.path.join(
                self.config.log_dir,
                f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            prof.export_chrome_trace(trace_file)

            # Analizar resultados
            self._analyze_profiler_results(prof)

            logger.info(f"Trace del profiling guardado en {trace_file}")

        except Exception as e:
            logger.error(f"Error procesando trace del profiling: {e}")

    def _analyze_profiler_results(self, prof) -> None:
        """
        Analiza los resultados del profiling.

        Args:
            prof: Instancia del profiler
        """
        try:
            # Obtener estadísticas de tiempo
            total_time = sum(event.cuda_time_total for event in prof.events())
            gpu_time = sum(event.cuda_time for event in prof.events())

            self.profiler_results = {
                'total_time': total_time,
                'gpu_time': gpu_time,
                'gpu_utilization': gpu_time / total_time if total_time > 0 else 0,
                'memory_peak': max(event.cuda_memory_usage for event in prof.events()),
                'events_count': len(prof.events())
            }

            logger.info(f"Análisis del profiling completado: {self.profiler_results}")

        except Exception as e:
            logger.error(f"Error analizando resultados del profiling: {e}")

    def start_monitoring(self) -> None:
        """Inicia el monitoreo en tiempo real."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.start()

        logger.info("Monitoreo en tiempo real iniciado")

    def stop_monitoring(self) -> None:
        """Detiene el monitoreo."""
        self.monitoring_active = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join()

        logger.info("Monitoreo detenido")

    def _monitoring_loop(self) -> None:
        """Loop principal de monitoreo."""
        while self.monitoring_active:
            try:
                # Recopilar métricas del sistema
                system_metrics = self._collect_system_metrics()

                # Recopilar métricas de GPU
                gpu_metrics = self._collect_gpu_metrics()

                # Combinar métricas
                all_metrics = {**system_metrics, **gpu_metrics}

                # Actualizar historial
                for key, value in all_metrics.items():
                    self.metrics_history[key].append({
                        'timestamp': time.time(),
                        'value': value
                    })

                # Detectar anomalías
                if self.config.use_anomaly_detection:
                    self._detect_anomalies(all_metrics)

                # Verificar alertas
                self._check_alerts(all_metrics)

                # Logging de métricas
                self._log_metrics(all_metrics)

                # Enviar a W&B
                if self.config.use_wandb:
                    self._log_to_wandb(all_metrics)

                # Enviar a TensorBoard
                if self.config.use_tensorboard:
                    self._log_to_tensorboard(all_metrics)

                time.sleep(self.config.metrics_update_frequency)

            except Exception as e:
                logger.error(f"Error en loop de monitoreo: {e}")
                time.sleep(5)  # Esperar antes de reintentar

    def _collect_system_metrics(self) -> Dict[str, float]:
        """
        Recopila métricas del sistema.

        Returns:
            Diccionario con métricas del sistema
        """
        try:
            # Métricas de CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Métricas de memoria
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0

            # Métricas de disco
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent / 100.0

            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'disk_usage': disk_percent,
                'cpu_count': cpu_count
            }

        except Exception as e:
            logger.error(f"Error recopilando métricas del sistema: {e}")
            return {}

    def _collect_gpu_metrics(self) -> Dict[str, float]:
        """
        Recopila métricas de GPU.

        Returns:
            Diccionario con métricas de GPU
        """
        if not torch.cuda.is_available():
            return {}

        try:
            # Métricas de memoria GPU
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            memory_total = torch.cuda.get_device_properties(0).total_memory

            memory_usage = memory_allocated / memory_total

            return {
                'gpu_memory_usage': memory_usage,
                'gpu_memory_allocated': memory_allocated,
                'gpu_memory_reserved': memory_reserved,
                'gpu_memory_total': memory_total
            }

        except Exception as e:
            logger.error(f"Error recopilando métricas de GPU: {e}")
            return {}

    def _detect_anomalies(self, metrics: Dict[str, float]) -> None:
        """
        Detecta anomalías en las métricas.

        Args:
            metrics: Métricas actuales
        """
        anomalies = self.anomaly_detector.detect(metrics)

        if anomalies:
            self.monitoring_stats['anomalies_detected'] += len(anomalies)

            for anomaly in anomalies:
                alert = {
                    'type': 'anomaly',
                    'metric': anomaly['metric'],
                    'value': anomaly['value'],
                    'severity': anomaly['severity'],
                    'timestamp': datetime.now(),
                    'description': f"Anomalía detectada en {anomaly['metric']}: {anomaly['value']}"
                }

                self.alerts.append(alert)
                logger.warning(f"Anomalía detectada: {alert['description']}")

    def _check_alerts(self, metrics: Dict[str, float]) -> None:
        """
        Verifica condiciones de alerta.

        Args:
            metrics: Métricas actuales
        """
        for metric_name, threshold in self.config.alert_thresholds.items():
            if metric_name in metrics and metrics[metric_name] > threshold:
                alert = {
                    'type': 'threshold',
                    'metric': metric_name,
                    'value': metrics[metric_name],
                    'threshold': threshold,
                    'severity': 'high',
                    'timestamp': datetime.now(),
                    'description': f"{metric_name} excede umbral: {metrics[metric_name]:.3f} > {threshold:.3f}"
                }

                self.alerts.append(alert)
                logger.warning(f"Alerta: {alert['description']}")

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Registra métricas en el log.

        Args:
            metrics: Métricas a registrar
        """
        metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        logger.info(f"Métricas: {metrics_str}")

    def _log_to_wandb(self, metrics: Dict[str, float]) -> None:
        """
        Envía métricas a Weights & Biases.

        Args:
            metrics: Métricas a enviar
        """
        try:
            import wandb
            wandb.log(metrics)
        except Exception as e:
            logger.error(f"Error enviando métricas a W&B: {e}")

    def _log_to_tensorboard(self, metrics: Dict[str, float]) -> None:
        """
        Envía métricas a TensorBoard.

        Args:
            metrics: Métricas a enviar
        """
        try:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"System/{key}", value, time.time())
        except Exception as e:
            logger.error(f"Error enviando métricas a TensorBoard: {e}")

    def log_training_metrics(self, epoch: int, loss: float, accuracy: float = None,
                             lr: float = None, **kwargs) -> None:
        """
        Registra métricas de entrenamiento.

        Args:
            epoch: Época actual
            loss: Pérdida actual
            accuracy: Precisión (opcional)
            lr: Learning rate (opcional)
            **kwargs: Métricas adicionales
        """
        metrics = {
            'epoch': epoch,
            'loss': loss,
            'learning_rate': lr,
            'accuracy': accuracy,
            **kwargs
        }

        # Actualizar estadísticas
        self.monitoring_stats['total_loss'] += loss

        # Enviar a sistemas de logging
        if self.config.use_wandb:
            self._log_to_wandb(metrics)

        if self.config.use_tensorboard:
            for key, value in metrics.items():
                if value is not None:
                    self.tb_writer.add_scalar(f"Training/{key}", value, epoch)

        logger.info(f"Época {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.4f if accuracy else 'N/A'}")

    def save_checkpoint(self, model: nn.Module, optimizer, epoch: int,
                        loss: float, path: str = None) -> str:
        """
        Guarda un checkpoint del modelo.

        Args:
            model: Modelo PyTorch
            optimizer: Optimizador
            epoch: Época actual
            loss: Pérdida actual
            path: Ruta donde guardar (opcional)

        Returns:
            Ruta del checkpoint guardado
        """
        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(self.config.log_dir, f"checkpoint_epoch_{epoch}_{timestamp}.pth")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'monitoring_stats': self.monitoring_stats
        }

        torch.save(checkpoint, path)

        self.checkpoints.append({
            'path': path,
            'epoch': epoch,
            'loss': loss,
            'timestamp': datetime.now()
        })

        self.monitoring_stats['checkpoints_saved'] += 1

        logger.info(f"Checkpoint guardado en {path}")

        return path

    def generate_report(self) -> Dict[str, Any]:
        """
        Genera un reporte completo del entrenamiento.

        Returns:
            Diccionario con el reporte
        """
        report = {
            'summary': {
                'total_training_time': self.monitoring_stats['total_training_time'],
                'peak_memory_usage': self.monitoring_stats['peak_memory_usage'],
                'average_gpu_utilization': self.monitoring_stats['average_gpu_utilization'],
                'total_loss': self.monitoring_stats['total_loss'],
                'anomalies_detected': self.monitoring_stats['anomalies_detected'],
                'checkpoints_saved': self.monitoring_stats['checkpoints_saved']
            },
            'alerts': self.alerts[-10:],  # Últimas 10 alertas
            'checkpoints': self.checkpoints,
            'profiler_results': self.profiler_results,
            'config': self.config.__dict__
        }

        return report

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """
        Obtiene las estadísticas de monitoreo.

        Returns:
            Diccionario con estadísticas
        """
        return self.monitoring_stats.copy()

    def cleanup(self) -> None:
        """Limpia recursos de monitoreo."""
        self.stop_monitoring()

        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()

        try:
            import wandb
            wandb.finish()
        except:
            pass

        logger.info("Recursos de monitoreo limpiados")


class AnomalyDetector:
    """
    Detector de anomalías en métricas.

    Implementa detección automática de anomalías usando técnicas
    estadísticas y de machine learning.
    """

    def __init__(self, window_size: int = 100):
        """
        Inicializa el detector de anomalías.

        Args:
            window_size: Tamaño de ventana para detección
        """
        self.window_size = window_size
        self.metric_history = defaultdict(lambda: deque(maxlen=window_size))
        self.thresholds = {}

    def detect(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Detecta anomalías en las métricas.

        Args:
            metrics: Métricas actuales

        Returns:
            Lista de anomalías detectadas
        """
        anomalies = []

        for metric_name, value in metrics.items():
            # Actualizar historial
            self.metric_history[metric_name].append(value)

            # Detectar anomalías si hay suficiente historial
            if len(self.metric_history[metric_name]) >= self.window_size:
                anomaly = self._detect_metric_anomaly(metric_name, value)
                if anomaly:
                    anomalies.append(anomaly)

        return anomalies

    def _detect_metric_anomaly(self, metric_name: str, value: float) -> Optional[Dict[str, Any]]:
        """
        Detecta anomalía en una métrica específica.

        Args:
            metric_name: Nombre de la métrica
            value: Valor actual

        Returns:
            Anomalía detectada o None
        """
        history = list(self.metric_history[metric_name])

        if len(history) < 10:  # Necesitamos al menos 10 puntos
            return None

        # Calcular estadísticas
        mean = np.mean(history[:-1])  # Excluir el valor actual
        std = np.std(history[:-1])

        if std == 0:
            return None

        # Detectar outliers usando Z-score
        z_score = abs(value - mean) / std

        if z_score > 3.0:  # Umbral de 3 desviaciones estándar
            severity = 'high' if z_score > 5.0 else 'medium'

            return {
                'metric': metric_name,
                'value': value,
                'mean': mean,
                'std': std,
                'z_score': z_score,
                'severity': severity
            }

        return None
