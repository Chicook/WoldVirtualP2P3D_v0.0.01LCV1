"""
MetaverseMetricsCollector - Recolector de Métricas del Metaverso
============================================================

Neurona especializada en recolectar métricas clave de OpenSimulator:
- Sim FPS (Frames por segundo de simulación)
- Uso de memoria y CPU
- Recuento de Prims (objetos)
- Scripts activos

Algoritmos 2024-2025:
- Real-time metric aggregation
- Statistical analysis
- Predictive monitoring
"""

import numpy as np
import time
from typing import Dict, Any, List
from datetime import datetime


class MetaverseMetricsCollector:
    """Recolector de métricas del metaverso OpenSim"""

    def __init__(self):
        self.metrics_history = []
        self.current_metrics = {
            'sim_fps': 0,
            'memory_mb': 0,
            'cpu_percent': 0,
            'prim_count': 0,
            'active_scripts': 0,
            'timestamp': None
        }

        # Algoritmos de análisis
        self.analysis = {
            'fps_trend': 'stable',
            'memory_trend': 'stable',
            'performance_score': 100.0,
            'quality_operational': 100.0
        }

        print("✓ MetaverseMetricsCollector inicializado")

    def collect_metrics(self, sim_fps: float, memory_mb: float,
                        cpu_percent: float, prim_count: int,
                        active_scripts: int) -> Dict[str, Any]:
        """
        Recolecta métricas del metaverso

        Args:
            sim_fps: FPS de simulación
            memory_mb: Memoria usada en MB
            cpu_percent: Uso de CPU en porcentaje
            prim_count: Número de prims (objetos)
            active_scripts: Scripts activos

        Returns:
            Métricas recolectadas con análisis
        """
        timestamp = datetime.now()

        metrics = {
            'sim_fps': sim_fps,
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'prim_count': prim_count,
            'active_scripts': active_scripts,
            'timestamp': timestamp.isoformat()
        }

        # Calcular calidad operacional
        qo = self._calculate_quality_operational(metrics)
        metrics['quality_operational'] = qo

        # Agregar al historial
        self.metrics_history.append(metrics)

        # Mantener solo últimos 60 segundos (60 puntos si 1 por segundo)
        if len(self.metrics_history) > 60:
            self.metrics_history.pop(0)

        # Actualizar análisis
        self._update_analysis()

        self.current_metrics = metrics
        return metrics

    def _calculate_quality_operational(self, metrics: Dict[str, Any]) -> float:
        """
        Calcula la Calidad Operacional (QO) basada en las métricas

        Escala: 0-100 (100 = óptimo)
        """
        # Factores de calidad
        fps_score = min(metrics['sim_fps'] / 45.0, 1.0) * 100  # Target 45 FPS
        memory_score = max(0, 100 - (metrics['memory_mb'] / 10))  # Penaliza >1GB
        cpu_score = max(0, 100 - metrics['cpu_percent'] * 1.5)  # Penaliza >67% CPU

        # Factor de carga (prims y scripts)
        load_factor = min(100, 100 - (metrics['prim_count'] / 100) - (metrics['active_scripts'] / 50))

        # QO combinado (pesos: FPS 40%, Memoria 30%, CPU 20%, Carga 10%)
        qo = (fps_score * 0.4 + memory_score * 0.3 + cpu_score * 0.2 + load_factor * 0.1)

        return max(0, min(100, qo))

    def _update_analysis(self):
        """Actualiza el análisis de tendencias"""
        if len(self.metrics_history) < 2:
            return

        recent = self.metrics_history[-10:]  # Últimos 10 puntos

        # Análisis de tendencia FPS
        fps_values = [m['sim_fps'] for m in recent]
        if len(fps_values) >= 3:
            trend_slope = (fps_values[-1] - fps_values[0]) / len(fps_values)
            if trend_slope > 1:
                self.analysis['fps_trend'] = 'improving'
            elif trend_slope < -1:
                self.analysis['fps_trend'] = 'degrading'
            else:
                self.analysis['fps_trend'] = 'stable'

        # Análisis de memoria
        memory_values = [m['memory_mb'] for m in recent]
        if len(memory_values) >= 3:
            trend_slope = (memory_values[-1] - memory_values[0]) / len(memory_values)
            if trend_slope > 5:
                self.analysis['memory_trend'] = 'increasing'
            elif trend_slope < -5:
                self.analysis['memory_trend'] = 'decreasing'
            else:
                self.analysis['memory_trend'] = 'stable'

        # Score de rendimiento
        avg_qo = np.mean([m['quality_operational'] for m in recent])
        self.analysis['performance_score'] = avg_qo
        self.analysis['quality_operational'] = avg_qo

    def get_quality_operational(self) -> float:
        """Obtiene la calidad operacional actual"""
        return self.analysis['quality_operational']

    def get_performance_report(self) -> Dict[str, Any]:
        """Genera reporte de rendimiento"""
        return {
            'current_metrics': self.current_metrics,
            'analysis': self.analysis,
            'history_samples': len(self.metrics_history),
            'avg_qo': np.mean([m['quality_operational'] for m in self.metrics_history]) if self.metrics_history else 0,
            'min_qo': np.min([m['quality_operational'] for m in self.metrics_history]) if self.metrics_history else 0,
            'max_qo': np.max([m['quality_operational'] for m in self.metrics_history]) if self.metrics_history else 0
        }


def test_metaverse_metrics():
    """Test del recolector de métricas"""
    print("\n" + "="*70)
    print("TEST: MetaverseMetricsCollector")
    print("="*70)

    collector = MetaverseMetricsCollector()

    # Test 1: Recolección de métricas
    print("\n✓ Test 1: Recolectando métricas...")

    # Simular varias muestras
    for i in range(5):
        sim_fps = 45 + np.random.rand() * 5 - 2.5  # 42.5-47.5 FPS
        memory = 500 + i * 10  # Aumenta memoria
        cpu = 50 + np.random.rand() * 10  # 50-60%
        prims = 1000 + np.random.randint(-50, 50)
        scripts = 50 + np.random.randint(-10, 10)

        metrics = collector.collect_metrics(sim_fps, memory, cpu, prims, scripts)
        print(f"  ✓ Muestra {i+1}: QO={metrics['quality_operational']:.1f}, FPS={sim_fps:.1f}")

    # Test 2: Reporte de rendimiento
    print("\n✓ Test 2: Generando reporte de rendimiento...")
    report = collector.get_performance_report()
    print(f"✓ QO promedio: {report['avg_qo']:.1f}")
    print(f"✓ QO mínima: {report['min_qo']:.1f}")
    print(f"✓ QO máxima: {report['max_qo']:.1f}")
    print(f"✓ Tendencia FPS: {report['analysis']['fps_trend']}")
    print(f"✓ Score rendimiento: {report['analysis']['performance_score']:.1f}")

    # Test 3: Calidad operacional actual
    print("\n✓ Test 3: Calidad operacional actual...")
    qo = collector.get_quality_operational()
    print(f"✓ QO: {qo:.1f}/100")
    print(f"✓ Estado: {'ÓPTIMO' if qo >= 80 else 'BUENO' if qo >= 60 else 'BAJO'}")

    print("\n✅ Tests completados exitosamente")
    print("="*70 + "\n")

    return report


if __name__ == "__main__":
    test_metaverse_metrics()
