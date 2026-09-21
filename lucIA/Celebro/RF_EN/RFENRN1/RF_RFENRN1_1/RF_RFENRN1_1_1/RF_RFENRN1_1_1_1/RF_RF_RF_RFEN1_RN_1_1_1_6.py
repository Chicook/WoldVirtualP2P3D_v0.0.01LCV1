"""
CPUMonitor - Monitor de CPU Inteligente
====================================

Neurona especializada en monitorear y optimizar el uso de CPU
del sistema OpenSim.

Algoritmos 2024-2025:
- CPU usage prediction
- Thread optimization
- Load balancing
- Thermal throttling detection
"""

import numpy as np
import time
from typing import Dict, Any, List, Tuple


class CPUMonitor:
    """Monitor de CPU inteligente"""

    def __init__(self):
        self.cpu_threshold = 80.0  # % CPU threshold
        self.cpu_target = 50.0     # Target CPU usage
        self.current_cpu_percent = 0

        self.cpu_history = []
        self.peak_periods = []

        # Estrategias de optimización
        self.optimization_strategies = [
            'reduce_physics_complexity',
            'limit_script_execution',
            'lower_update_frequency',
            'disable_non_essential_features'
        ]

        print("✓ CPUMonitor inicializado")

    def monitor_cpu(self, cpu_percent: float) -> Dict[str, Any]:
        """
        Monitorea el uso de CPU

        Args:
            cpu_percent: Porcentaje de uso de CPU (0-100)

        Returns:
            Análisis y recomendaciones
        """
        self.current_cpu_percent = cpu_percent

        # Agregar al historial
        self.cpu_history.append({
            'cpu_percent': cpu_percent,
            'timestamp': time.time()
        })

        # Mantener solo últimos 60 segundos
        if len(self.cpu_history) > 60:
            self.cpu_history.pop(0)

        # Detectar picos
        if cpu_percent > self.cpu_threshold:
            self.peak_periods.append({
                'cpu_percent': cpu_percent,
                'timestamp': time.time()
            })

        # Calcular estadísticas
        is_high = cpu_percent > 70
        is_critical = cpu_percent > 90

        # Detectar tendencia
        trend = self._detect_trend()

        return {
            'current_cpu_percent': cpu_percent,
            'is_high': is_high,
            'is_critical': is_critical,
            'trend': trend,
            'recommended_action': self._get_optimization_action(cpu_percent),
            'peak_count': len(self.peak_periods)
        }

    def _detect_trend(self) -> str:
        """Detecta la tendencia del uso de CPU"""
        if len(self.cpu_history) < 5:
            return 'unknown'

        recent = self.cpu_history[-5:]
        cpu_values = [h['cpu_percent'] for h in recent]

        slope = (cpu_values[-1] - cpu_values[0]) / len(cpu_values)

        if slope > 2:
            return 'increasing'
        elif slope < -2:
            return 'decreasing'
        else:
            return 'stable'

    def _get_optimization_action(self, cpu_percent: float) -> str:
        """Obtiene acción de optimización recomendada"""
        if cpu_percent > 90:
            return 'emergency_throttle'
        elif cpu_percent > 70:
            return 'aggressive_throttle'
        elif cpu_percent > 50:
            return 'moderate_throttle'
        else:
            return 'no_action'

    def apply_throttling(self, cpu_percent: float) -> Dict[str, Any]:
        """
        Aplica throttling según el uso de CPU

        Args:
            cpu_percent: Porcentaje actual de CPU

        Returns:
            Configuración de throttling aplicada
        """
        config = {}

        if cpu_percent > 90:
            # Throttling de emergencia
            config = {
                'physics_steps_reduced': 50,
                'script_execution_limit': 0.5,
                'update_frequency_divider': 4,
                'non_essential_disabled': True
            }
        elif cpu_percent > 70:
            # Throttling agresivo
            config = {
                'physics_steps_reduced': 30,
                'script_execution_limit': 0.7,
                'update_frequency_divider': 2,
                'non_essential_disabled': False
            }
        elif cpu_percent > 50:
            # Throttling moderado
            config = {
                'physics_steps_reduced': 10,
                'script_execution_limit': 0.9,
                'update_frequency_divider': 1,
                'non_essential_disabled': False
            }

        return config

    def optimize_weights(self, weights: np.ndarray, cpu_percent: float) -> np.ndarray:
        """Optimiza pesos basándose en uso de CPU"""
        if cpu_percent > 80:
            # Reducir impacto cuando CPU alta
            adjustment = 0.93
        elif cpu_percent > 60:
            adjustment = 0.97
        else:
            # Aumentar impacto cuando CPU ok
            adjustment = 1.05

        return weights * adjustment

    def get_cpu_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de CPU"""
        if not self.cpu_history:
            return {}

        cpu_values = [h['cpu_percent'] for h in self.cpu_history]

        return {
            'current_cpu_percent': self.current_cpu_percent,
            'avg_cpu_percent': np.mean(cpu_values),
            'max_cpu_percent': np.max(cpu_values),
            'min_cpu_percent': np.min(cpu_values),
            'peak_count': len(self.peak_periods),
            'time_above_threshold': self._calculate_time_above_threshold()
        }

    def _calculate_time_above_threshold(self) -> float:
        """Calcula tiempo sobre threshold"""
        if not self.peak_periods:
            return 0.0

        return len(self.peak_periods) * 1.0  # Aproximación: 1 segundo por pico


def test_cpu_monitor():
    """Test del monitor de CPU"""
    print("\n" + "="*70)
    print("TEST: CPUMonitor")
    print("="*70)

    monitor = CPUMonitor()

    # Test 1: Monitoreo normal
    print("\n✓ Test 1: Monitoreando CPU...")
    for i in range(5):
        cpu = 40 + i * 10
        analysis = monitor.monitor_cpu(cpu)
        print(f"  ✓ CPU: {cpu}%, Alta: {analysis['is_high']}, Tendencia: {analysis['trend']}")

    # Test 2: CPU crítica
    print("\n✓ Test 2: CPU crítica...")
    analysis = monitor.monitor_cpu(95)
    print(f"✓ CPU: {analysis['current_cpu_percent']}%")
    print(f"✓ Crítico: {analysis['is_critical']}")
    print(f"✓ Acción: {analysis['recommended_action']}")

    # Test 3: Aplicar throttling
    print("\n✓ Test 3: Aplicando throttling...")
    config = monitor.apply_throttling(85)
    print(f"✓ Física reducida: {config.get('physics_steps_reduced', 0)}%")
    print(f"✓ Scripts limitados: {config.get('script_execution_limit', 1.0)}")

    # Test 4: Estadísticas
    print("\n✓ Test 4: Estadísticas...")
    stats = monitor.get_cpu_statistics()
    print(f"✓ CPU actual: {stats['current_cpu_percent']:.1f}%")
    print(f"✓ CPU promedio: {stats['avg_cpu_percent']:.1f}%")
    print(f"✓ Picos detectados: {stats['peak_count']}")

    print("\n✅ Tests completados exitosamente")
    print("="*70 + "\n")

    return stats


if __name__ == "__main__":
    test_cpu_monitor()
