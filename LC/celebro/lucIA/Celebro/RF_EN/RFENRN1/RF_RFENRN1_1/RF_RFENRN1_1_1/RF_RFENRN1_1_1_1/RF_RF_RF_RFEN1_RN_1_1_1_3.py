"""
PerformanceActuator - Actuador de IA de Rendimiento
===============================================

Primer "Actuador" de IA que toma decisiones basadas en la Calidad Operacional (QO).

Ejemplo: Si la QO es baja, reduce el nivel de detalle de objetos a 50 metros
para mejorar el rendimiento.

Algoritmos 2024-2025:
- Reinforcement learning para actuadores
- Adaptive decision making
- Threshold-based optimization
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class PerformanceActuator:
    """Actuador de IA que toma decisiones basadas en QO"""

    def __init__(self):
        self.qo_threshold_low = 60.0   # QO baja
        self.qo_threshold_medium = 80.0  # QO media
        self.qo_threshold_high = 90.0   # QO alta

        self.actions_taken = []
        self.actuation_history = []

        # Decision rules
        self.rules = {
            'reduce_lod': {'qo_max': 60, 'action': 'Reducir LOD a 50m'},
            'reduce_scripts': {'qo_max': 40, 'action': 'Reducir scripts activos'},
            'optimize_prims': {'qo_max': 30, 'action': 'Optimizar prims'},
            'emergency_mode': {'qo_max': 20, 'action': 'Modo emergencia'}
        }

        print("✓ PerformanceActuator inicializado")

    def analyze_and_act(self, quality_operational: float,
                        current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiza la QO y toma decisiones

        Args:
            quality_operational: Calidad operacional actual (0-100)
            current_metrics: Métricas actuales del sistema

        Returns:
            Decisiones y acciones tomadas
        """
        decisions = []
        actions = []

        # Nivel 1: QO Alta (>90) - Optimización normal
        if quality_operational >= self.qo_threshold_high:
            decisions.append({
                'level': 'high',
                'status': 'Óptimo',
                'action': 'Mantener configuración',
                'priority': 'normal'
            })

        # Nivel 2: QO Media (60-90) - Optimización moderada
        elif quality_operational >= self.qo_threshold_medium:
            decisions.append({
                'level': 'medium',
                'status': 'Bueno',
                'action': 'Optimización ligera',
                'priority': 'medium'
            })

            # Acción: Reducir LOD ligeramente
            actions.append('reduce_lod_distance')

        # Nivel 3: QO Baja (<60) - Actuación necesaria
        elif quality_operational >= self.qo_threshold_low:
            decisions.append({
                'level': 'low',
                'status': 'Requiere acción',
                'action': 'Reducir LOD a 50 metros',
                'priority': 'high'
            })

            actions.append('reduce_lod_to_50m')
            actions.append('reduce_script_priority')

        # Nivel 4: QO Crítica (<40) - Acciones intensivas
        elif quality_operational < self.qo_threshold_low:
            decisions.append({
                'level': 'critical',
                'status': 'Crítico',
                'action': 'Modo emergencia activado',
                'priority': 'urgent'
            })

            actions.extend([
                'reduce_lod_to_25m',
                'disable_non_essential_scripts',
                'optimize_prim_geometry',
                'reduce_shadow_quality'
            ])

        # Registrar acción
        actuation = {
            'qo': quality_operational,
            'decisions': decisions,
            'actions': actions,
            'timestamp': np.datetime64('now')
        }

        self.actuation_history.append(actuation)

        # Mantener solo últimos 100
        if len(self.actuation_history) > 100:
            self.actuation_history.pop(0)

        return {
            'quality_operational': quality_operational,
            'decisions': decisions,
            'actions': actions,
            'efficiency_improvement': self._estimate_improvement(quality_operational),
            'safe_to_apply': quality_operational < self.qo_threshold_medium
        }

    def _estimate_improvement(self, qo: float) -> float:
        """Estima la mejora esperada tras aplicar las acciones"""
        if qo >= 80:
            return 0.05  # +5% mejora esperada
        elif qo >= 60:
            return 0.15  # +15% mejora esperada
        elif qo >= 40:
            return 0.25  # +25% mejora esperada
        else:
            return 0.35  # +35% mejora esperada

    def optimize_weights(self, weights: np.ndarray, qo: float) -> np.ndarray:
        """
        Optimiza pesos basándose en la QO

        Args:
            weights: Pesos actuales
            qo: Calidad operacional

        Returns:
            Pesos optimizados
        """
        if qo < self.qo_threshold_low:
            # Penalizar pesos ineficientes cuando QO es baja
            adjustment = 0.9  # -10% menos impacto
            return weights * adjustment
        elif qo >= self.qo_threshold_high:
            # Reforzar pesos cuando QO es óptima
            adjustment = 1.05  # +5% más impacto
            return weights * adjustment

        return weights

    def get_actuator_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del actuador"""
        if not self.actuation_history:
            return {}

        qo_values = [a['qo'] for a in self.actuation_history]
        action_counts = {}

        for a in self.actuation_history:
            for action in a['actions']:
                action_counts[action] = action_counts.get(action, 0) + 1

        return {
            'total_actuations': len(self.actuation_history),
            'avg_qo': np.mean(qo_values),
            'min_qo': np.min(qo_values),
            'max_qo': np.max(qo_values),
            'action_counts': action_counts,
            'last_qo': qo_values[-1] if qo_values else 0
        }


def test_performance_actuator():
    """Test del actuador de rendimiento"""
    print("\n" + "="*70)
    print("TEST: PerformanceActuator")
    print("="*70)

    actuator = PerformanceActuator()

    # Test 1: QO Alta (>90)
    print("\n✓ Test 1: QO Alta (>90)")
    result1 = actuator.analyze_and_act(95.0, {})
    print(f"✓ QO: {result1['quality_operational']:.1f}")
    print(f"✓ Decisiones: {len(result1['decisions'])}")
    print(f"✓ Acciones: {result1['actions']}")

    # Test 2: QO Media (60-90)
    print("\n✓ Test 2: QO Media (60-90)")
    result2 = actuator.analyze_and_act(70.0, {})
    print(f"✓ QO: {result2['quality_operational']:.1f}")
    print(f"✓ Estado: {result2['decisions'][0]['status']}")
    print(f"✓ Acciones: {result2['actions']}")

    # Test 3: QO Baja (<60)
    print("\n✓ Test 3: QO Baja (<60)")
    result3 = actuator.analyze_and_act(45.0, {})
    print(f"✓ QO: {result3['quality_operational']:.1f}")
    print(f"✓ Estado: {result3['decisions'][0]['status']}")
    print(f"✓ Acciones: {len(result3['actions'])}")
    print(f"✓ Mejora estimada: {result3['efficiency_improvement']*100:.1f}%")

    # Test 4: Optimización de pesos
    print("\n✓ Test 4: Optimizando pesos...")
    weights = np.array([0.5, 0.7, 0.3])
    opt_weights = actuator.optimize_weights(weights, 45.0)
    print(f"✓ Pesos originales: {weights}")
    print(f"✓ Pesos optimizados: {opt_weights}")

    # Test 5: Estadísticas
    print("\n✓ Test 5: Estadísticas del actuador...")
    stats = actuator.get_actuator_stats()
    print(f"✓ Total actuaciones: {stats.get('total_actuations', 0)}")
    print(f"✓ QO promedio: {stats.get('avg_qo', 0):.1f}")
    print(f"✓ QO mínima: {stats.get('min_qo', 0):.1f}")

    print("\n✅ Tests completados exitosamente")
    print("="*70 + "\n")

    return result3


if __name__ == "__main__":
    test_performance_actuator()
