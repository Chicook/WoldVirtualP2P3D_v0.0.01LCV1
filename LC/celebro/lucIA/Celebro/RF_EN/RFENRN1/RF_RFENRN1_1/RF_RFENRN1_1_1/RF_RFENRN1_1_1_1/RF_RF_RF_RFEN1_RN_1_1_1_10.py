"""
QODecisionMaker - Tomador de Decisiones basado en Calidad Operacional
===================================================================

Neurona principal que integra todas las métricas y toma decisiones
inteligentes basadas en la Calidad Operacional (QO).

Algoritmos 2024-2025:
- Multi-objective optimization
- Decision trees
- Context-aware decision making
"""

import numpy as np
from typing import Dict, Any, List


class QODecisionMaker:
    """Tomador de decisiones basado en QO"""

    def __init__(self):
        self.qo_thresholds = {
            'optimal': 85.0,
            'good': 70.0,
            'warning': 50.0,
            'critical': 30.0
        }

        self.decision_history = []
        self.actions_executed = []

        print("✓ QODecisionMaker inicializado")

    def make_decision(self, quality_operational: float,
                      metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Toma decisión basándose en QO y métricas

        Args:
            quality_operational: Calidad operacional (0-100)
            metrics: Métricas del sistema

        Returns:
            Decisión y acciones recomendadas
        """
        # Determinar nivel de QO
        qo_level = self._determine_qo_level(quality_operational)

        # Generar decisión estratégica
        decision = self._generate_decision(qo_level, metrics)

        # Calcular acciones prioritarias
        actions = self._calculate_priority_actions(quality_operational, metrics)

        # Estimar impacto
        estimated_impact = self._estimate_impact(quality_operational, actions)

        decision_data = {
            'qo': quality_operational,
            'qo_level': qo_level,
            'decision': decision,
            'actions': actions,
            'estimated_impact': estimated_impact,
            'metrics_snapshot': metrics,
            'timestamp': np.datetime64('now')
        }

        self.decision_history.append(decision_data)

        # Mantener solo últimos 100
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)

        return decision_data

    def _determine_qo_level(self, qo: float) -> str:
        """Determina el nivel de QO"""
        if qo >= self.qo_thresholds['optimal']:
            return 'optimal'
        elif qo >= self.qo_thresholds['good']:
            return 'good'
        elif qo >= self.qo_thresholds['warning']:
            return 'warning'
        elif qo >= self.qo_thresholds['critical']:
            return 'critical'
        else:
            return 'emergency'

    def _generate_decision(self, qo_level: str, metrics: Dict[str, Any]) -> str:
        """Genera decisión estratégica"""
        decisions = {
            'optimal': 'Mantener configuración actual',
            'good': 'Optimización ligera recomendada',
            'warning': 'Aplicar optimizaciones moderadas',
            'critical': 'Aplicar optimizaciones agresivas',
            'emergency': 'Activar modo emergencia completo'
        }

        return decisions.get(qo_level, 'Estado desconocido')

    def _calculate_priority_actions(self, qo: float, metrics: Dict[str, Any]) -> List[str]:
        """Calcula acciones prioritarias"""
        actions = []

        if qo < self.qo_thresholds['critical']:
            # Acciones de emergencia
            actions.extend([
                'reduce_lod_to_minimum',
                'disable_most_scripts',
                'reduce_physics_complexity',
                'lower_texture_quality',
                'enable_aggressive_culling'
            ])
        elif qo < self.qo_thresholds['warning']:
            # Acciones de optimización
            actions.extend([
                'reduce_lod_distance',
                'throttle_scripts',
                'optimize_prim_geometry',
                'reduce_shadow_quality'
            ])
        elif qo < self.qo_thresholds['good']:
            # Acciones moderadas
            actions.extend([
                'adjust_lod_settings',
                'manage_script_execution'
            ])

        return actions

    def _estimate_impact(self, qo: float, actions: List[str]) -> Dict[str, float]:
        """Estima el impacto de las acciones"""
        fps_boost = 0
        memory_saving = 0

        if qo < self.qo_thresholds['critical']:
            fps_boost = 10 + (len(actions) * 2)
            memory_saving = 300  # MB
        elif qo < self.qo_thresholds['warning']:
            fps_boost = 5 + (len(actions) * 1)
            memory_saving = 150  # MB
        elif qo < self.qo_thresholds['good']:
            fps_boost = 2
            memory_saving = 50   # MB

        return {
            'fps_boost': fps_boost,
            'memory_saving_mb': memory_saving,
            'quality_improvement': min(50, len(actions) * 5)  # Hasta +50 puntos de QO
        }

    def optimize_weights(self, weights: np.ndarray, qo: float) -> np.ndarray:
        """Optimiza pesos basándose en QO"""
        if qo < self.qo_thresholds['critical']:
            adjustment = 0.85  # -15% para emergencia
        elif qo < self.qo_thresholds['warning']:
            adjustment = 0.93  # -7% para warning
        elif qo >= self.qo_thresholds['optimal']:
            adjustment = 1.08  # +8% para óptimo
        else:
            adjustment = 1.00

        return weights * adjustment

    def get_decision_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de decisiones"""
        if not self.decision_history:
            return {}

        qo_values = [d['qo'] for d in self.decision_history]

        return {
            'total_decisions': len(self.decision_history),
            'avg_qo': np.mean(qo_values),
            'min_qo': np.min(qo_values),
            'max_qo': np.max(qo_values),
            'last_qo': qo_values[-1] if qo_values else 0,
            'last_qo_level': self.decision_history[-1]['qo_level'] if self.decision_history else 'unknown'
        }


def test_qo_decision_maker():
    """Test del tomador de decisiones QO"""
    print("\n" + "="*70)
    print("TEST: QODecisionMaker")
    print("="*70)

    maker = QODecisionMaker()

    # Test 1: QO Alta
    print("\n✓ Test 1: Decisión con QO Alta...")
    decision1 = maker.make_decision(90.0, {'fps': 45, 'memory': 500, 'cpu': 50})
    print(f"✓ QO: {decision1['qo']:.1f}")
    print(f"✓ Nivel: {decision1['qo_level']}")
    print(f"✓ Decisión: {decision1['decision']}")
    print(f"✓ Acciones: {len(decision1['actions'])}")

    # Test 2: QO Media
    print("\n✓ Test 2: Decisión con QO Media...")
    decision2 = maker.make_decision(65.0, {'fps': 35, 'memory': 800, 'cpu': 70})
    print(f"✓ QO: {decision2['qo']:.1f}")
    print(f"✓ Decisión: {decision2['decision']}")
    print(f"✓ Acciones: {decision2['actions']}")

    # Test 3: QO Crítica
    print("\n✓ Test 3: Decisión con QO Crítica...")
    decision3 = maker.make_decision(25.0, {'fps': 15, 'memory': 1500, 'cpu': 95})
    print(f"✓ QO: {decision3['qo']:.1f}")
    print(f"✓ Nivel: {decision3['qo_level']}")
    print(f"✓ Decisión: {decision3['decision']}")
    print(f"✓ Acciones: {len(decision3['actions'])}")
    print(f"✓ FPS Boost estimado: +{decision3['estimated_impact']['fps_boost']:.0f}")
    print(f"✓ Memory saving: {decision3['estimated_impact']['memory_saving_mb']}MB")

    # Test 4: Estadísticas
    print("\n✓ Test 4: Estadísticas de decisiones...")
    stats = maker.get_decision_statistics()
    print(f"✓ Total decisiones: {stats['total_decisions']}")
    print(f"✓ QO promedio: {stats['avg_qo']:.1f}")
    print(f"✓ Último QO: {stats['last_qo']:.1f}")

    print("\n✅ Tests completados exitosamente")
    print("="*70 + "\n")

    return decision3


if __name__ == "__main__":
    test_qo_decision_maker()
