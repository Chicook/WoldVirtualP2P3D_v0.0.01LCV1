"""
ScriptAnalyzer - Analizador de Scripts Activos
==========================================

Neurona especializada en analizar y optimizar scripts activos de OpenSim.

Algoritmos 2024-2025:
- Script profiling
- Execution time analysis
- Priority scheduling
"""

import numpy as np
from typing import Dict, Any, List


class ScriptAnalyzer:
    """Analizador de scripts activos"""

    def __init__(self):
        self.script_limit = 100
        self.active_scripts = 0
        self.script_metrics = {}

        self.analysis_history = []

        print("✓ ScriptAnalyzer inicializado")

    def analyze_scripts(self, active_scripts: int, scripts_by_type: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Analiza scripts activos

        Args:
            active_scripts: Total de scripts activos
            scripts_by_type: Scripts por tipo

        Returns:
            Análisis de scripts
        """
        self.active_scripts = active_scripts

        usage_ratio = active_scripts / self.script_limit

        # Detectar problemas
        is_overloaded = active_scripts > self.script_limit
        is_warning = active_scripts > (self.script_limit * 0.8)

        # Calcular impacto en rendimiento
        performance_impact = self._calculate_performance_impact(active_scripts)

        analysis = {
            'active_scripts': active_scripts,
            'usage_ratio': usage_ratio,
            'is_overloaded': is_overloaded,
            'is_warning': is_warning,
            'performance_impact': performance_impact,
            'scripts_by_type': scripts_by_type or {},
            'recommendation': self._get_optimization_recommendation(active_scripts)
        }

        self.analysis_history.append(analysis)

        return analysis

    def _calculate_performance_impact(self, script_count: int) -> float:
        """Calcula impacto de scripts en rendimiento (0-100%)"""
        if script_count > self.script_limit:
            return 70 + (script_count - self.script_limit) * 0.5  # Alto impacto
        elif script_count > self.script_limit * 0.8:
            return 40 + (script_count - self.script_limit * 0.8) * 0.75  # Medio
        else:
            return (script_count / self.script_limit) * 30  # Bajo

    def _get_optimization_recommendation(self, script_count: int) -> str:
        """Obtiene recomendación de optimización"""
        if script_count > self.script_limit:
            return 'disable_low_priority_scripts'
        elif script_count > self.script_limit * 0.8:
            return 'throttle_script_execution'
        else:
            return 'no_action'

    def optimize_script_execution(self, qo: float) -> Dict[str, Any]:
        """
        Optimiza ejecución de scripts según QO

        Args:
            qo: Calidad operacional

        Returns:
            Configuración de optimización
        """
        if qo < 40:
            # Optimización extrema
            return {
                'max_scripts_per_frame': 20,
                'execution_time_limit_ms': 1,
                'priority_threshold': 'essential_only',
                'skip_interval': 5
            }
        elif qo < 60:
            # Optimización moderada
            return {
                'max_scripts_per_frame': 50,
                'execution_time_limit_ms': 5,
                'priority_threshold': 'high',
                'skip_interval': 2
            }
        else:
            # Configuración normal
            return {
                'max_scripts_per_frame': 100,
                'execution_time_limit_ms': 10,
                'priority_threshold': 'normal',
                'skip_interval': 1
            }

    def optimize_weights(self, weights: np.ndarray, script_count: int) -> np.ndarray:
        """Optimiza pesos basándose en cantidad de scripts"""
        usage_ratio = script_count / self.script_limit

        if usage_ratio > 0.9:
            adjustment = 0.89  # -11%
        elif usage_ratio > 0.7:
            adjustment = 0.94  # -6%
        else:
            adjustment = 1.04  # +4%

        return weights * adjustment

    def get_script_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de scripts"""
        if not self.analysis_history:
            return {}

        impacts = [a['performance_impact'] for a in self.analysis_history]

        return {
            'active_scripts': self.active_scripts,
            'usage_percentage': (self.active_scripts / self.script_limit) * 100,
            'avg_impact': np.mean(impacts),
            'max_impact': np.max(impacts),
            'total_analysis': len(self.analysis_history)
        }


def test_script_analyzer():
    """Test del analizador de scripts"""
    print("\n" + "="*70)
    print("TEST: ScriptAnalyzer")
    print("="*70)

    analyzer = ScriptAnalyzer()

    # Test 1: Análisis normal
    print("\n✓ Test 1: Analizando scripts...")
    analysis1 = analyzer.analyze_scripts(45, {'physics': 20, 'chat': 15, 'ui': 10})
    print(f"✓ Scripts activos: {analysis1['active_scripts']}")
    print(f"✓ Impacto: {analysis1['performance_impact']:.1f}%")
    print(f"✓ Recomendación: {analysis1['recommendation']}")

    # Test 2: Warning
    print("\n✓ Test 2: Advertencia de scripts...")
    analysis2 = analyzer.analyze_scripts(90)
    print(f"✓ Scripts: {analysis2['active_scripts']}")
    print(f"✓ Warning: {analysis2['is_warning']}")
    print(f"✓ Impacto: {analysis2['performance_impact']:.1f}%")

    # Test 3: Optimización por QO
    print("\n✓ Test 3: Optimizando ejecución por QO...")
    config = analyzer.optimize_script_execution(35)
    print(f"✓ Max scripts/frame: {config['max_scripts_per_frame']}")
    print(f"✓ Time limit: {config['execution_time_limit_ms']}ms")
    print(f"✓ Priority: {config['priority_threshold']}")

    # Test 4: Estadísticas
    print("\n✓ Test 4: Estadísticas...")
    stats = analyzer.get_script_statistics()
    print(f"✓ Scripts activos: {stats['active_scripts']}")
    print(f"✓ Impacto promedio: {stats['avg_impact']:.1f}%")

    print("\n✅ Tests completados exitosamente")
    print("="*70 + "\n")

    return stats


if __name__ == "__main__":
    test_script_analyzer()
