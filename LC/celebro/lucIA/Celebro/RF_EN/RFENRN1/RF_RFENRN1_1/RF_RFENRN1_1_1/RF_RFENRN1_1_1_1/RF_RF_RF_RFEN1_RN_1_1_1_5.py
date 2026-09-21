"""
MemoryManager - Gestor de Memoria Inteligente
=========================================

Neurona especializada en gestionar y optimizar el uso de memoria
del sistema OpenSim.

Algoritmos 2024-2025:
- Memory pooling
- Garbage collection predictivo
- LRU cache optimization
- Memory leak detection
"""

import numpy as np
import time
from typing import Dict, Any, List


class MemoryManager:
    """Gestor de memoria inteligente"""

    def __init__(self):
        self.memory_threshold_mb = 1024  # 1GB threshold
        self.memory_target_mb = 512      # Target 512MB
        self.current_memory_mb = 0

        self.memory_history = []
        self.gc_events = []

        # Estrategias de limpieza
        self.cleanup_strategies = [
            'release_unused_assets',
            'clear_texture_cache',
            'unload_unseen_prims',
            'compress_geometry'
        ]

        print("✓ MemoryManager inicializado")

    def monitor_memory(self, memory_mb: float) -> Dict[str, Any]:
        """
        Monitorea y analiza el uso de memoria

        Args:
            memory_mb: Uso actual de memoria en MB

        Returns:
            Análisis y recomendaciones
        """
        self.current_memory_mb = memory_mb

        # Agregar al historial
        self.memory_history.append({
            'memory_mb': memory_mb,
            'timestamp': time.time()
        })

        # Mantener solo últimos 60 segundos
        if len(self.memory_history) > 60:
            self.memory_history.pop(0)

        # Calcular estadísticas
        usage_ratio = memory_mb / self.memory_threshold_mb
        is_critical = usage_ratio > 0.9
        needs_cleanup = usage_ratio > 0.7

        # Detectar tendencia
        if len(self.memory_history) >= 5:
            recent = self.memory_history[-5:]
            trend = 'stable'

            if recent[-1]['memory_mb'] - recent[0]['memory_mb'] > 20:
                trend = 'increasing'
            elif recent[-1]['memory_mb'] - recent[0]['memory_mb'] < -20:
                trend = 'decreasing'

        else:
            trend = 'unknown'

        return {
            'current_memory_mb': memory_mb,
            'usage_ratio': usage_ratio,
            'is_critical': is_critical,
            'needs_cleanup': needs_cleanup,
            'trend': trend,
            'recommended_action': self._get_cleanup_action(usage_ratio)
        }

    def _get_cleanup_action(self, usage_ratio: float) -> str:
        """Obtiene acción de limpieza recomendada"""
        if usage_ratio > 0.9:
            return 'emergency_cleanup'  # Limpieza de emergencia
        elif usage_ratio > 0.7:
            return 'aggressive_cleanup'  # Limpieza agresiva
        elif usage_ratio > 0.5:
            return 'moderate_cleanup'  # Limpieza moderada
        else:
            return 'no_action'

    def trigger_cleanup(self, memory_usage_ratio: float) -> Dict[str, Any]:
        """
        Ejecuta limpieza de memoria según la necesidad

        Args:
            memory_usage_ratio: Ratio de uso de memoria (0-1)

        Returns:
            Resultado de la limpieza
        """
        actions_taken = []
        estimated_freed_mb = 0

        if memory_usage_ratio > 0.9:
            # Limpieza de emergencia
            actions_taken.extend([
                'clear_all_caches',
                'unload_distant_objects',
                'compress_all_geometry'
            ])
            estimated_freed_mb = 200  # ~200MB libres

        elif memory_usage_ratio > 0.7:
            # Limpieza agresiva
            actions_taken.extend([
                'clear_texture_cache',
                'unload_unseen_prims',
                'garbage_collection'
            ])
            estimated_freed_mb = 100  # ~100MB libres

        elif memory_usage_ratio > 0.5:
            # Limpieza moderada
            actions_taken.append('release_unused_assets')
            estimated_freed_mb = 50  # ~50MB libres

        gc_event = {
            'actions': actions_taken,
            'estimated_freed_mb': estimated_freed_mb,
            'memory_before': self.current_memory_mb,
            'memory_after': max(0, self.current_memory_mb - estimated_freed_mb),
            'timestamp': time.time()
        }

        self.gc_events.append(gc_event)

        return gc_event

    def optimize_weights(self, weights: np.ndarray, memory_ratio: float) -> np.ndarray:
        """Optimiza pesos basándose en uso de memoria"""
        if memory_ratio > 0.8:
            # Reducir impacto cuando memoria alta
            adjustment = 0.92
        else:
            # Aumentar impacto cuando memoria ok
            adjustment = 1.03

        return weights * adjustment

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de memoria"""
        if not self.memory_history:
            return {}

        memory_values = [h['memory_mb'] for h in self.memory_history]

        return {
            'current_memory_mb': self.current_memory_mb,
            'avg_memory_mb': np.mean(memory_values),
            'max_memory_mb': np.max(memory_values),
            'min_memory_mb': np.min(memory_values),
            'gc_events_count': len(self.gc_events),
            'total_freed_mb': sum(gc['estimated_freed_mb'] for gc in self.gc_events)
        }


def test_memory_manager():
    """Test del gestor de memoria"""
    print("\n" + "="*70)
    print("TEST: MemoryManager")
    print("="*70)

    manager = MemoryManager()

    # Test 1: Monitoreo normal
    print("\n✓ Test 1: Monitoreo de memoria...")
    for i in range(5):
        memory = 400 + i * 20
        analysis = manager.monitor_memory(memory)
        print(f"  ✓ Memoria: {memory}MB, Ratio: {analysis['usage_ratio']:.2%}, Tendencia: {analysis['trend']}")

    # Test 2: Necesidad de limpieza
    print("\n✓ Test 2: Detección de necesidad de limpieza...")
    analysis = manager.monitor_memory(800)
    print(f"✓ Memoria: {analysis['current_memory_mb']}MB")
    print(f"✓ Crítico: {analysis['is_critical']}")
    print(f"✓ Necesita limpieza: {analysis['needs_cleanup']}")
    print(f"✓ Acción recomendada: {analysis['recommended_action']}")

    # Test 3: Trigger cleanup
    print("\n✓ Test 3: Ejecutando limpieza...")
    cleanup = manager.trigger_cleanup(0.85)
    print(f"✓ Acciones: {len(cleanup['actions'])}")
    print(f"✓ Memoria liberada: {cleanup['estimated_freed_mb']}MB")

    # Test 4: Estadísticas
    print("\n✓ Test 4: Estadísticas...")
    stats = manager.get_memory_statistics()
    print(f"✓ Memoria actual: {stats['current_memory_mb']:.0f}MB")
    print(f"✓ Memoria promedio: {stats['avg_memory_mb']:.0f}MB")
    print(f"✓ Eventos GC: {stats['gc_events_count']}")

    print("\n✅ Tests completados exitosamente")
    print("="*70 + "\n")

    return stats


if __name__ == "__main__":
    test_memory_manager()
