"""
DistributedStorageNeuron - Almacenamiento Distribuido
===========================================

Neurona especializada en gestionar almacenamiento distribuido de assets.

Funcionalidad:
- Replicación de assets
- Gestión de espacio
- Priorización de assets críticos
"""

import numpy as np
from typing import Dict, Any, List


class DistributedStorageNeuron:
    """Gestiona almacenamiento distribuido"""

    def __init__(self):
        self.stored_assets = {}
        self.replication_history = []

        print("✓ DistributedStorageNeuron inicializado")

    def decide_replication(self, asset: Dict[str, Any],
                           available_storage_mb: float) -> Dict[str, Any]:
        """
        Decide si replicar un asset

        Args:
            asset: Información del asset
            available_storage_mb: Almacenamiento disponible

        Returns:
            Decisión de replicación
        """
        asset_size_mb = asset.get('size_mb', 1)
        priority = asset.get('priority', 'normal')

        should_replicate = False
        replica_count = 0

        if priority == 'critical':
            if available_storage_mb > asset_size_mb * 3:
                should_replicate = True
                replica_count = 2
        elif priority == 'high':
            if available_storage_mb > asset_size_mb * 2:
                should_replicate = True
                replica_count = 1

        decision = {
            'asset_id': asset['id'],
            'should_replicate': should_replicate,
            'replica_count': replica_count,
            'required_storage_mb': asset_size_mb * (replica_count + 1),
            'available_storage_mb': available_storage_mb
        }

        if should_replicate:
            self.replication_history.append(decision)

        return decision

    def optimize_weights(self, weights: np.ndarray, storage_usage_percent: float) -> np.ndarray:
        """Optimiza pesos según uso de almacenamiento"""
        if storage_usage_percent > 80:
            return weights * 0.94  # -6%
        elif storage_usage_percent > 60:
            return weights * 0.97  # -3%
        else:
            return weights * 1.04  # +4%

    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        return {
            'total_assets': len(self.stored_assets),
            'total_replications': len(self.replication_history),
            'replication_rate': len(self.replication_history) / len(self.stored_assets)
            if self.stored_assets else 0
        }


def test_distributed_storage_neuron():
    """Test de la neurona de almacenamiento distribuido"""
    print("\n" + "="*70)
    print("TEST: DistributedStorageNeuron")
    print("="*70)

    neuron = DistributedStorageNeuron()

    # Test 1: Decidir replicación
    print("\n✓ Test 1: Decidiendo replicación...")
    asset = {'id': 'asset_001', 'size_mb': 50, 'priority': 'critical'}
    decision = neuron.decide_replication(asset, 200)
    print(f"✓ Replicar: {decision['should_replicate']}")
    print(f"✓ Réplicas: {decision['replica_count']}")
    print(f"✓ Almacenamiento requerido: {decision['required_storage_mb']}MB")

    print("\n✅ Tests completados exitosamente")
    print("="*70 + "\n")

    return decision


if __name__ == "__main__":
    test_distributed_storage_neuron()
