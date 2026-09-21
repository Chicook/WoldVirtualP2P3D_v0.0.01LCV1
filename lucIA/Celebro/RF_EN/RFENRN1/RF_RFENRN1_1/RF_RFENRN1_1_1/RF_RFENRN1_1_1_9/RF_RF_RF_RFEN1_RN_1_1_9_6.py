"""
Neurona Especializada 6: Fusionador de Incrustaciones Contextuales (CEM)
Contextual Embedding Merger

Esta capa está dedicada a concatenar y normalizar las incrustaciones de
características aprendidas de la red neuronal con datos sensoriales externos
de baja dimensión (e.g., telemetría del entorno Godot), asegurando que ambas
fuentes de datos contribuyan equitativamente a la decisión.

La fusión contextual permite:
- Integrar información de múltiples fuentes
- Normalizar diferentes escalas de datos
- Mantener información tanto de la red como del entorno

Autor: Sistema WoldVirtual3DlucIA v0.6.0
Fecha: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class ContextualEmbeddingMerger(nn.Module):
    """
    Fusionador de Incrustaciones Contextuales (CEM)

    Concatena y normaliza incrustaciones aprendidas con datos sensoriales
    externos, asegurando contribución equitativa de ambas fuentes.

    Args:
        embedding_dim: Dimensión de las incrustaciones aprendidas
        sensor_dim: Dimensión de los datos sensoriales externos
        fusion_method: Método de fusión ('concat', 'add', 'weighted', 'attention') (default: 'concat')
        normalize: Si True, normaliza ambas fuentes antes de fusionar (default: True)
        use_layer_norm: Si True, aplica LayerNorm después de la fusión (default: True)
        dropout: Tasa de dropout (default: 0.1)
        projection_dim: Dimensión de proyección opcional (default: None)

    Input Shape:
        embedding: (batch_size, embedding_dim) o (batch_size, *, embedding_dim)
        sensors: (batch_size, sensor_dim) o (batch_size, *, sensor_dim)

    Output Shape:
        (batch_size, fusion_dim) o (batch_size, *, fusion_dim)
    """

    def __init__(
        self,
        embedding_dim: int,
        sensor_dim: int,
        fusion_method: str = 'concat',
        normalize: bool = True,
        use_layer_norm: bool = True,
        dropout: float = 0.1,
        projection_dim: Optional[int] = None
    ) -> None:
        super(ContextualEmbeddingMerger, self).__init__()

        if embedding_dim <= 0 or sensor_dim <= 0:
            raise ValueError("embedding_dim y sensor_dim deben ser positivos")

        if fusion_method not in ['concat', 'add', 'weighted', 'attention']:
            raise ValueError(f"fusion_method debe ser uno de: 'concat', 'add', 'weighted', 'attention'")

        self.embedding_dim = embedding_dim
        self.sensor_dim = sensor_dim
        self.fusion_method = fusion_method
        self.normalize = normalize

        # Proyección de sensores si es necesario
        if fusion_method == 'add' and embedding_dim != sensor_dim:
            self.sensor_projection = nn.Linear(sensor_dim, embedding_dim)
        elif fusion_method == 'weighted':
            # Pesos aprendibles para cada fuente
            self.embedding_weight = nn.Parameter(torch.tensor(0.5))
            self.sensor_weight = nn.Parameter(torch.tensor(0.5))
            if embedding_dim != sensor_dim:
                self.sensor_projection = nn.Linear(sensor_dim, embedding_dim)
            else:
                self.sensor_projection = nn.Identity()
        elif fusion_method == 'attention':
            # Mecanismo de atención para fusionar
            self.attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=False
            )
            if sensor_dim != embedding_dim:
                self.sensor_projection = nn.Linear(sensor_dim, embedding_dim)
            else:
                self.sensor_projection = nn.Identity()
        else:
            # Para 'concat', no necesitamos proyección
            self.sensor_projection = nn.Identity()

        # Dimensión de salida después de la fusión
        if fusion_method == 'concat':
            fusion_dim = embedding_dim + sensor_dim
        else:
            fusion_dim = embedding_dim

        # Proyección opcional
        if projection_dim is not None:
            self.projection = nn.Linear(fusion_dim, projection_dim)
            output_dim = projection_dim
        else:
            self.projection = nn.Identity()
            output_dim = fusion_dim

        self.output_dim = output_dim

        # Normalización
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        embedding: torch.Tensor,
        sensors: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass del fusionador contextual.

        Args:
            embedding: Tensor de incrustaciones aprendidas
            sensors: Tensor de datos sensoriales externos

        Returns:
            Tensor fusionado y normalizado
        """
        # Normalizar si está habilitado
        if self.normalize:
            embedding = F.layer_norm(embedding, embedding.shape[-1:])
            sensors = F.layer_norm(sensors, sensors.shape[-1:])

        # Proyectar sensores si es necesario
        sensors_proj = self.sensor_projection(sensors)

        # Aplicar método de fusión
        if self.fusion_method == 'concat':
            # Concatenar a lo largo de la última dimensión
            fused = torch.cat([embedding, sensors], dim=-1)

        elif self.fusion_method == 'add':
            # Suma directa (requiere misma dimensión)
            fused = embedding + sensors_proj

        elif self.fusion_method == 'weighted':
            # Combinación ponderada aprendible
            weight_emb = torch.sigmoid(self.embedding_weight)
            weight_sens = torch.sigmoid(self.sensor_weight)
            # Normalizar pesos
            total_weight = weight_emb + weight_sens + 1e-8
            weight_emb = weight_emb / total_weight
            weight_sens = weight_sens / total_weight

            fused = weight_emb * embedding + weight_sens * sensors_proj

        elif self.fusion_method == 'attention':
            # Fusión mediante atención
            # Convertir a formato (seq_len, batch_size, embed_dim) si es necesario
            original_shape = embedding.shape
            if embedding.dim() == 2:
                embedding = embedding.unsqueeze(0)
                sensors_proj = sensors_proj.unsqueeze(0)

            # Aplicar atención: embedding como query, sensors como key y value
            fused, _ = self.attention(embedding, sensors_proj, sensors_proj)

            # Restaurar forma original
            if len(original_shape) == 2:
                fused = fused.squeeze(0)

        # Aplicar proyección
        fused = self.projection(fused)

        # Aplicar normalización de capa
        if self.layer_norm is not None:
            fused = self.layer_norm(fused)

        # Aplicar dropout
        fused = self.dropout(fused)

        return fused

    def get_fusion_weights(self) -> Optional[Tuple[float, float]]:
        """
        Obtiene los pesos de fusión si el método es 'weighted'.

        Returns:
            Tupla (peso_embedding, peso_sensor) o None
        """
        if self.fusion_method == 'weighted':
            weight_emb = torch.sigmoid(self.embedding_weight).item()
            weight_sens = torch.sigmoid(self.sensor_weight).item()
            total = weight_emb + weight_sens + 1e-8
            return (weight_emb / total, weight_sens / total)
        return None

    def extra_repr(self) -> str:
        """Representación adicional para debugging"""
        return (f'embedding_dim={self.embedding_dim}, sensor_dim={self.sensor_dim}, '
                f'fusion_method={self.fusion_method}, output_dim={self.output_dim}, '
                f'normalize={self.normalize}')


# Stub de demostración y verificación
if __name__ == "__main__":
    print("=" * 60)
    print("Prueba de ContextualEmbeddingMerger (CEM)")
    print("=" * 60)

    # Configuración
    batch_size = 4
    embedding_dim = 16
    sensor_dim = 8

    # Probar diferentes métodos de fusión
    fusion_methods = ['concat', 'add', 'weighted', 'attention']

    for method in fusion_methods:
        print(f"\n{'='*60}")
        print(f"Prueba con método: {method}")
        print(f"{'='*60}")

        # Crear instancia
        cem = ContextualEmbeddingMerger(
            embedding_dim=embedding_dim,
            sensor_dim=sensor_dim,
            fusion_method=method,
            normalize=True,
            use_layer_norm=True,
            dropout=0.1,
            projection_dim=None
        )

        print(f"\nConfiguración:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Embedding dim: {embedding_dim}")
        print(f"  - Sensor dim: {sensor_dim}")
        print(f"  - Output dim: {cem.output_dim}")

        # Crear tensores de entrada
        embedding = torch.randn(batch_size, embedding_dim, requires_grad=True)
        sensors = torch.randn(batch_size, sensor_dim, requires_grad=True)

        print(f"\nInput shapes:")
        print(f"  - Embedding: {embedding.shape}")
        print(f"  - Sensors: {sensors.shape}")

        # Forward pass
        output = cem(embedding, sensors)
        print(f"\nOutput shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

        # Verificar gradientes
        loss = output.sum()
        loss.backward()

        print(f"\nVerificación de gradientes:")
        print(f"  - Embedding grad: {embedding.grad is not None}")
        print(f"  - Sensors grad: {sensors.grad is not None}")
        print(f"  - Output grad_fn: {output.grad_fn is not None}")

        if output.grad_fn is not None:
            print("  ✅ Gradientes funcionando correctamente")
        else:
            print("  ❌ Error: No se detectaron gradientes")

        # Mostrar pesos de fusión si aplica
        if method == 'weighted':
            weights = cem.get_fusion_weights()
            if weights:
                print(f"\nPesos de fusión:")
                print(f"  - Embedding weight: {weights[0]:.4f}")
                print(f"  - Sensor weight: {weights[1]:.4f}")

    print("\n" + "=" * 60)
    print("Prueba completada exitosamente")
    print("=" * 60)
