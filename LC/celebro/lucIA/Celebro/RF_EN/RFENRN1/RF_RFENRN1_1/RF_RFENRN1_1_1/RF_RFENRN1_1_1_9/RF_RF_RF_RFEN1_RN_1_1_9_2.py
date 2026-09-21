"""
Neurona Especializada 2: Unidad Recurrente con Compuerta de Atención (AGRU)
Attention Gated Recurrent Unit

Esta neurona implementa una modificación de la Unidad Recurrente con Compuerta
(GRU) donde las características de entrada son ponderadas inicialmente por un
mecanismo de autoatención antes del procesamiento recurrente.

La arquitectura combina:
1. Mecanismo de atención multi-head para ponderar características de entrada
2. Procesamiento recurrente GRU estándar
3. Fusión de información temporal y contextual

Autor: Sistema WoldVirtual3DlucIA v0.6.0
Fecha: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AttentionGatedRecurrentUnit(nn.Module):
    """
    Unidad Recurrente con Compuerta de Atención (AGRU)

    Una modificación de GRU donde las características de entrada son
    ponderadas por un mecanismo de autoatención antes del procesamiento
    recurrente.

    Args:
        input_size: Tamaño de las características de entrada
        hidden_size: Tamaño del estado oculto
        num_heads: Número de cabezas de atención (default: 4)
        dropout: Tasa de dropout (default: 0.1)
        num_layers: Número de capas GRU apiladas (default: 1)
        bidirectional: Si True, usa GRU bidireccional (default: False)

    Input Shape:
        (seq_len, batch_size, input_size) o (batch_size, seq_len, input_size)

    Output Shape:
        (batch_size, seq_len, hidden_size) o (seq_len, batch_size, hidden_size)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 1,
        bidirectional: bool = False
    ) -> None:
        super(AttentionGatedRecurrentUnit, self).__init__()

        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size y hidden_size deben ser positivos")

        if num_heads <= 0 or input_size % num_heads != 0:
            raise ValueError("num_heads debe dividir exactamente a input_size")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.head_dim = input_size // num_heads

        # Mecanismo de atención multi-head
        self.attention = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )

        # Capa de normalización para atención
        self.attention_norm = nn.LayerNorm(input_size)

        # Proyección para adaptar input_size a hidden_size si es necesario
        if input_size != hidden_size:
            self.input_projection = nn.Linear(input_size, hidden_size)
        else:
            self.input_projection = nn.Identity()

        # Capa GRU recurrente
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=False
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Factor de escala para estabilidad
        self.scale_factor = 1.0 / (self.head_dim ** 0.5)

    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass de AGRU.

        Args:
            x: Tensor de entrada de forma (seq_len, batch_size, input_size)
               o (batch_size, seq_len, input_size)
            h_0: Estado oculto inicial (opcional)

        Returns:
            output: Tensor de salida de forma (seq_len, batch_size, hidden_size)
            h_n: Estado oculto final
        """
        # Detectar formato de entrada
        batch_first = x.dim() == 3 and x.size(0) != self.input_size

        if batch_first:
            # Convertir a (seq_len, batch_size, input_size)
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape

        # Aplicar atención multi-head a las características de entrada
        # La atención se aplica sobre la dimensión de secuencia
        x_attended, attention_weights = self.attention(x, x, x)

        # Residual connection y normalización
        x_attended = self.attention_norm(x_attended + x)
        x_attended = self.dropout(x_attended)

        # Proyectar a hidden_size si es necesario
        x_projected = self.input_projection(x_attended)

        # Inicializar estado oculto si no se proporciona
        if h_0 is None:
            num_directions = 2 if self.bidirectional else 1
            h_0 = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype
            )

        # Procesamiento recurrente con GRU
        gru_output, h_n = self.gru(x_projected, h_0)

        # Aplicar dropout al output
        gru_output = self.dropout(gru_output)

        # Convertir de vuelta a batch_first si era el formato original
        if batch_first:
            gru_output = gru_output.transpose(0, 1)

        return gru_output, h_n

    def get_attention_weights(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Obtiene los pesos de atención sin procesamiento recurrente.

        Args:
            x: Tensor de entrada

        Returns:
            Tensor de pesos de atención
        """
        batch_first = x.dim() == 3 and x.size(0) != self.input_size

        if batch_first:
            x = x.transpose(0, 1)

        _, attention_weights = self.attention(x, x, x)
        return attention_weights

    def extra_repr(self) -> str:
        """Representación adicional para debugging"""
        return (f'input_size={self.input_size}, hidden_size={self.hidden_size}, '
                f'num_heads={self.num_heads}, num_layers={self.num_layers}, '
                f'bidirectional={self.bidirectional}')


# Stub de demostración y verificación
if __name__ == "__main__":
    print("=" * 60)
    print("Prueba de AttentionGatedRecurrentUnit (AGRU)")
    print("=" * 60)

    # Configuración
    batch_size = 2
    seq_len = 10
    input_size = 16
    hidden_size = 32

    # Crear instancia
    agru = AttentionGatedRecurrentUnit(
        input_size=input_size,
        hidden_size=hidden_size,
        num_heads=4,
        dropout=0.1,
        num_layers=1,
        bidirectional=False
    )

    print(f"\nConfiguración:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Input size: {input_size}")
    print(f"  - Hidden size: {hidden_size}")

    # Crear tensor de entrada (seq_len, batch_size, input_size)
    x = torch.randn(seq_len, batch_size, input_size, requires_grad=True)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output, h_n = agru(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Hidden state shape: {h_n.shape}")

    # Verificar gradientes
    loss = output.sum()
    loss.backward()

    print(f"\nVerificación de gradientes:")
    print(f"  - Input grad: {x.grad is not None}")
    print(f"  - Output grad_fn: {output.grad_fn is not None}")

    if output.grad_fn is not None:
        print("  ✅ Gradientes funcionando correctamente")
    else:
        print("  ❌ Error: No se detectaron gradientes")

    # Obtener pesos de atención
    attention_weights = agru.get_attention_weights(x)
    print(f"\nAttention weights shape: {attention_weights.shape}")
    print(f"  - Min: {attention_weights.min().item():.4f}")
    print(f"  - Max: {attention_weights.max().item():.4f}")

    print("\n" + "=" * 60)
    print("Prueba completada exitosamente")
    print("=" * 60)
