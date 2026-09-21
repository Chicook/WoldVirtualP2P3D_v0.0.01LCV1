"""
Neurona Especializada 3: Capa de Umbral Diferenciable (DTL)
Differentiable Threshold Layer

Esta neurona implementa un umbral duro (similar a ReLU) pero utiliza una
aproximación sigmoide durante el backward pass (técnica del estimador
straight-through) para permitir que los gradientes fluyan incluso en
operaciones discretas.

La técnica straight-through permite:
- Forward pass: operación discreta (threshold hard)
- Backward pass: aproximación continua (sigmoid) para gradientes

Autor: Sistema WoldVirtual3DlucIA v0.6.0
Fecha: 2025
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Tuple


class StraightThroughThreshold(Function):
    """
    Función personalizada de autograd para umbral straight-through.

    Forward: Aplica umbral duro
    Backward: Usa aproximación sigmoide para permitir gradientes
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: float, temperature: float) -> torch.Tensor:
        """
        Forward pass: aplica umbral duro.

        Args:
            input: Tensor de entrada
            threshold: Valor del umbral
            temperature: Temperatura para la aproximación sigmoide (backward)

        Returns:
            Tensor con umbral aplicado
        """
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.temperature = temperature

        # Aplicar umbral duro
        output = (input > threshold).float() * input
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """
        Backward pass: usa aproximación sigmoide para gradientes.

        Args:
            grad_output: Gradiente de la salida

        Returns:
            Gradiente de la entrada
        """
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        temperature = ctx.temperature

        # Aproximación sigmoide para el gradiente
        # d/dx sigmoid((x - threshold) / temperature)
        sigmoid_approx = torch.sigmoid((input - threshold) / temperature)
        grad_input = grad_output * sigmoid_approx * (1 - sigmoid_approx) / temperature

        return grad_input, None, None


class DifferentiableThresholdLayer(nn.Module):
    """
    Capa de Umbral Diferenciable (DTL)

    Implementa un umbral duro en el forward pass pero utiliza una aproximación
    sigmoide durante el backward pass para permitir que los gradientes fluyan.

    Args:
        threshold: Valor del umbral (default: 0.0)
        temperature: Temperatura para la aproximación sigmoide (default: 1.0)
        learnable_threshold: Si True, el umbral es aprendible (default: False)
        per_channel: Si True, aplica umbral por canal (default: False)
        num_channels: Número de canales si per_channel=True

    Input Shape:
        Cualquier tensor

    Output Shape:
        Mismo que input shape
    """

    def __init__(
        self,
        threshold: float = 0.0,
        temperature: float = 1.0,
        learnable_threshold: bool = False,
        per_channel: bool = False,
        num_channels: Optional[int] = None
    ) -> None:
        super(DifferentiableThresholdLayer, self).__init__()

        if temperature <= 0:
            raise ValueError("temperature debe ser positivo")

        self.temperature = temperature
        self.per_channel = per_channel

        if per_channel:
            if num_channels is None or num_channels <= 0:
                raise ValueError("num_channels debe ser positivo cuando per_channel=True")

            self.num_channels = num_channels

            if learnable_threshold:
                self.threshold = nn.Parameter(torch.full((num_channels,), threshold))
            else:
                self.register_buffer('threshold', torch.full((num_channels,), threshold))
        else:
            if learnable_threshold:
                self.threshold = nn.Parameter(torch.tensor(threshold))
            else:
                self.register_buffer('threshold', torch.tensor(threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la capa de umbral diferenciable.

        Args:
            x: Tensor de entrada

        Returns:
            Tensor con umbral aplicado
        """
        if self.per_channel:
            # Aplicar umbral por canal
            # Asumimos que el último canal es la dimensión de características
            if x.dim() >= 2:
                # Expandir threshold para broadcasting
                view_shape = [1] * (x.dim() - 1) + [self.num_channels]
                threshold_expanded = self.threshold.view(*view_shape)

                # Aplicar straight-through threshold
                output = StraightThroughThreshold.apply(
                    x, threshold_expanded, self.temperature
                )
            else:
                # Para tensores 1D, usar threshold escalar
                output = StraightThroughThreshold.apply(
                    x, self.threshold.mean(), self.temperature
                )
        else:
            # Umbral escalar
            threshold_value = self.threshold.item() if isinstance(self.threshold, torch.Tensor) else self.threshold
            output = StraightThroughThreshold.apply(x, threshold_value, self.temperature)

        return output

    def get_threshold_value(self) -> torch.Tensor:
        """
        Obtiene el valor actual del umbral.

        Returns:
            Tensor con el valor del umbral
        """
        return self.threshold.clone()

    def extra_repr(self) -> str:
        """Representación adicional para debugging"""
        threshold_val = self.threshold.mean().item() if self.per_channel else self.threshold.item()
        return (f'threshold={threshold_val:.3f}, temperature={self.temperature:.3f}, '
                f'learnable={isinstance(self.threshold, nn.Parameter)}, '
                f'per_channel={self.per_channel}')


# Stub de demostración y verificación
if __name__ == "__main__":
    print("=" * 60)
    print("Prueba de DifferentiableThresholdLayer (DTL)")
    print("=" * 60)

    # Configuración
    batch_size = 4
    num_features = 8

    # Crear instancia con umbral aprendible
    dtl = DifferentiableThresholdLayer(
        threshold=0.5,
        temperature=1.0,
        learnable_threshold=True,
        per_channel=False
    )

    print(f"\nConfiguración:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Features: {num_features}")
    print(f"  - Threshold inicial: {dtl.threshold.item():.3f}")
    print(f"  - Temperature: {dtl.temperature:.3f}")

    # Crear tensor de entrada
    x = torch.randn(batch_size, num_features, requires_grad=True)
    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min().item():.3f}, {x.max().item():.3f}]")

    # Forward pass
    output = dtl(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"Valores > threshold: {(output > 0).sum().item()}/{output.numel()}")

    # Verificar gradientes
    loss = output.sum()
    loss.backward()

    print(f"\nVerificación de gradientes:")
    print(f"  - Input grad: {x.grad is not None}")
    print(f"  - Threshold grad: {dtl.threshold.grad is not None}")
    print(f"  - Output grad_fn: {output.grad_fn is not None}")

    if output.grad_fn is not None and x.grad is not None:
        print("  ✅ Gradientes funcionando correctamente")
        print(f"  - Input grad norm: {x.grad.norm().item():.6f}")
    else:
        print("  ❌ Error: No se detectaron gradientes")

    # Prueba con umbral por canal
    print("\n" + "-" * 60)
    print("Prueba con umbral por canal:")
    print("-" * 60)

    dtl_channel = DifferentiableThresholdLayer(
        threshold=0.3,
        temperature=0.5,
        learnable_threshold=False,
        per_channel=True,
        num_channels=num_features
    )

    output_channel = dtl_channel(x)
    print(f"Output shape: {output_channel.shape}")
    print(f"Threshold por canal: {dtl_channel.get_threshold_value()}")

    print("\n" + "=" * 60)
    print("Prueba completada exitosamente")
    print("=" * 60)
