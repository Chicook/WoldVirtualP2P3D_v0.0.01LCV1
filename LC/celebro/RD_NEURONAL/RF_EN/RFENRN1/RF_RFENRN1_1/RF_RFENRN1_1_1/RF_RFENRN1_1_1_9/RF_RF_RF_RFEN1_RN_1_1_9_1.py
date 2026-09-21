"""
Neurona Especializada 1: Función de Activación Adaptativa (AAF)
Adaptive Activation Function

Esta neurona implementa una función de activación no lineal donde un parámetro
interno aprendido α ajusta dinámicamente la pendiente basándose en la magnitud
de la entrada. El parámetro α es registrable y entrenable dentro del módulo.

Matemáticamente:
    f(x) = α * tanh(x / α) si |x| < threshold
    f(x) = sign(x) * (|x| - threshold + α * tanh(threshold / α)) en otro caso

Donde α es un parámetro aprendido que se ajusta durante el entrenamiento.

Autor: Sistema WoldVirtual3DlucIA v0.6.0
Fecha: 2025
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass  # dependencia pesada opcional
from typing import Optional, Tuple


class AdaptiveActivationFunction(nn.Module):
    """
    Función de Activación Adaptativa (AAF)

    Una activación no lineal donde un parámetro interno aprendido α ajusta
    dinámicamente la pendiente basándose en la magnitud de la entrada.

    Args:
        num_features: Número de características de entrada
        alpha_init: Valor inicial para el parámetro α (default: 1.0)
        alpha_min: Valor mínimo permitido para α (default: 0.1)
        alpha_max: Valor máximo permitido para α (default: 10.0)
        threshold: Umbral para cambio de comportamiento (default: 2.0)
        learnable: Si True, α es entrenable (default: True)

    Input Shape:
        (batch_size, num_features) o (batch_size, *, num_features)

    Output Shape:
        Mismo que input shape
    """

    def __init__(
        self,
        num_features: int,
        alpha_init: float = 1.0,
        alpha_min: float = 0.1,
        alpha_max: float = 10.0,
        threshold: float = 2.0,
        learnable: bool = True
    ) -> None:
        super(AdaptiveActivationFunction, self).__init__()

        if not isinstance(num_features, int) or num_features <= 0:
            raise ValueError("num_features debe ser un entero positivo")

        if alpha_min >= alpha_max:
            raise ValueError("alpha_min debe ser menor que alpha_max")

        if not (alpha_min <= alpha_init <= alpha_max):
            raise ValueError("alpha_init debe estar entre alpha_min y alpha_max")

        self.num_features = num_features
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.threshold = threshold

        # Parámetro α aprendible por característica
        if learnable:
            # Inicializar α como parámetro entrenable
            alpha_param = torch.full((num_features,), alpha_init, dtype=torch.float32)
            self.alpha = nn.Parameter(alpha_param)
        else:
            # α como buffer no entrenable
            self.register_buffer('alpha', torch.full((num_features,), alpha_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la función de activación adaptativa.

        Args:
            x: Tensor de entrada de forma (batch_size, num_features) o
               (batch_size, *, num_features)

        Returns:
            Tensor activado con la misma forma que x
        """
        # Asegurar que α esté en el rango permitido
        alpha_clamped = torch.clamp(self.alpha, self.alpha_min, self.alpha_max)

        # Expandir alpha para broadcasting si es necesario
        if x.dim() > 2:
            # Para tensores multidimensionales, expandir alpha
            view_shape = [1] * (x.dim() - 1) + [self.num_features]
            alpha_expanded = alpha_clamped.view(*view_shape)
        else:
            alpha_expanded = alpha_clamped

        # Calcular máscara para valores dentro del umbral
        abs_x = torch.abs(x)
        mask = abs_x < self.threshold

        # Aplicar función adaptativa
        # Para valores pequeños: α * tanh(x / α)
        small_values = alpha_expanded * torch.tanh(x / (alpha_expanded + 1e-8))

        # Para valores grandes: sign(x) * (|x| - threshold + α * tanh(threshold / α))
        large_values = torch.sign(x) * (
            abs_x - self.threshold +
            alpha_expanded * torch.tanh(self.threshold / (alpha_expanded + 1e-8))
        )

        # Combinar según la máscara
        output = torch.where(mask, small_values, large_values)

        return output

    def get_alpha_values(self) -> torch.Tensor:
        """
        Obtiene los valores actuales de α (clamped).

        Returns:
            Tensor con los valores de α en el rango [alpha_min, alpha_max]
        """
        return torch.clamp(self.alpha, self.alpha_min, self.alpha_max)

    def extra_repr(self) -> str:
        """Representación adicional para debugging"""
        return f'num_features={self.num_features}, alpha_range=[{self.alpha_min}, {self.alpha_max}], threshold={self.threshold}'


# Stub de demostración y verificación
if __name__ == "__main__":
    print("=" * 60)
    print("Prueba de AdaptiveActivationFunction (AAF)")
    print("=" * 60)

    # Configuración
    batch_size = 4
    num_features = 8

    # Crear instancia
    aaf = AdaptiveActivationFunction(
        num_features=num_features,
        alpha_init=1.5,
        alpha_min=0.1,
        alpha_max=5.0,
        threshold=2.0,
        learnable=True
    )

    print(f"\nConfiguración:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Features: {num_features}")
    print(f"  - Alpha inicial: {aaf.alpha.data.mean().item():.3f}")

    # Crear tensor de entrada
    x = torch.randn(batch_size, num_features, requires_grad=True)
    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min().item():.3f}, {x.max().item():.3f}]")

    # Forward pass
    output = aaf(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Verificar gradientes
    loss = output.sum()
    loss.backward()

    print(f"\nVerificación de gradientes:")
    print(f"  - Input grad: {x.grad is not None}")
    print(f"  - Alpha grad: {aaf.alpha.grad is not None}")
    print(f"  - Output grad_fn: {output.grad_fn is not None}")

    if output.grad_fn is not None:
        print("  ✅ Gradientes funcionando correctamente")
    else:
        print("  ❌ Error: No se detectaron gradientes")

    # Mostrar valores de alpha
    alpha_values = aaf.get_alpha_values()
    print(f"\nValores de α (clamped):")
    print(f"  - Min: {alpha_values.min().item():.3f}")
    print(f"  - Max: {alpha_values.max().item():.3f}")
    print(f"  - Mean: {alpha_values.mean().item():.3f}")

    print("\n" + "=" * 60)
    print("Prueba completada exitosamente")
    print("=" * 60)
