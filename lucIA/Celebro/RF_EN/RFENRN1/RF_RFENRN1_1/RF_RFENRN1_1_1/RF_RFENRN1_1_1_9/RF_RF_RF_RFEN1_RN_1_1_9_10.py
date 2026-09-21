"""
Neurona Especializada 10: Punto de Inyección de Pérdida Personalizada (CLIP)
Custom Loss Injection Point

Una estructura de capa que proporciona un hook dedicado para inyectar cálculos
de pérdida auxiliar específicos a la función de la neurona (e.g., una pérdida
de regularidad), esencial para arquitecturas de formación de objetivos múltiples.

La inyección de pérdida permite:
- Agregar objetivos auxiliares durante el entrenamiento
- Regularizar comportamientos específicos
- Implementar arquitecturas multi-task

Autor: Sistema WoldVirtual3DlucIA v0.6.0
Fecha: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List, Dict, Any


class CustomLossInjectionPoint(nn.Module):
    """
    Punto de Inyección de Pérdida Personalizada (CLIP)

    Proporciona un hook dedicado para inyectar cálculos de pérdida auxiliar
    específicos durante el forward pass.

    Args:
        in_features: Número de características de entrada
        out_features: Número de características de salida
        loss_functions: Lista de funciones de pérdida personalizadas (default: None)
        loss_weights: Pesos para cada función de pérdida (default: None)
        use_layer_norm: Si True, aplica LayerNorm (default: True)
        dropout: Tasa de dropout (default: 0.1)
        activation: Función de activación (default: 'relu')

    Input Shape:
        (batch_size, in_features) o (batch_size, *, in_features)

    Output Shape:
        (batch_size, out_features) o (batch_size, *, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        loss_functions: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        loss_weights: Optional[List[float]] = None,
        use_layer_norm: bool = True,
        dropout: float = 0.1,
        activation: str = 'relu'
    ) -> None:
        super(CustomLossInjectionPoint, self).__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features y out_features deben ser positivos")

        self.in_features = in_features
        self.out_features = out_features

        # Capa lineal principal
        self.linear = nn.Linear(in_features, out_features)

        # Normalización de capa
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_features)
        else:
            self.layer_norm = None

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Función de activación
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Activación '{activation}' no soportada")

        # Funciones de pérdida personalizadas
        self.loss_functions = loss_functions if loss_functions is not None else []

        # Pesos de pérdida
        if loss_weights is not None:
            if len(loss_weights) != len(self.loss_functions):
                raise ValueError("loss_weights debe tener la misma longitud que loss_functions")
            self.loss_weights = nn.Parameter(torch.tensor(loss_weights, dtype=torch.float32))
        else:
            self.loss_weights = nn.Parameter(torch.ones(len(self.loss_functions), dtype=torch.float32))

        # Registrar pérdidas calculadas
        self.register_buffer('computed_losses', None)
        self.register_buffer('total_auxiliary_loss', torch.tensor(0.0))

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass con inyección de pérdida personalizada.

        Args:
            x: Tensor de entrada
            target: Tensor objetivo opcional para algunas pérdidas

        Returns:
            Tensor de salida
        """
        # Transformación lineal
        x = self.linear(x)

        # Normalización de capa
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # Activación
        x = self.activation(x)

        # Calcular pérdidas auxiliares si hay funciones de pérdida
        if len(self.loss_functions) > 0:
            computed_losses = []
            for i, loss_fn in enumerate(self.loss_functions):
                # Calcular pérdida
                if target is not None:
                    # Si hay target, pasar ambos
                    loss = loss_fn(x, target)
                else:
                    # Solo pasar la salida
                    loss = loss_fn(x)

                # Aplicar peso
                weighted_loss = self.loss_weights[i] * loss
                computed_losses.append(weighted_loss)

            # Guardar pérdidas calculadas
            self.computed_losses = torch.stack(computed_losses)
            self.total_auxiliary_loss = self.computed_losses.sum()
        else:
            self.computed_losses = None
            self.total_auxiliary_loss = torch.tensor(0.0, device=x.device)

        # Aplicar dropout
        x = self.dropout(x)

        return x

    def get_auxiliary_loss(self) -> torch.Tensor:
        """
        Obtiene la pérdida auxiliar total calculada en el último forward pass.

        Returns:
            Tensor con la pérdida auxiliar total
        """
        return self.total_auxiliary_loss

    def get_individual_losses(self) -> Optional[torch.Tensor]:
        """
        Obtiene las pérdidas individuales calculadas.

        Returns:
            Tensor con pérdidas individuales o None
        """
        return self.computed_losses

    def add_loss_function(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        weight: float = 1.0
    ) -> None:
        """
        Agrega una nueva función de pérdida.

        Args:
            loss_fn: Función de pérdida
            weight: Peso inicial para la pérdida
        """
        self.loss_functions.append(loss_fn)

        # Agregar peso
        current_weights = self.loss_weights.data.tolist()
        current_weights.append(weight)
        self.loss_weights = nn.Parameter(torch.tensor(current_weights, dtype=torch.float32))

    def extra_repr(self) -> str:
        """Representación adicional para debugging"""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'num_loss_functions={len(self.loss_functions)}')


# Funciones de pérdida auxiliares comunes
def l1_regularization(x: torch.Tensor) -> torch.Tensor:
    """Pérdida de regularización L1"""
    return torch.abs(x).mean()


def l2_regularization(x: torch.Tensor) -> torch.Tensor:
    """Pérdida de regularización L2"""
    return (x ** 2).mean()


def sparsity_loss(x: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """Pérdida que favorece dispersión"""
    return (torch.abs(x) < threshold).float().mean()


def smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    """Pérdida que favorece suavidad (diferencias pequeñas entre elementos adyacentes)"""
    if x.dim() == 1:
        return torch.tensor(0.0, device=x.device)
    diff = x[1:] - x[:-1]
    return (diff ** 2).mean()


# Stub de demostración y verificación
if __name__ == "__main__":
    print("=" * 60)
    print("Prueba de CustomLossInjectionPoint (CLIP)")
    print("=" * 60)

    # Configuración
    batch_size = 4
    in_features = 16
    out_features = 8

    # Definir funciones de pérdida personalizadas
    loss_fns = [
        l1_regularization,
        l2_regularization,
        sparsity_loss
    ]

    loss_weights = [0.01, 0.01, 0.1]

    # Crear instancia
    clip = CustomLossInjectionPoint(
        in_features=in_features,
        out_features=out_features,
        loss_functions=loss_fns,
        loss_weights=loss_weights,
        use_layer_norm=True,
        dropout=0.1,
        activation='relu'
    )

    print(f"\nConfiguración:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Input features: {in_features}")
    print(f"  - Output features: {out_features}")
    print(f"  - Loss functions: {len(loss_fns)}")
    print(f"  - Loss weights: {loss_weights}")

    # Crear tensor de entrada
    x = torch.randn(batch_size, in_features, requires_grad=True)
    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min().item():.3f}, {x.max().item():.3f}]")

    # Forward pass
    output = clip(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Obtener pérdidas auxiliares
    aux_loss = clip.get_auxiliary_loss()
    individual_losses = clip.get_individual_losses()

    print(f"\nPérdidas auxiliares:")
    print(f"  - Total auxiliary loss: {aux_loss.item():.6f}")
    if individual_losses is not None:
        print(f"  - Individual losses:")
        for i, loss_val in enumerate(individual_losses):
            print(f"    Loss {i+1} (weighted): {loss_val.item():.6f}")

    # Verificar gradientes
    total_loss = output.sum() + aux_loss
    total_loss.backward()

    print(f"\nVerificación de gradientes:")
    print(f"  - Input grad: {x.grad is not None}")
    print(f"  - Weight grad: {clip.linear.weight.grad is not None}")
    print(f"  - Loss weights grad: {clip.loss_weights.grad is not None}")
    print(f"  - Output grad_fn: {output.grad_fn is not None}")

    if output.grad_fn is not None and x.grad is not None:
        print("  ✅ Gradientes funcionando correctamente")
        print(f"  - Input grad norm: {x.grad.norm().item():.6f}")
    else:
        print("  ❌ Error: No se detectaron gradientes")

    # Agregar nueva función de pérdida
    print("\n" + "-" * 60)
    print("Agregando nueva función de pérdida:")
    print("-" * 60)

    clip.add_loss_function(smoothness_loss, weight=0.05)
    print(f"  - Nuevo número de funciones: {len(clip.loss_functions)}")
    print(f"  - Nuevos pesos: {clip.loss_weights.data.tolist()}")

    # Forward pass con nueva pérdida
    output2 = clip(x)
    aux_loss2 = clip.get_auxiliary_loss()
    print(f"  - Nueva pérdida auxiliar total: {aux_loss2.item():.6f}")

    print("\n" + "=" * 60)
    print("Prueba completada exitosamente")
    print("=" * 60)
