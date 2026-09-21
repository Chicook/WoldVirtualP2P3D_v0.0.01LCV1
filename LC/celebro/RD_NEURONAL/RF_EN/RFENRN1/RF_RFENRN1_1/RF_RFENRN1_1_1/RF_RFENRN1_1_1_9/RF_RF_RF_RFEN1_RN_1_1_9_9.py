"""
Neurona Especializada 9: Compuerta de Normalización de Características (FNG)
Feature Normalization Gate

Una compuerta dinámica que normaliza las características de entrada basándose
en estadísticas globales aprendidas, lo cual es vital para gestionar los
cambios de distribución de entrada que provienen de flujos de datos variables
del metaverso.

La normalización adaptativa permite:
- Adaptarse a cambios de distribución (domain adaptation)
- Mantener estabilidad durante entrenamiento
- Gestionar datos no estacionarios

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


class FeatureNormalizationGate(nn.Module):
    """
    Compuerta de Normalización de Características (FNG)

    Normaliza características de entrada basándose en estadísticas globales
    aprendidas, adaptándose a cambios de distribución.

    Args:
        num_features: Número de características
        momentum: Momentum para actualización de estadísticas (default: 0.1)
        eps: Épsilon para estabilidad numérica (default: 1e-5)
        affine: Si True, aplica transformación afín aprendible (default: True)
        track_running_stats: Si True, rastrea estadísticas durante entrenamiento (default: True)
        use_gating: Si True, usa compuerta para controlar normalización (default: True)

    Input Shape:
        (batch_size, num_features) o (batch_size, *, num_features)

    Output Shape:
        Mismo que input shape
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = True,
        use_gating: bool = True
    ) -> None:
        super(FeatureNormalizationGate, self).__init__()

        if num_features <= 0:
            raise ValueError("num_features debe ser positivo")

        if not (0 <= momentum <= 1):
            raise ValueError("momentum debe estar en [0, 1]")

        if eps <= 0:
            raise ValueError("eps debe ser positivo")

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats
        self.use_gating = use_gating

        # Parámetros de transformación afín
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # Estadísticas globales aprendidas
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        # Compuerta opcional para controlar normalización
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(num_features, num_features),
                nn.Sigmoid()
            )
        else:
            self.gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la compuerta de normalización.

        Args:
            x: Tensor de entrada

        Returns:
            Tensor normalizado
        """
        # Detectar forma de entrada
        original_shape = x.shape
        if x.dim() > 2:
            # Para tensores multidimensionales, aplanar excepto última dimensión
            x = x.view(-1, self.num_features)
            reshape_output = True
        else:
            reshape_output = False

        # Calcular estadísticas del batch actual
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Actualizar estadísticas globales si está habilitado
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    if self.momentum is None:
                        # Promedio exponencial
                        exp_avg_factor = 1.0 / self.num_batches_tracked.item()
                    else:
                        exp_avg_factor = self.momentum

                    self.running_mean = (1 - exp_avg_factor) * self.running_mean + exp_avg_factor * batch_mean
                    self.running_var = (1 - exp_avg_factor) * self.running_var + exp_avg_factor * batch_var

            # Usar estadísticas del batch durante entrenamiento
            mean = batch_mean
            var = batch_var
        else:
            # Usar estadísticas globales durante inferencia
            if self.track_running_stats:
                mean = self.running_mean
                var = self.running_var
            else:
                # Si no se rastrean estadísticas, usar estadísticas del batch
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)

        # Normalizar
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Aplicar transformación afín si está habilitada
        if self.affine:
            x_normalized = self.weight * x_normalized + self.bias

        # Aplicar compuerta si está habilitada
        if self.use_gating and self.gate is not None:
            gate_values = self.gate(x_normalized)
            x_normalized = gate_values * x_normalized + (1 - gate_values) * x

        # Restaurar forma original si es necesario
        if reshape_output:
            x_normalized = x_normalized.view(original_shape)

        return x_normalized

    @property
    def affine(self) -> bool:
        """Indica si la transformación afín está habilitada"""
        return self.weight is not None

    def get_running_stats(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Obtiene las estadísticas globales actuales.

        Returns:
            Tupla (mean, var) o None si no se rastrean estadísticas
        """
        if self.track_running_stats:
            return (self.running_mean.clone(), self.running_var.clone())
        return None

    def reset_running_stats(self) -> None:
        """Resetea las estadísticas globales"""
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def extra_repr(self) -> str:
        """Representación adicional para debugging"""
        return (f'num_features={self.num_features}, momentum={self.momentum:.3f}, '
                f'eps={self.eps:.2e}, affine={self.affine}, '
                f'track_running_stats={self.track_running_stats}, use_gating={self.use_gating}')


# Stub de demostración y verificación
if __name__ == "__main__":
    print("=" * 60)
    print("Prueba de FeatureNormalizationGate (FNG)")
    print("=" * 60)

    # Configuración
    batch_size = 4
    num_features = 8

    # Crear instancia
    fng = FeatureNormalizationGate(
        num_features=num_features,
        momentum=0.1,
        eps=1e-5,
        affine=True,
        track_running_stats=True,
        use_gating=True
    )

    print(f"\nConfiguración:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Features: {num_features}")
    print(f"  - Momentum: {fng.momentum:.3f}")
    print(f"  - Use gating: {fng.use_gating}")

    # Crear tensor de entrada
    x = torch.randn(batch_size, num_features, requires_grad=True)
    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min().item():.3f}, {x.max().item():.3f}]")
    print(f"Input mean: {x.mean().item():.3f}")
    print(f"Input std: {x.std().item():.3f}")

    # Forward pass (entrenamiento)
    fng.train()
    output_train = fng(x)
    print(f"\nOutput (training) shape: {output_train.shape}")
    print(f"Output range: [{output_train.min().item():.3f}, {output_train.max().item():.3f}]")
    print(f"Output mean: {output_train.mean().item():.3f}")
    print(f"Output std: {output_train.std().item():.3f}")

    # Verificar estadísticas globales
    running_mean, running_var = fng.get_running_stats()
    print(f"\nRunning statistics:")
    print(f"  - Mean: {running_mean.mean().item():.3f}")
    print(f"  - Var: {running_var.mean().item():.3f}")
    print(f"  - Batches tracked: {fng.num_batches_tracked.item()}")

    # Forward pass (inferencia)
    fng.eval()
    with torch.no_grad():
        output_eval = fng(x)
        print(f"\nOutput (eval) shape: {output_eval.shape}")
        print(f"Output range: [{output_eval.min().item():.3f}, {output_eval.max().item():.3f}]")
        print(f"Output mean: {output_eval.mean().item():.3f}")
        print(f"Output std: {output_eval.std().item():.3f}")

    # Verificar gradientes
    fng.train()
    loss = output_train.sum()
    loss.backward()

    print(f"\nVerificación de gradientes:")
    print(f"  - Input grad: {x.grad is not None}")
    print(f"  - Weight grad: {fng.weight.grad is not None}")
    print(f"  - Bias grad: {fng.bias.grad is not None}")
    print(f"  - Output grad_fn: {output_train.grad_fn is not None}")

    if output_train.grad_fn is not None and x.grad is not None:
        print("  ✅ Gradientes funcionando correctamente")
    else:
        print("  ❌ Error: No se detectaron gradientes")

    # Prueba con tensor multidimensional
    print("\n" + "-" * 60)
    print("Prueba con tensor multidimensional:")
    print("-" * 60)

    x_multi = torch.randn(2, 3, num_features, requires_grad=True)
    output_multi = fng(x_multi)
    print(f"Input shape: {x_multi.shape}")
    print(f"Output shape: {output_multi.shape}")

    print("\n" + "=" * 60)
    print("Prueba completada exitosamente")
    print("=" * 60)
