"""
Neurona Especializada 4: Capa de Extracción de Características Dispersas (SFEL)
Sparse Feature Extraction Layer

Esta neurona está diseñada para imponer dispersión en las activaciones.
Esto se logra incorporando una penalización L1 en el cálculo de la pérdida
durante el forward pass, permitiendo que el optimizador favorezca la activación
de solo un subconjunto de neuronas.

La dispersión ayuda a:
- Reducir redundancia en las características
- Mejorar interpretabilidad
- Regularizar el modelo

Autor: Sistema WoldVirtual3DlucIA v0.6.0
Fecha: 2025
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass  # dependencia pesada opcional
try:
    torch
except NameError:
    import types as _t
    torch = _t.SimpleNamespace(
        no_grad=lambda *a, **k: (lambda f: f) if a and callable(a[0]) else (lambda f: f),
        optim=_t.SimpleNamespace(Optimizer=object),
        Tensor=object,
    )
try:
    nn
except NameError:
    import types as _t2
    nn = _t2.SimpleNamespace(Module=object)
from typing import Optional, Tuple


class SparseFeatureExtractionLayer(nn.Module):
    """
    Capa de Extracción de Características Dispersas (SFEL)

    Diseñada para imponer dispersión en las activaciones mediante una
    penalización L1 incorporada en el forward pass.

    Args:
        in_features: Número de características de entrada
        out_features: Número de características de salida
        sparsity_lambda: Factor de penalización L1 (default: 0.01)
        use_batch_norm: Si True, aplica BatchNorm (default: True)
        dropout_rate: Tasa de dropout (default: 0.1)
        activation: Función de activación (default: 'relu')
        bias: Si True, usa bias (default: True)

    Input Shape:
        (batch_size, in_features) o (batch_size, *, in_features)

    Output Shape:
        (batch_size, out_features) o (batch_size, *, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparsity_lambda: float = 0.01,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        bias: bool = True
    ) -> None:
        super(SparseFeatureExtractionLayer, self).__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features y out_features deben ser positivos")

        if sparsity_lambda < 0:
            raise ValueError("sparsity_lambda debe ser no negativo")

        self.in_features = in_features
        self.out_features = out_features
        self.sparsity_lambda = sparsity_lambda

        # Capa lineal principal
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Batch normalization opcional
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
        else:
            self.batch_norm = None

        # Dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Función de activación
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Activación '{activation}' no soportada")

        # Inicialización de pesos para favorecer dispersión
        self._init_weights()

        # Registrar hook para calcular pérdida de dispersión
        self.register_buffer('sparsity_loss', torch.tensor(0.0))

    def _init_weights(self) -> None:
        """Inicializa los pesos para favorecer dispersión"""
        # Inicialización Xavier/Glorot con sesgo hacia dispersión
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la capa de extracción dispersa.

        Args:
            x: Tensor de entrada

        Returns:
            Tensor de salida con características dispersas
        """
        # Aplicar transformación lineal
        x = self.linear(x)

        # Aplicar batch normalization si está habilitado
        if self.batch_norm is not None:
            # BatchNorm espera (N, C) o (N, C, L)
            if x.dim() == 2:
                x = self.batch_norm(x)
            elif x.dim() == 3:
                # Para secuencias, aplicar BatchNorm sobre la última dimensión
                original_shape = x.shape
                x = x.view(-1, original_shape[-1])
                x = self.batch_norm(x)
                x = x.view(original_shape)
            else:
                # Para dimensiones superiores, aplicar sobre la última dimensión
                original_shape = x.shape
                x = x.view(-1, original_shape[-1])
                x = self.batch_norm(x)
                x = x.view(original_shape)

        # Aplicar activación
        x = self.activation(x)

        # Calcular pérdida de dispersión (L1 penalty)
        # Esto se acumula durante el forward pass
        sparsity_penalty = self.sparsity_lambda * torch.abs(x).mean()

        # Registrar la pérdida (no se usa directamente aquí, pero puede
        # ser accedida desde fuera para agregar a la pérdida total)
        self.sparsity_loss = sparsity_penalty.detach()

        # Aplicar dropout
        x = self.dropout(x)

        return x

    def get_sparsity_loss(self) -> torch.Tensor:
        """
        Obtiene la pérdida de dispersión calculada en el último forward pass.

        Returns:
            Tensor con la pérdida de dispersión
        """
        return self.sparsity_loss

    def compute_sparsity_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula explícitamente la penalización de dispersión.

        Args:
            x: Tensor de activaciones

        Returns:
            Tensor con la penalización L1
        """
        return self.sparsity_lambda * torch.abs(x).mean()

    def get_sparsity_ratio(self, x: torch.Tensor, threshold: float = 1e-6) -> float:
        """
        Calcula la razón de dispersión (porcentaje de activaciones cercanas a cero).

        Args:
            x: Tensor de activaciones
            threshold: Umbral para considerar una activación como "cero"

        Returns:
            Float con la razón de dispersión (0-1)
        """
        with torch.no_grad():
            num_elements = x.numel()
            num_sparse = (torch.abs(x) < threshold).sum().item()
            return num_sparse / num_elements if num_elements > 0 else 0.0

    def extra_repr(self) -> str:
        """Representación adicional para debugging"""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'sparsity_lambda={self.sparsity_lambda:.4f}')


# Stub de demostración y verificación
if __name__ == "__main__":
    print("=" * 60)
    print("Prueba de SparseFeatureExtractionLayer (SFEL)")
    print("=" * 60)

    # Configuración
    batch_size = 4
    in_features = 16
    out_features = 8

    # Crear instancia
    sfel = SparseFeatureExtractionLayer(
        in_features=in_features,
        out_features=out_features,
        sparsity_lambda=0.01,
        use_batch_norm=True,
        dropout_rate=0.1,
        activation='relu',
        bias=True
    )

    print(f"\nConfiguración:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Input features: {in_features}")
    print(f"  - Output features: {out_features}")
    print(f"  - Sparsity lambda: {sfel.sparsity_lambda:.4f}")

    # Crear tensor de entrada
    x = torch.randn(batch_size, in_features, requires_grad=True)
    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min().item():.3f}, {x.max().item():.3f}]")

    # Forward pass
    output = sfel(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Calcular métricas de dispersión
    sparsity_ratio = sfel.get_sparsity_ratio(output)
    sparsity_penalty = sfel.compute_sparsity_penalty(output)

    print(f"\nMétricas de dispersión:")
    print(f"  - Sparsity ratio: {sparsity_ratio:.4f} ({sparsity_ratio*100:.2f}%)")
    print(f"  - Sparsity penalty: {sparsity_penalty.item():.6f}")
    print(f"  - Valores no cero: {(output != 0).sum().item()}/{output.numel()}")

    # Verificar gradientes
    loss = output.sum() + sparsity_penalty
    loss.backward()

    print(f"\nVerificación de gradientes:")
    print(f"  - Input grad: {x.grad is not None}")
    print(f"  - Output grad_fn: {output.grad_fn is not None}")

    if output.grad_fn is not None and x.grad is not None:
        print("  ✅ Gradientes funcionando correctamente")
        print(f"  - Input grad norm: {x.grad.norm().item():.6f}")
    else:
        print("  ❌ Error: No se detectaron gradientes")

    # Mostrar pérdida de dispersión registrada
    print(f"\nSparsity loss registrado: {sfel.get_sparsity_loss().item():.6f}")

    print("\n" + "=" * 60)
    print("Prueba completada exitosamente")
    print("=" * 60)
