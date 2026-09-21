"""
Neurona Especializada 8: Capa de Restricción de Pesos (WCL)
Weight Constraint Layer

Esta capa aplica restricciones específicas a las matrices de pesos, como la
ortogonalidad, para mejorar la estabilidad del entrenamiento. Esto exige la
definición de complejas operaciones de álgebra matricial.

Las restricciones ayudan a:
- Mejorar estabilidad numérica
- Prevenir gradientes explosivos
- Mantener propiedades deseables de las transformaciones

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
from typing import Optional, Literal


class WeightConstraintLayer(nn.Module):
    """
    Capa de Restricción de Pesos (WCL)

    Aplica restricciones específicas a las matrices de pesos para mejorar
    la estabilidad del entrenamiento.

    Args:
        in_features: Número de características de entrada
        out_features: Número de características de salida
        constraint_type: Tipo de restricción ('orthogonal', 'spectral', 'frobenius', 'none') (default: 'orthogonal')
        constraint_strength: Fuerza de la restricción (default: 1.0)
        use_bias: Si True, usa bias (default: True)
        apply_during_training: Si True, aplica restricción durante entrenamiento (default: True)

    Input Shape:
        (batch_size, in_features) o (batch_size, *, in_features)

    Output Shape:
        (batch_size, out_features) o (batch_size, *, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        constraint_type: Literal['orthogonal', 'spectral', 'frobenius', 'none'] = 'orthogonal',
        constraint_strength: float = 1.0,
        use_bias: bool = True,
        apply_during_training: bool = True
    ) -> None:
        super(WeightConstraintLayer, self).__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features y out_features deben ser positivos")

        if constraint_type not in ['orthogonal', 'spectral', 'frobenius', 'none']:
            raise ValueError("constraint_type debe ser 'orthogonal', 'spectral', 'frobenius' o 'none'")

        self.in_features = in_features
        self.out_features = out_features
        self.constraint_type = constraint_type
        self.constraint_strength = constraint_strength
        self.apply_during_training = apply_during_training

        # Capa lineal base
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)

        # Inicialización según el tipo de restricción
        self._init_weights()

    def _init_weights(self) -> None:
        """Inicializa los pesos según el tipo de restricción"""
        if self.constraint_type == 'orthogonal':
            # Inicialización ortogonal
            nn.init.orthogonal_(self.linear.weight)
        elif self.constraint_type == 'spectral':
            # Inicialización espectral (valores singulares controlados)
            nn.init.xavier_uniform_(self.linear.weight)
        elif self.constraint_type == 'frobenius':
            # Inicialización con norma de Frobenius controlada
            nn.init.xavier_uniform_(self.linear.weight)
        else:
            # Inicialización estándar
            nn.init.xavier_uniform_(self.linear.weight)

        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.0)

    def _apply_orthogonal_constraint(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Aplica restricción ortogonal usando descomposición SVD.

        Args:
            weight: Matriz de pesos

        Returns:
            Matriz ortogonalizada
        """
        # Para matrices rectangulares, usar SVD
        if weight.shape[0] != weight.shape[1]:
            U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
            # Reconstruir con valores singulares de 1
            orthogonal_weight = U @ Vt
        else:
            # Para matrices cuadradas, usar método más directo
            U, S, Vt = torch.linalg.svd(weight)
            orthogonal_weight = U @ Vt

        # Interpolar entre peso original y ortogonal según constraint_strength
        constrained_weight = (
            (1 - self.constraint_strength) * weight +
            self.constraint_strength * orthogonal_weight
        )

        return constrained_weight

    def _apply_spectral_constraint(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Aplica restricción espectral (controla el radio espectral).

        Args:
            weight: Matriz de pesos

        Returns:
            Matriz con radio espectral controlado
        """
        # Calcular valor singular máximo
        max_singular = torch.linalg.norm(weight, ord=2)

        # Normalizar si excede el límite
        max_allowed = 1.0 * self.constraint_strength
        if max_singular > max_allowed:
            weight = weight * (max_allowed / (max_singular + 1e-8))

        return weight

    def _apply_frobenius_constraint(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Aplica restricción de norma de Frobenius.

        Args:
            weight: Matriz de pesos

        Returns:
            Matriz con norma de Frobenius controlada
        """
        # Calcular norma de Frobenius
        frobenius_norm = torch.linalg.norm(weight, ord='fro')

        # Normalizar si excede el límite
        max_allowed = self.constraint_strength * (self.in_features ** 0.5)
        if frobenius_norm > max_allowed:
            weight = weight * (max_allowed / (frobenius_norm + 1e-8))

        return weight

    def _apply_constraint(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Aplica la restricción correspondiente según constraint_type.

        Args:
            weight: Matriz de pesos

        Returns:
            Matriz con restricción aplicada
        """
        if self.constraint_type == 'orthogonal':
            return self._apply_orthogonal_constraint(weight)
        elif self.constraint_type == 'spectral':
            return self._apply_spectral_constraint(weight)
        elif self.constraint_type == 'frobenius':
            return self._apply_frobenius_constraint(weight)
        else:
            return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la capa con restricción de pesos.

        Args:
            x: Tensor de entrada

        Returns:
            Tensor de salida
        """
        # Aplicar restricción durante entrenamiento si está habilitado
        if self.training and self.apply_during_training:
            # Guardar peso original
            original_weight = self.linear.weight.data.clone()

            # Aplicar restricción
            constrained_weight = self._apply_constraint(self.linear.weight.data)

            # Actualizar peso
            self.linear.weight.data = constrained_weight

        # Forward pass estándar
        output = self.linear(x)

        return output

    def get_constraint_penalty(self) -> torch.Tensor:
        """
        Calcula la penalización por violación de restricción.

        Returns:
            Tensor con la penalización
        """
        weight = self.linear.weight

        if self.constraint_type == 'orthogonal':
            # Penalización por no ortogonalidad
            if weight.shape[0] == weight.shape[1]:
                # Para matrices cuadradas: ||W^T W - I||_F
                penalty = torch.linalg.norm(weight.T @ weight - torch.eye(weight.shape[0], device=weight.device), ord='fro')
            else:
                # Para matrices rectangulares: usar SVD
                U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
                # Penalización por valores singulares diferentes de 1
                penalty = torch.linalg.norm(S - torch.ones_like(S), ord=2)

        elif self.constraint_type == 'spectral':
            # Penalización por radio espectral excesivo
            max_singular = torch.linalg.norm(weight, ord=2)
            max_allowed = 1.0 * self.constraint_strength
            penalty = F.relu(max_singular - max_allowed)

        elif self.constraint_type == 'frobenius':
            # Penalización por norma de Frobenius excesiva
            frobenius_norm = torch.linalg.norm(weight, ord='fro')
            max_allowed = self.constraint_strength * (self.in_features ** 0.5)
            penalty = F.relu(frobenius_norm - max_allowed)

        else:
            penalty = torch.tensor(0.0, device=weight.device)

        return penalty

    def extra_repr(self) -> str:
        """Representación adicional para debugging"""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'constraint_type={self.constraint_type}, '
                f'constraint_strength={self.constraint_strength:.3f}')


# Stub de demostración y verificación
if __name__ == "__main__":
    print("=" * 60)
    print("Prueba de WeightConstraintLayer (WCL)")
    print("=" * 60)

    # Configuración
    batch_size = 4
    in_features = 16
    out_features = 8

    # Probar diferentes tipos de restricción
    constraint_types = ['orthogonal', 'spectral', 'frobenius', 'none']

    for constraint_type in constraint_types:
        print(f"\n{'='*60}")
        print(f"Prueba con restricción: {constraint_type}")
        print(f"{'='*60}")

        # Crear instancia
        wcl = WeightConstraintLayer(
            in_features=in_features,
            out_features=out_features,
            constraint_type=constraint_type,
            constraint_strength=1.0,
            use_bias=True,
            apply_during_training=True
        )

        print(f"\nConfiguración:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Input features: {in_features}")
        print(f"  - Output features: {out_features}")
        print(f"  - Constraint type: {constraint_type}")

        # Crear tensor de entrada
        x = torch.randn(batch_size, in_features, requires_grad=True)
        print(f"\nInput shape: {x.shape}")

        # Forward pass
        output = wcl(x)
        print(f"\nOutput shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

        # Calcular penalización de restricción
        penalty = wcl.get_constraint_penalty()
        print(f"\nConstraint penalty: {penalty.item():.6f}")

        # Verificar gradientes
        loss = output.sum() + 0.1 * penalty
        loss.backward()

        print(f"\nVerificación de gradientes:")
        print(f"  - Input grad: {x.grad is not None}")
        print(f"  - Weight grad: {wcl.linear.weight.grad is not None}")
        print(f"  - Output grad_fn: {output.grad_fn is not None}")

        if output.grad_fn is not None and x.grad is not None:
            print("  ✅ Gradientes funcionando correctamente")
        else:
            print("  ❌ Error: No se detectaron gradientes")

        # Mostrar propiedades de la matriz de pesos
        weight = wcl.linear.weight
        print(f"\nPropiedades de la matriz de pesos:")
        print(f"  - Shape: {weight.shape}")
        print(f"  - Norm (Frobenius): {torch.linalg.norm(weight, ord='fro').item():.4f}")
        print(f"  - Norm (Spectral): {torch.linalg.norm(weight, ord=2).item():.4f}")

        if constraint_type == 'orthogonal' and weight.shape[0] == weight.shape[1]:
            # Verificar ortogonalidad
            orthogonality = torch.linalg.norm(weight.T @ weight - torch.eye(weight.shape[0]), ord='fro')
            print(f"  - Orthogonality error: {orthogonality.item():.6f}")

    print("\n" + "=" * 60)
    print("Prueba completada exitosamente")
    print("=" * 60)
