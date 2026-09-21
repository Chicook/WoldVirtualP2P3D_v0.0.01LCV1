"""
Neurona Especializada 5: Módulo de Memoria de Decaimiento Temporal (TDMM)
Temporal Decay Memory Module

Este módulo simula la pérdida de memoria multiplicando los estados celulares
internos por un factor de decaimiento exponencial, dependiente del número de
pasos de tiempo, siguiendo la fórmula:

    M_t = M_{t-1} * exp(-λ * t)

Donde:
    M_t: Memoria en el tiempo t
    λ: Tasa de decaimiento
    t: Paso de tiempo

Esto permite modelar el olvido gradual de información a lo largo del tiempo.

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


class TemporalDecayMemoryModule(nn.Module):
    """
    Módulo de Memoria de Decaimiento Temporal (TDMM)

    Simula la pérdida de memoria mediante decaimiento exponencial de los
    estados internos a lo largo del tiempo.

    Args:
        memory_size: Tamaño del vector de memoria
        decay_rate: Tasa de decaimiento inicial λ (default: 0.1)
        learnable_decay: Si True, la tasa de decaimiento es aprendible (default: True)
        min_decay: Tasa de decaimiento mínima permitida (default: 0.001)
        max_decay: Tasa de decaimiento máxima permitida (default: 1.0)
        use_gating: Si True, usa compuerta para controlar el decaimiento (default: True)
        reset_on_new_sequence: Si True, resetea memoria al inicio de nueva secuencia (default: True)

    Input Shape:
        (batch_size, memory_size) o (seq_len, batch_size, memory_size)

    Output Shape:
        Mismo que input shape
    """

    def __init__(
        self,
        memory_size: int,
        decay_rate: float = 0.1,
        learnable_decay: bool = True,
        min_decay: float = 0.001,
        max_decay: float = 1.0,
        use_gating: bool = True,
        reset_on_new_sequence: bool = True
    ) -> None:
        super(TemporalDecayMemoryModule, self).__init__()

        if memory_size <= 0:
            raise ValueError("memory_size debe ser positivo")

        if not (min_decay < decay_rate < max_decay):
            raise ValueError(f"decay_rate debe estar entre {min_decay} y {max_decay}")

        self.memory_size = memory_size
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.use_gating = use_gating
        self.reset_on_new_sequence = reset_on_new_sequence

        # Tasa de decaimiento aprendible o fija
        if learnable_decay:
            # Usar sigmoid para mantener en rango [min_decay, max_decay]
            self.decay_logit = nn.Parameter(torch.tensor(self._inverse_sigmoid(decay_rate, min_decay, max_decay)))
        else:
            self.register_buffer('decay_logit', torch.tensor(self._inverse_sigmoid(decay_rate, min_decay, max_decay)))

        # Compuerta opcional para controlar el decaimiento
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(memory_size, memory_size),
                nn.Sigmoid()
            )
        else:
            self.gate = None

        # Estado de memoria (se inicializa en el primer forward)
        self.register_buffer('memory_state', None)
        self.register_buffer('time_step', torch.tensor(0, dtype=torch.long))

    @staticmethod
    def _inverse_sigmoid(value: float, min_val: float, max_val: float) -> float:
        """
        Convierte un valor en rango [min_val, max_val] a un logit para sigmoid.

        Args:
            value: Valor en el rango [min_val, max_val]
            min_val: Valor mínimo
            max_val: Valor máximo

        Returns:
            Logit correspondiente
        """
        # Normalizar a [0, 1]
        normalized = (value - min_val) / (max_val - min_val)
        # Convertir a logit (inverso de sigmoid)
        logit = torch.log(torch.tensor(normalized / (1 - normalized + 1e-8)))
        return logit.item()

    def _get_decay_rate(self) -> torch.Tensor:
        """
        Obtiene la tasa de decaimiento actual (clamped).

        Returns:
            Tensor con la tasa de decaimiento
        """
        # Aplicar sigmoid y escalar al rango [min_decay, max_decay]
        sigmoid_val = torch.sigmoid(self.decay_logit)
        decay = self.min_decay + sigmoid_val * (self.max_decay - self.min_decay)
        return decay

    def reset_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        Resetea el estado de memoria.

        Args:
            batch_size: Tamaño del batch
            device: Dispositivo del tensor
            dtype: Tipo de dato
        """
        self.memory_state = torch.zeros(batch_size, self.memory_size, device=device, dtype=dtype)
        self.time_step = torch.tensor(0, dtype=torch.long, device=device)

    def forward(
        self,
        x: torch.Tensor,
        reset: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Forward pass del módulo de memoria con decaimiento temporal.

        Args:
            x: Tensor de entrada de forma (batch_size, memory_size) o
               (seq_len, batch_size, memory_size)
            reset: Si True, resetea la memoria (default: None, usa reset_on_new_sequence)

        Returns:
            Tensor de memoria actualizada con la misma forma que x
        """
        is_sequence = x.dim() == 3
        batch_size = x.size(0) if not is_sequence else x.size(1)
        device = x.device
        dtype = x.dtype

        # Determinar si resetear
        should_reset = reset if reset is not None else self.reset_on_new_sequence

        # Inicializar o resetear memoria si es necesario
        if self.memory_state is None or should_reset:
            self.reset_memory(batch_size, device, dtype)

        # Obtener tasa de decaimiento
        decay_rate = self._get_decay_rate()

        if is_sequence:
            # Procesar secuencia paso a paso
            seq_len = x.size(0)
            outputs = []

            for t in range(seq_len):
                # Actualizar paso de tiempo
                self.time_step += 1

                # Calcular factor de decaimiento exponencial
                decay_factor = torch.exp(-decay_rate * self.time_step.float())

                # Aplicar decaimiento a la memoria anterior
                decayed_memory = self.memory_state * decay_factor

                # Combinar con entrada actual
                if self.use_gating and self.gate is not None:
                    gate_values = self.gate(x[t])
                    combined = gate_values * x[t] + (1 - gate_values) * decayed_memory
                else:
                    combined = x[t] + decayed_memory

                # Actualizar estado de memoria
                self.memory_state = combined
                outputs.append(combined)

            # Apilar outputs
            output = torch.stack(outputs, dim=0)
        else:
            # Procesar un solo paso
            self.time_step += 1

            # Calcular factor de decaimiento exponencial
            decay_factor = torch.exp(-decay_rate * self.time_step.float())

            # Aplicar decaimiento a la memoria anterior
            decayed_memory = self.memory_state * decay_factor

            # Combinar con entrada actual
            if self.use_gating and self.gate is not None:
                gate_values = self.gate(x)
                combined = gate_values * x + (1 - gate_values) * decayed_memory
            else:
                combined = x + decayed_memory

            # Actualizar estado de memoria
            self.memory_state = combined
            output = combined

        return output

    def get_memory_state(self) -> Optional[torch.Tensor]:
        """
        Obtiene el estado actual de la memoria.

        Returns:
            Tensor con el estado de memoria o None si no está inicializado
        """
        return self.memory_state

    def get_decay_rate(self) -> float:
        """
        Obtiene la tasa de decaimiento actual.

        Returns:
            Float con la tasa de decaimiento
        """
        return self._get_decay_rate().item()

    def extra_repr(self) -> str:
        """Representación adicional para debugging"""
        decay = self._get_decay_rate().item()
        return (f'memory_size={self.memory_size}, decay_rate={decay:.4f}, '
                f'use_gating={self.use_gating}, reset_on_new_sequence={self.reset_on_new_sequence}')


# Stub de demostración y verificación
if __name__ == "__main__":
    print("=" * 60)
    print("Prueba de TemporalDecayMemoryModule (TDMM)")
    print("=" * 60)

    # Configuración
    batch_size = 2
    memory_size = 8
    seq_len = 5

    # Crear instancia
    tdmm = TemporalDecayMemoryModule(
        memory_size=memory_size,
        decay_rate=0.1,
        learnable_decay=True,
        min_decay=0.001,
        max_decay=1.0,
        use_gating=True,
        reset_on_new_sequence=True
    )

    print(f"\nConfiguración:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Memory size: {memory_size}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Decay rate inicial: {tdmm.get_decay_rate():.4f}")

    # Crear tensor de entrada (secuencia)
    x = torch.randn(seq_len, batch_size, memory_size, requires_grad=True)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    output = tdmm(x, reset=True)
    print(f"\nOutput shape: {output.shape}")

    # Verificar decaimiento
    print(f"\nVerificación de decaimiento:")
    memory_state = tdmm.get_memory_state()
    print(f"  - Memory state shape: {memory_state.shape}")
    print(f"  - Memory state norm: {memory_state.norm().item():.4f}")
    print(f"  - Time step: {tdmm.time_step.item()}")

    # Comparar primer y último paso
    first_step = output[0]
    last_step = output[-1]
    print(f"  - First step norm: {first_step.norm().item():.4f}")
    print(f"  - Last step norm: {last_step.norm().item():.4f}")

    # Verificar gradientes
    loss = output.sum()
    loss.backward()

    print(f"\nVerificación de gradientes:")
    print(f"  - Input grad: {x.grad is not None}")
    print(f"  - Decay logit grad: {tdmm.decay_logit.grad is not None}")
    print(f"  - Output grad_fn: {output.grad_fn is not None}")

    if output.grad_fn is not None and x.grad is not None:
        print("  ✅ Gradientes funcionando correctamente")
    else:
        print("  ❌ Error: No se detectaron gradientes")

    # Prueba con entrada simple (no secuencia)
    print("\n" + "-" * 60)
    print("Prueba con entrada simple:")
    print("-" * 60)

    x_simple = torch.randn(batch_size, memory_size, requires_grad=True)
    output_simple = tdmm(x_simple, reset=True)
    print(f"Input shape: {x_simple.shape}")
    print(f"Output shape: {output_simple.shape}")

    print("\n" + "=" * 60)
    print("Prueba completada exitosamente")
    print("=" * 60)
