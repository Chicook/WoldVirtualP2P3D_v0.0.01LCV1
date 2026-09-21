"""
Neurona Especializada 7: Capa de Aproximación de Neurona de Espiga (SNAL)
Spiking Neuron Approximation Layer

Esta neurona requiere la implementación de una aproximación de las dinámicas
temporales de las Redes Neuronales de Espiga (SNN) utilizando funciones
continuas que imitan el comportamiento de la espiga, pero son necesarias para
la entrenabilidad mediante métodos de descenso de gradiente.

La aproximación permite:
- Modelar dinámicas temporales de neuronas biológicas
- Mantener diferenciabilidad para backpropagation
- Simular comportamiento de espigas sin operaciones discretas

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


class SpikingNeuronApproximationLayer(nn.Module):
    """
    Capa de Aproximación de Neurona de Espiga (SNAL)

    Implementa una aproximación diferenciable de las dinámicas de neuronas
    de espiga, permitiendo entrenamiento con backpropagation estándar.

    Modelo: Integrate-and-Fire con aproximación suave

    Args:
        num_neurons: Número de neuronas en la capa
        threshold: Umbral de activación (default: 1.0)
        tau: Constante de tiempo de decaimiento (default: 10.0)
        reset_value: Valor de reset después de espiga (default: 0.0)
        temperature: Temperatura para aproximación suave (default: 1.0)
        learnable_threshold: Si True, el umbral es aprendible (default: False)
        use_batch_norm: Si True, aplica BatchNorm (default: True)

    Input Shape:
        (batch_size, num_neurons) o (seq_len, batch_size, num_neurons)

    Output Shape:
        Mismo que input shape
    """

    def __init__(
        self,
        num_neurons: int,
        threshold: float = 1.0,
        tau: float = 10.0,
        reset_value: float = 0.0,
        temperature: float = 1.0,
        learnable_threshold: bool = False,
        use_batch_norm: bool = True
    ) -> None:
        super(SpikingNeuronApproximationLayer, self).__init__()

        if num_neurons <= 0:
            raise ValueError("num_neurons debe ser positivo")

        if threshold <= 0:
            raise ValueError("threshold debe ser positivo")

        if tau <= 0:
            raise ValueError("tau debe ser positivo")

        if temperature <= 0:
            raise ValueError("temperature debe ser positivo")

        self.num_neurons = num_neurons
        self.tau = tau
        self.reset_value = reset_value
        self.temperature = temperature

        # Umbral aprendible o fijo
        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold))
        else:
            self.register_buffer('threshold', torch.tensor(threshold))

        # Batch normalization opcional
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_neurons)
        else:
            self.batch_norm = None

        # Estado de membrana (se inicializa en el primer forward)
        self.register_buffer('membrane_potential', None)
        self.register_buffer('spike_history', None)
        self.register_buffer('time_step', torch.tensor(0, dtype=torch.long))

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        Resetea el estado de las neuronas.

        Args:
            batch_size: Tamaño del batch
            device: Dispositivo del tensor
            dtype: Tipo de dato
        """
        self.membrane_potential = torch.zeros(batch_size, self.num_neurons, device=device, dtype=dtype)
        self.spike_history = torch.zeros(batch_size, self.num_neurons, device=device, dtype=dtype)
        self.time_step = torch.tensor(0, dtype=torch.long, device=device)

    def _smooth_spike_approximation(self, potential: torch.Tensor) -> torch.Tensor:
        """
        Aproximación suave de la función de espiga usando sigmoid.

        Args:
            potential: Potencial de membrana

        Returns:
            Aproximación suave de la espiga (0-1)
        """
        # Aproximación sigmoide: spike ≈ sigmoid((V - threshold) / temperature)
        normalized = (potential - self.threshold) / self.temperature
        spike_approx = torch.sigmoid(normalized)
        return spike_approx

    def _leaky_integrate(self, input_current: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Integración con fuga (leaky integrate).

        Args:
            input_current: Corriente de entrada
            dt: Paso de tiempo

        Returns:
            Potencial de membrana actualizado
        """
        # dV/dt = -V/tau + I
        # Solución: V(t+dt) = V(t) * exp(-dt/tau) + I * tau * (1 - exp(-dt/tau))
        decay_factor = torch.exp(-dt / self.tau)
        integration_factor = self.tau * (1 - decay_factor)

        new_potential = self.membrane_potential * decay_factor + input_current * integration_factor
        return new_potential

    def forward(
        self,
        x: torch.Tensor,
        reset: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Forward pass de la capa de aproximación de espiga.

        Args:
            x: Tensor de entrada (corriente de entrada)
            reset: Si True, resetea el estado (default: None)

        Returns:
            Tensor de espigas aproximadas
        """
        is_sequence = x.dim() == 3
        batch_size = x.size(0) if not is_sequence else x.size(1)
        device = x.device
        dtype = x.dtype

        # Inicializar o resetear estado si es necesario
        if self.membrane_potential is None or reset:
            self.reset_state(batch_size, device, dtype)

        # Aplicar batch normalization si está habilitado
        if self.batch_norm is not None:
            if is_sequence:
                # Para secuencias, aplicar sobre cada paso de tiempo
                original_shape = x.shape
                x = x.view(-1, original_shape[-1])
                x = self.batch_norm(x)
                x = x.view(original_shape)
            else:
                x = self.batch_norm(x)

        if is_sequence:
            # Procesar secuencia paso a paso
            seq_len = x.size(0)
            outputs = []

            for t in range(seq_len):
                self.time_step += 1

                # Integrar entrada actual
                self.membrane_potential = self._leaky_integrate(x[t])

                # Calcular espigas aproximadas
                spikes = self._smooth_spike_approximation(self.membrane_potential)

                # Reset después de espiga (aproximado)
                reset_mask = spikes > 0.5  # Umbral para considerar espiga
                self.membrane_potential = torch.where(
                    reset_mask,
                    torch.full_like(self.membrane_potential, self.reset_value),
                    self.membrane_potential
                )

                # Actualizar historial de espigas
                self.spike_history = spikes
                outputs.append(spikes)

            # Apilar outputs
            output = torch.stack(outputs, dim=0)
        else:
            # Procesar un solo paso
            self.time_step += 1

            # Integrar entrada actual
            self.membrane_potential = self._leaky_integrate(x)

            # Calcular espigas aproximadas
            spikes = self._smooth_spike_approximation(self.membrane_potential)

            # Reset después de espiga (aproximado)
            reset_mask = spikes > 0.5
            self.membrane_potential = torch.where(
                reset_mask,
                torch.full_like(self.membrane_potential, self.reset_value),
                self.membrane_potential
            )

            # Actualizar historial de espigas
            self.spike_history = spikes
            output = spikes

        return output

    def get_membrane_potential(self) -> Optional[torch.Tensor]:
        """
        Obtiene el potencial de membrana actual.

        Returns:
            Tensor con el potencial de membrana o None
        """
        return self.membrane_potential

    def get_spike_history(self) -> Optional[torch.Tensor]:
        """
        Obtiene el historial de espigas.

        Returns:
            Tensor con el historial de espigas o None
        """
        return self.spike_history

    def extra_repr(self) -> str:
        """Representación adicional para debugging"""
        threshold_val = self.threshold.item() if isinstance(self.threshold, torch.Tensor) else self.threshold
        return (f'num_neurons={self.num_neurons}, threshold={threshold_val:.3f}, '
                f'tau={self.tau:.3f}, temperature={self.temperature:.3f}')


# Stub de demostración y verificación
if __name__ == "__main__":
    print("=" * 60)
    print("Prueba de SpikingNeuronApproximationLayer (SNAL)")
    print("=" * 60)

    # Configuración
    batch_size = 2
    num_neurons = 8
    seq_len = 10

    # Crear instancia
    snn = SpikingNeuronApproximationLayer(
        num_neurons=num_neurons,
        threshold=1.0,
        tau=10.0,
        reset_value=0.0,
        temperature=1.0,
        learnable_threshold=False,
        use_batch_norm=True
    )

    print(f"\nConfiguración:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Neurons: {num_neurons}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Threshold: {snn.threshold.item():.3f}")
    print(f"  - Tau: {snn.tau:.3f}")

    # Crear tensor de entrada (secuencia de corrientes)
    x = torch.randn(seq_len, batch_size, num_neurons, requires_grad=True)
    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min().item():.3f}, {x.max().item():.3f}]")

    # Forward pass
    output = snn(x, reset=True)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print(f"Spike rate: {(output > 0.5).float().mean().item():.4f}")

    # Verificar estado
    membrane = snn.get_membrane_potential()
    spikes = snn.get_spike_history()
    print(f"\nEstado de neuronas:")
    print(f"  - Membrane potential shape: {membrane.shape}")
    print(f"  - Membrane potential range: [{membrane.min().item():.3f}, {membrane.max().item():.3f}]")
    print(f"  - Spike history shape: {spikes.shape}")
    print(f"  - Time step: {snn.time_step.item()}")

    # Verificar gradientes
    loss = output.sum()
    loss.backward()

    print(f"\nVerificación de gradientes:")
    print(f"  - Input grad: {x.grad is not None}")
    print(f"  - Output grad_fn: {output.grad_fn is not None}")

    if output.grad_fn is not None and x.grad is not None:
        print("  ✅ Gradientes funcionando correctamente")
        print(f"  - Input grad norm: {x.grad.norm().item():.6f}")
    else:
        print("  ❌ Error: No se detectaron gradientes")

    # Prueba con entrada simple
    print("\n" + "-" * 60)
    print("Prueba con entrada simple:")
    print("-" * 60)

    x_simple = torch.randn(batch_size, num_neurons, requires_grad=True)
    output_simple = snn(x_simple, reset=True)
    print(f"Input shape: {x_simple.shape}")
    print(f"Output shape: {output_simple.shape}")
    print(f"Spike rate: {(output_simple > 0.5).float().mean().item():.4f}")

    print("\n" + "=" * 60)
    print("Prueba completada exitosamente")
    print("=" * 60)
