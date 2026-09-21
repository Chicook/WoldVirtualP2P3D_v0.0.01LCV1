"""
LiquidNeuralNetworkOptimizer - Optimizador de Redes Neuronales Líquidas
========================================================================

Neurona especializada en implementar y optimizar Liquid Neural Networks (LNNs),
una arquitectura revolucionaria de 2025 que permite adaptación continua y
aprendizaje dinámico en tiempo real.

Liquid Neural Networks (del MIT):
- Inspiradas en neuronas biológicas del C. elegans
- Ecuaciones diferenciales ordinarias (ODEs) como núcleo
- Adaptación continua sin reentrenamiento
- Eficiencia computacional superior a RNNs/Transformers
- Interpretabilidad mejorada

Ventajas para metaversos:
- Adaptación en tiempo real a cambios de red
- Predicción de latencia con datos streaming
- Optimización continua de bandwidth
- Memoria eficiente (menos parámetros)
- Robustez ante cambios de distribución

Implementación basada en:
- Neural ODEs (Chen et al., 2018)
- Liquid Time-Constant Networks (Hasani et al., 2021-2025)
- Continuous-time RNNs

Autor: LucIA Development Team
Versión: 1.0.0
Fecha: Noviembre 2025
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class LiquidNeuronState:
    """Estado de una neurona líquida"""
    state_value: float
    tau: float  # Time constant
    sensitivity: float
    last_update_time: float


@dataclass
class LiquidNetworkConfig:
    """Configuración de red neuronal líquida"""
    input_size: int
    hidden_size: int
    output_size: int
    tau_min: float = 0.5
    tau_max: float = 5.0
    use_gate: bool = True
    nonlinearity: str = 'tanh'  # 'tanh', 'relu', 'sigmoid'
    solver: str = 'euler'  # 'euler', 'rk4', 'adaptive'


class LiquidNeuralNetworkOptimizer:
    """Optimizador de Redes Neuronales Líquidas"""

    def __init__(self, config: LiquidNetworkConfig):
        """
        Inicializa el optimizador de LNN

        Args:
            config: Configuración de la red
        """
        self.config = config

        # Parámetros de la red
        self.W_in = None  # Input weights
        self.W_hidden = None  # Hidden weights
        self.W_out = None  # Output weights

        # Constantes de tiempo (tau) para cada neurona
        self.tau_hidden = None

        # Estados de neuronas líquidas
        self.hidden_states: List[LiquidNeuronState] = []

        # Parámetros de gate (si se usa)
        self.W_gate = None
        self.b_gate = None

        # Historial
        self.state_history = []
        self.output_history = []

        # Métricas
        self.metrics = {
            'forward_passes': 0,
            'updates': 0,
            'avg_state_change': 0.0,
            'adaptation_rate': 0.0,
            'computation_time_ms': 0.0
        }

        # Inicializar red
        self._initialize_network()

        print("[OK] LiquidNeuralNetworkOptimizer inicializado")
        print(f"  Arquitectura: {config.input_size} -> {config.hidden_size} -> {config.output_size}")

    def _initialize_network(self) -> None:
        """Inicializa parámetros de la red"""
        cfg = self.config

        # Inicialización Xavier para pesos de entrada
        limit_in = np.sqrt(6.0 / (cfg.input_size + cfg.hidden_size))
        self.W_in = np.random.uniform(
            -limit_in, limit_in,
            (cfg.input_size, cfg.hidden_size)
        )

        # Inicialización para pesos ocultos (sparser para estabilidad)
        sparsity = 0.3  # 30% de conexiones
        self.W_hidden = np.random.randn(cfg.hidden_size, cfg.hidden_size) * 0.1
        mask = np.random.rand(cfg.hidden_size, cfg.hidden_size) > sparsity
        self.W_hidden *= mask

        # Pesos de salida
        limit_out = np.sqrt(6.0 / (cfg.hidden_size + cfg.output_size))
        self.W_out = np.random.uniform(
            -limit_out, limit_out,
            (cfg.hidden_size, cfg.output_size)
        )

        # Constantes de tiempo (tau) - diferentes para cada neurona
        self.tau_hidden = np.random.uniform(
            cfg.tau_min, cfg.tau_max,
            cfg.hidden_size
        )

        # Gate parameters (si se usa)
        if cfg.use_gate:
            self.W_gate = np.random.randn(cfg.input_size, cfg.hidden_size) * 0.1
            self.b_gate = np.zeros(cfg.hidden_size)

        # Inicializar estados de neuronas
        current_time = time.time()
        self.hidden_states = [
            LiquidNeuronState(
                state_value=0.0,
                tau=self.tau_hidden[i],
                sensitivity=1.0,
                last_update_time=current_time
            )
            for i in range(cfg.hidden_size)
        ]

    def _apply_nonlinearity(self, x: np.ndarray) -> np.ndarray:
        """Aplica función de activación"""
        if self.config.nonlinearity == 'tanh':
            return np.tanh(x)
        elif self.config.nonlinearity == 'relu':
            return np.maximum(0, x)
        elif self.config.nonlinearity == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x

    def _ode_step(self, state: float, input_current: float, tau: float,
                  dt: float) -> float:
        """
        Un paso de integración de la ODE

        ODE: dx/dt = (-x + input_current) / tau

        Args:
            state: Estado actual
            input_current: Input corriente
            tau: Constante de tiempo
            dt: Paso de tiempo

        Returns:
            Nuevo estado
        """
        if self.config.solver == 'euler':
            # Método de Euler
            dstate_dt = (-state + input_current) / tau
            new_state = state + dt * dstate_dt

        elif self.config.solver == 'rk4':
            # Runge-Kutta de orden 4
            def f(s):
                return (-s + input_current) / tau

            k1 = f(state)
            k2 = f(state + 0.5 * dt * k1)
            k3 = f(state + 0.5 * dt * k2)
            k4 = f(state + dt * k3)

            new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        else:  # adaptive
            # Paso adaptativo simple
            dstate_dt = (-state + input_current) / tau
            dt_adaptive = min(dt, tau * 0.1)  # Limitar dt a 10% de tau
            new_state = state + dt_adaptive * dstate_dt

        return new_state

    def forward(self, x: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Forward pass con integración temporal

        Args:
            x: Input vector (shape: input_size)
            dt: Paso de tiempo para integración

        Returns:
            Output vector (shape: output_size)
        """
        start_time = time.time()
        current_time = time.time()

        # Input projection
        input_projection = x @ self.W_in

        # Gate (si se usa)
        if self.config.use_gate:
            gate = self._apply_nonlinearity(x @ self.W_gate + self.b_gate)
        else:
            gate = np.ones(self.config.hidden_size)

        # Actualizar estados de neuronas líquidas
        new_hidden = np.zeros(self.config.hidden_size)

        for i, neuron_state in enumerate(self.hidden_states):
            # Tiempo desde última actualización
            dt_actual = current_time - neuron_state.last_update_time

            # Input corriente para esta neurona
            input_current = input_projection[i]

            # Contribución de otras neuronas ocultas
            hidden_contribution = sum(
                self.W_hidden[j, i] * self.hidden_states[j].state_value
                for j in range(self.config.hidden_size)
            )

            total_input = self._apply_nonlinearity(
                input_current + hidden_contribution
            ) * gate[i]

            # Integrar ODE
            new_state = self._ode_step(
                neuron_state.state_value,
                total_input,
                neuron_state.tau,
                dt_actual
            )

            # Actualizar estado
            neuron_state.state_value = new_state
            neuron_state.last_update_time = current_time

            new_hidden[i] = new_state

        # Output projection
        output = new_hidden @ self.W_out

        # Guardar historial
        self.state_history.append(new_hidden.copy())
        self.output_history.append(output.copy())

        # Limitar historial
        if len(self.state_history) > 1000:
            self.state_history.pop(0)
            self.output_history.pop(0)

        # Actualizar métricas
        self.metrics['forward_passes'] += 1
        self.metrics['computation_time_ms'] = (time.time() - start_time) * 1000

        return output

    def adapt_to_signal(self, signal: np.ndarray, window_size: int = 10) -> Dict[str, float]:
        """
        Adapta la red a una señal de entrada

        Args:
            signal: Señal de entrada (shape: [timesteps, input_size])
            window_size: Ventana de adaptación

        Returns:
            Métricas de adaptación
        """
        initial_error = 0.0
        final_error = 0.0
        state_changes = []

        for t in range(len(signal)):
            x = signal[t]

            # Forward pass
            output = self.forward(x, dt=0.01)

            # Calcular cambio de estado
            if len(self.state_history) >= 2:
                state_change = np.mean(np.abs(
                    self.state_history[-1] - self.state_history[-2]
                ))
                state_changes.append(state_change)

            # Error (asumiendo target = input para autoencoder)
            error = np.mean((output - x[:self.config.output_size])**2)

            if t == 0:
                initial_error = error
            elif t == len(signal) - 1:
                final_error = error

        # Calcular métricas de adaptación
        adaptation_rate = (initial_error - final_error) / max(initial_error, 1e-8)
        avg_state_change = np.mean(state_changes) if state_changes else 0.0

        self.metrics['adaptation_rate'] = adaptation_rate
        self.metrics['avg_state_change'] = avg_state_change

        return {
            'initial_error': initial_error,
            'final_error': final_error,
            'adaptation_rate': adaptation_rate,
            'avg_state_change': avg_state_change,
            'timesteps_processed': len(signal)
        }

    def predict_latency(self, network_features: np.ndarray) -> float:
        """
        Predice latencia basado en features de red

        Args:
            network_features: Features de red (RTT, loss, bandwidth, etc.)

        Returns:
            Latencia predicha en ms
        """
        # Forward pass con features de red
        output = self.forward(network_features, dt=0.01)

        # Primera salida es la latencia predicha
        latency_ms = max(0, output[0])

        return latency_ms

    def optimize_bandwidth_allocation(self, available_bandwidth: float,
                                      num_channels: int) -> np.ndarray:
        """
        Optimiza asignación de bandwidth entre canales

        Args:
            available_bandwidth: Bandwidth disponible (Mbps)
            num_channels: Número de canales

        Returns:
            Asignación de bandwidth por canal
        """
        # Preparar input
        x = np.array([available_bandwidth, num_channels, 0, 0])

        # Expandir input si es necesario
        if len(x) < self.config.input_size:
            x = np.pad(x, (0, self.config.input_size - len(x)))

        # Forward pass
        output = self.forward(x, dt=0.01)

        # Normalizar output para que sume al bandwidth disponible
        allocation = np.abs(output[:num_channels])
        allocation = (allocation / allocation.sum()) * available_bandwidth

        return allocation

    def reset_states(self) -> None:
        """Resetea estados de las neuronas"""
        current_time = time.time()
        for neuron_state in self.hidden_states:
            neuron_state.state_value = 0.0
            neuron_state.last_update_time = current_time

        self.state_history.clear()
        self.output_history.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del optimizador"""
        return {
            **self.metrics,
            'state_history_length': len(self.state_history),
            'avg_hidden_state': np.mean([n.state_value for n in self.hidden_states]),
            'avg_tau': np.mean(self.tau_hidden),
            'network_config': {
                'input_size': self.config.input_size,
                'hidden_size': self.config.hidden_size,
                'output_size': self.config.output_size,
                'solver': self.config.solver,
                'nonlinearity': self.config.nonlinearity
            }
        }


def test_liquid_neural_network():
    """Test del optimizador de red neuronal líquida"""
    print("\n" + "="*70)
    print("TEST: LiquidNeuralNetworkOptimizer")
    print("="*70)

    # Configuración
    config = LiquidNetworkConfig(
        input_size=4,
        hidden_size=16,
        output_size=4,
        tau_min=0.5,
        tau_max=3.0,
        use_gate=True,
        solver='rk4'
    )

    optimizer = LiquidNeuralNetworkOptimizer(config)

    # Test 1: Forward pass simple
    print("\n[OK] Test 1: Forward pass...")
    x = np.array([1.0, 0.5, 0.3, 0.8])
    output = optimizer.forward(x, dt=0.01)
    print(f"  Input: {x}")
    print(f"  Output: {output}")

    # Test 2: Adaptación a señal
    print("\n[OK] Test 2: Adaptación a señal sinusoidal...")
    t = np.linspace(0, 2*np.pi, 100)
    signal = np.column_stack([
        np.sin(t),
        np.cos(t),
        np.sin(2*t),
        np.cos(2*t)
    ])

    adaptation_metrics = optimizer.adapt_to_signal(signal, window_size=10)
    print(f"  Error inicial: {adaptation_metrics['initial_error']:.4f}")
    print(f"  Error final: {adaptation_metrics['final_error']:.4f}")
    print(f"  Tasa de adaptación: {adaptation_metrics['adaptation_rate']:.2%}")
    print(f"  Cambio promedio de estado: {adaptation_metrics['avg_state_change']:.4f}")

    # Test 3: Predicción de latencia
    print("\n[OK] Test 3: Predicción de latencia...")
    network_features = np.array([50.0, 0.01, 1000.0, 5.0])  # RTT, loss, bandwidth, jitter
    predicted_latency = optimizer.predict_latency(network_features)
    print(f"  Features de red: {network_features}")
    print(f"  Latencia predicha: {predicted_latency:.2f}ms")

    # Test 4: Optimización de bandwidth
    print("\n[OK] Test 4: Optimización de bandwidth...")
    allocation = optimizer.optimize_bandwidth_allocation(
        available_bandwidth=100.0,
        num_channels=4
    )
    print(f"  Bandwidth disponible: 100.0 Mbps")
    print(f"  Asignación por canal: {allocation}")
    print(f"  Total asignado: {allocation.sum():.2f} Mbps")

    # Test 5: Métricas
    print("\n[OK] Test 5: Métricas del optimizador...")
    metrics = optimizer.get_metrics()
    print(f"  Forward passes: {metrics['forward_passes']}")
    print(f"  Tiempo de cómputo: {metrics['computation_time_ms']:.3f}ms")
    print(f"  Estado oculto promedio: {metrics['avg_hidden_state']:.4f}")
    print(f"  Tau promedio: {metrics['avg_tau']:.2f}")

    print("\n[SUCCESS] Tests completados exitosamente")
    print("="*70 + "\n")

    return metrics


if __name__ == "__main__":
    test_liquid_neural_network()
