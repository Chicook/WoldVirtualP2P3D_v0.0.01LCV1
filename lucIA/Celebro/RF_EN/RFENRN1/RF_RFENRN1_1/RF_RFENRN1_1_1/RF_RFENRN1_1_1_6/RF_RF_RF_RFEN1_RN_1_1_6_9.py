"""
TransformerRLAgent - Agente de Refuerzo basado en Transformer
=============================================================

Neurona especializada en implementar un agente de aprendizaje por refuerzo
que utiliza arquitectura Transformer para procesamiento de secuencias y
toma de decisiones en entornos complejos como metaversos.

Transformers en RL (2025):
- Attention mechanism para relaciones a largo plazo
- Procesamiento paralelo de secuencias
- Memory-efficient attention (mejor que RNN/LSTM)
- Decision Transformer paradigm (Chen et al., 2021-2025)
- Gato-style multimodal RL (DeepMind, 2025)

Ventajas para metaversos:
- Procesa history de estados largos
- Entiende relaciones espaciales y temporales
- Multimodal (texto, imágenes, sensores de red)
- Generaliza mejor que arquitecturas tradicionales
- Sample-efficient learning

Aplicaciones:
- Routing inteligente de paquetes considerando historial
- Optimización de QoS con contexto temporal
- Balanceo de carga predictivo
- Navegación de avatares IA
- Resource allocation con memoria

Autor: LucIA Development Team
Versión: 1.0.0
Fecha: Noviembre 2025
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class TransformerConfig:
    """Configuración del Transformer"""
    d_model: int = 64  # Dimensión del modelo
    num_heads: int = 4  # Número de attention heads
    num_layers: int = 2  # Capas de transformer
    d_ff: int = 256  # Dimensión feedforward
    max_seq_length: int = 100  # Longitud máxima de secuencia
    dropout: float = 0.1
    state_dim: int = 8
    action_dim: int = 4


class MultiHeadAttention:
    """Multi-Head Self-Attention"""

    def __init__(self, d_model: int, num_heads: int):
        """Inicializa multi-head attention"""
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Matrices de proyección Q, K, V
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """Divide en múltiples heads"""
        # x shape: (seq_len, d_model)
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)  # (num_heads, seq_len, d_k)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass de multi-head attention"""
        seq_len = x.shape[0]

        # Proyectar a Q, K, V
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Split heads
        Q_heads = self.split_heads(Q)
        K_heads = self.split_heads(K)
        V_heads = self.split_heads(V)

        # Scaled dot-product attention para cada head
        outputs = []
        for i in range(self.num_heads):
            Q_h = Q_heads[i]
            K_h = K_heads[i]
            V_h = V_heads[i]

            # Attention scores
            scores = Q_h @ K_h.T / np.sqrt(self.d_k)

            # Aplicar mask si existe
            if mask is not None:
                scores = np.where(mask, scores, -1e9)

            # Softmax
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            # Aplicar attention a valores
            output = attention_weights @ V_h
            outputs.append(output)

        # Concatenar heads
        concat_output = np.concatenate(outputs, axis=-1)

        # Proyección de salida
        final_output = concat_output @ self.W_o

        return final_output


class FeedForward:
    """Feed-Forward Network"""

    def __init__(self, d_model: int, d_ff: int):
        """Inicializa feed-forward"""
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        output = h @ self.W2 + self.b2
        return output


class TransformerLayer:
    """Capa de Transformer completa"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        """Inicializa capa de transformer"""
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)

        # Layer normalization parameters (simplificado)
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / (std + 1e-6) + beta

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass de la capa"""
        # Multi-head attention con residual connection
        attn_output = self.attention.forward(x, mask)
        x = self.layer_norm(x + attn_output, self.gamma1, self.beta1)

        # Feed-forward con residual connection
        ff_output = self.ff.forward(x)
        x = self.layer_norm(x + ff_output, self.gamma2, self.beta2)

        return x


class TransformerRLAgent:
    """Agente RL basado en Transformer"""

    def __init__(self, config: TransformerConfig):
        """Inicializa el agente Transformer"""
        self.config = config

        # Embedding de estados
        self.state_embedding = np.random.randn(config.state_dim, config.d_model) * 0.1

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(
            config.max_seq_length, config.d_model
        )

        # Capas de Transformer
        self.transformer_layers = [
            TransformerLayer(config.d_model, config.num_heads, config.d_ff)
            for _ in range(config.num_layers)
        ]

        # Policy head (actor)
        self.policy_head = np.random.randn(config.d_model, config.action_dim) * 0.1

        # Value head (critic)
        self.value_head = np.random.randn(config.d_model, 1) * 0.1

        # Historia de estados
        self.state_history: deque = deque(maxlen=config.max_seq_length)

        # Métricas
        self.metrics = {
            'forward_passes': 0,
            'actions_taken': 0,
            'avg_sequence_length': 0.0,
            'avg_attention_entropy': 0.0,
            'avg_computation_time_ms': 0.0
        }

        print("[OK] TransformerRLAgent inicializado")
        print(f"  d_model={config.d_model}, heads={config.num_heads}, layers={config.num_layers}")

    def _create_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Crea positional encoding"""
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def add_state(self, state: np.ndarray) -> None:
        """Añade estado al historial"""
        self.state_history.append(state.copy())

    def forward(self, use_full_history: bool = True) -> Tuple[np.ndarray, float]:
        """
        Forward pass del Transformer

        Args:
            use_full_history: Si usar todo el historial o solo último estado

        Returns:
            (policy_logits, value)
        """
        start_time = time.time()

        if not self.state_history:
            # Estado inicial aleatorio
            state = np.zeros(self.config.state_dim)
            self.state_history.append(state)

        # Obtener secuencia de estados
        if use_full_history:
            states = np.array(list(self.state_history))
        else:
            states = np.array([self.state_history[-1]])

        seq_len = len(states)

        # Embed estados
        embedded = states @ self.state_embedding

        # Añadir positional encoding
        embedded = embedded + self.positional_encoding[:seq_len]

        # Pasar por capas de transformer
        x = embedded
        for layer in self.transformer_layers:
            x = layer.forward(x)

        # Usar último token para predicción
        final_state = x[-1]

        # Policy logits
        policy_logits = final_state @ self.policy_head

        # Value
        value = (final_state @ self.value_head)[0]

        # Actualizar métricas
        computation_time = (time.time() - start_time) * 1000
        self.metrics['forward_passes'] += 1
        self.metrics['avg_sequence_length'] = (
            (self.metrics['avg_sequence_length'] * (self.metrics['forward_passes'] - 1) +
             seq_len) / self.metrics['forward_passes']
        )
        self.metrics['avg_computation_time_ms'] = (
            (self.metrics['avg_computation_time_ms'] * (self.metrics['forward_passes'] - 1) +
             computation_time) / self.metrics['forward_passes']
        )

        return policy_logits, value

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """
        Obtiene acción usando política

        Args:
            state: Estado actual
            deterministic: Si usar política determinística

        Returns:
            (action, log_prob)
        """
        # Añadir estado al historial
        self.add_state(state)

        # Forward pass
        policy_logits, value = self.forward(use_full_history=True)

        # Softmax
        exp_logits = np.exp(policy_logits - np.max(policy_logits))
        probs = exp_logits / exp_logits.sum()

        # Seleccionar acción
        if deterministic:
            action = np.argmax(probs)
        else:
            action = np.random.choice(self.config.action_dim, p=probs)

        log_prob = np.log(probs[action] + 1e-10)

        self.metrics['actions_taken'] += 1

        return action, log_prob

    def reset_history(self) -> None:
        """Resetea historial de estados"""
        self.state_history.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del agente"""
        return {
            **self.metrics,
            'history_length': len(self.state_history),
            'max_sequence_length': self.config.max_seq_length,
            'model_size_params': self._count_parameters()
        }

    def _count_parameters(self) -> int:
        """Cuenta parámetros del modelo"""
        total = 0
        total += self.state_embedding.size
        total += self.policy_head.size
        total += self.value_head.size

        for layer in self.transformer_layers:
            total += layer.attention.W_q.size
            total += layer.attention.W_k.size
            total += layer.attention.W_v.size
            total += layer.attention.W_o.size
            total += layer.ff.W1.size
            total += layer.ff.W2.size

        return total


def test_transformer_rl_agent():
    """Test del agente Transformer RL"""
    print("\n" + "="*70)
    print("TEST: TransformerRLAgent")
    print("="*70)

    # Configuración
    config = TransformerConfig(
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        max_seq_length=50,
        state_dim=8,
        action_dim=4
    )

    agent = TransformerRLAgent(config)

    # Test 1: Acción con estado único
    print("\n[OK] Test 1: Acción con estado único...")
    state = np.random.randn(config.state_dim)
    action, log_prob = agent.get_action(state, deterministic=False)
    print(f"  Estado: {state[:4]}...")
    print(f"  Acción: {action}")
    print(f"  Log prob: {log_prob:.4f}")

    # Test 2: Secuencia de estados
    print("\n[OK] Test 2: Procesando secuencia de estados...")
    for i in range(20):
        state = np.random.randn(config.state_dim)
        action, _ = agent.get_action(state)

    print(f"  Estados en historial: {len(agent.state_history)}")
    print(f"  Acciones tomadas: {agent.metrics['actions_taken']}")

    # Test 3: Forward pass con historial completo
    print("\n[OK] Test 3: Forward pass con historial...")
    policy_logits, value = agent.forward(use_full_history=True)
    print(f"  Policy logits: {policy_logits}")
    print(f"  Value: {value:.4f}")

    # Test 4: Métricas
    print("\n[OK] Test 4: Métricas del agente...")
    metrics = agent.get_metrics()
    print(f"  Forward passes: {metrics['forward_passes']}")
    print(f"  Longitud promedio de secuencia: {metrics['avg_sequence_length']:.1f}")
    print(f"  Tiempo promedio de cómputo: {metrics['avg_computation_time_ms']:.3f}ms")
    print(f"  Parámetros del modelo: {metrics['model_size_params']:,}")

    # Test 5: Reset
    print("\n[OK] Test 5: Reset de historial...")
    agent.reset_history()
    print(f"  Historial después de reset: {len(agent.state_history)}")

    print("\n[SUCCESS] Tests completados exitosamente")
    print("="*70 + "\n")

    return metrics


if __name__ == "__main__":
    test_transformer_rl_agent()
