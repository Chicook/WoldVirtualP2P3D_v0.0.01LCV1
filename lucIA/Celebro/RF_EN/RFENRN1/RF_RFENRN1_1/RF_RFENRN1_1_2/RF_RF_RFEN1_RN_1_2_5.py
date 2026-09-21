"""
Mecanismos de Atención - RF_RF_RFEN1_RN_1_2_5
Attention 2020-2025
"""
import numpy as np


class AttentionMechanisms:
    """Implementa mecanismos de atención avanzados"""

    def __init__(self, input_size=100, learning_rate=0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.attention_weights = []

    def forward(self, features):
        return np.clip(np.dot(features, self.weights) + self.bias, -1, 1)

    def backward(self, error, features):
        gradient = error
        self.weights -= self.learning_rate * gradient * features
        self.bias -= self.learning_rate * gradient
        return gradient * features, gradient

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """Multi-Head Attention (2020-2025)"""
        d_k = query.shape[-1]
        scores = np.dot(query, key.T) / np.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask, scores, float('-inf'))
        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
        return np.dot(attention_weights, value)

    def self_attention(self, x):
        """Self-attention mechanism"""
        # Create Q, K, V from input
        q = k = v = x  # Simplificado
        return self.scaled_dot_product_attention(q, k, v)
