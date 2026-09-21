"""
Federated Learning - RF_RF_RFEN1_RN_1_2_9
Aprendizaje federado 2020-2025
"""
import numpy as np


class FederatedLearning:
    """Coordina aprendizaje entre múltiples clientes"""

    def __init__(self, input_size=100, learning_rate=0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.clients = []

    def forward(self, features):
        return 1 / (1 + np.exp(-np.clip(np.dot(features, self.weights) + self.bias, -250, 250)))

    def backward(self, error, features):
        x = np.dot(features, self.weights) + self.bias
        s = 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        gradient = error * s * (1 - s)
        self.weights -= self.learning_rate * gradient * features
        self.bias -= self.learning_rate * gradient
        return gradient * features, gradient

    def federated_averaging(self, client_updates):
        """FedAvg - Algoritmo principal (2020-2025)"""
        if not client_updates:
            return self.weights

        # Promedio ponderado de actualizaciones
        averaged = np.mean(client_updates, axis=0)
        return averaged

    def differential_privacy(self, sensitivity=1.0):
        """Añade privacidad diferencial"""
        noise = np.random.laplace(0, sensitivity)
        return noise
