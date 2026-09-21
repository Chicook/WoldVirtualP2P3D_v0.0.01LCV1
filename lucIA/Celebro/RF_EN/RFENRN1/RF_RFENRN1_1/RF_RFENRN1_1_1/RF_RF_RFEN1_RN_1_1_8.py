"""
Protocolo de Red - RF_RF_RFEN1_RN_1_1_8
"""
import numpy as np


class NetworkProtocol:
    """Protocolo de red para metaverso"""

    def __init__(self, input_size=100, learning_rate=0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)

    def forward(self, features):
        return np.clip(np.dot(features, self.weights) + self.bias, 0, 1)

    def backward(self, error, features):
        gradient = error
        self.weights -= self.learning_rate * gradient * features
        self.bias -= self.learning_rate * gradient
        return gradient * features, gradient

    def send_packet(self, data):
        return {'sent': True, 'size': len(str(data))}
