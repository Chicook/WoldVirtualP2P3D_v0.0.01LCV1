"""
Redes Neuronales de Grafos - RF_RF_RFEN1_RN_1_2_4
Algoritmos GNN 2020-2025
"""
import numpy as np


class GraphNeuralNetworks:
    """Implementa GNN para datos estructurados"""

    def __init__(self, input_size=100, learning_rate=0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.graph_features = {}

    def forward(self, features):
        return 1 / (1 + np.exp(-np.clip(np.dot(features, self.weights) + self.bias, -250, 250)))

    def backward(self, error, features):
        x = np.dot(features, self.weights) + self.bias
        s = 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        gradient = error * s * (1 - s)
        self.weights -= self.learning_rate * gradient * features
        self.bias -= self.learning_rate * gradient
        return gradient * features, gradient

    def propagate_graph(self, adjacency_matrix, node_features):
        """Propaga mensajes en el grafo"""
        # Graph Convolution Network (2017-2025)
        # Graph Attention Network (2020)
        # Graph Transformer (2022-2025)
        propagated = np.dot(adjacency_matrix, node_features)
        return propagated

    def aggregate_features(self, neighbor_features):
        """Agrega features de vecinos"""
        return np.mean(neighbor_features, axis=0)
