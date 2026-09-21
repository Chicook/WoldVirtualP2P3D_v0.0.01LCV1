"""
Transfer Learning - RF_RF_RFEN1_RN_1_2_6
Aprendizaje por transferencia 2020-2025
"""
import numpy as np


class TransferLearning:
    """Aplica transfer learning"""

    def __init__(self, input_size=100, learning_rate=0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.pre_trained_models = {}

    def forward(self, features):
        return np.maximum(0, np.dot(features, self.weights) + self.bias)

    def backward(self, error, features):
        x = np.dot(features, self.weights) + self.bias
        gradient = error * ((x > 0).astype(float))
        self.weights -= self.learning_rate * gradient * features
        self.bias -= self.learning_rate * gradient
        return gradient * features, gradient

    def fine_tune_pretrained(self, model, target_task):
        """Fine-tuning de modelo pre-entrenado"""
        # Freeze early layers, train last layers
        return {'frozen_layers': len(model) // 2, 'trainable_layers': len(model) // 2}

    def feature_extraction(self, model, data):
        """Extrae features de modelo pre-entrenado"""
        return model.predict(data, verbose=0)
