"""
Continual Learning - RF_RF_RFEN1_RN_1_2_10
Aprendizaje continuo sin catástrofe 2020-2025
"""
import numpy as np


class ContinualLearning:
    """Permite aprendizaje continuo sin olvidar"""

    def __init__(self, input_size=100, learning_rate=0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.memory_buffer = []

    def forward(self, features):
        return np.maximum(0, np.dot(features, self.weights) + self.bias)

    def backward(self, error, features):
        x = np.dot(features, self.weights) + self.bias
        gradient = error * ((x > 0).astype(float))
        self.weights -= self.learning_rate * gradient * features
        self.bias -= self.learning_rate * gradient
        return gradient * features, gradient

    def experience_replay(self, samples, buffer_size=1000):
        """Almacena y usa experiencias pasadas"""
        # Para prevenir catastrophic forgetting
        self.memory_buffer.extend(samples)
        if len(self.memory_buffer) > buffer_size:
            self.memory_buffer = self.memory_buffer[-buffer_size:]

        return np.array(self.memory_buffer)

    def elastic_weight_consolidation(self, fisher_info, lambda_reg=1000):
        """EWC - Evita olvidar tareas anteriores"""
        # Fisher Information Matrix
        # Penaliza cambios en parámetros importantes
        return {'ewc_loss': lambda_reg * fisher_info}

    def gradient_episodic_memory(self, current_grad, memory_grads):
        """GEM - Evita interferencias negativas"""
        # Proyecta gradiente para no empeorar tareas pasadas
        for mem_grad in memory_grads:
            if np.dot(current_grad, mem_grad) < 0:
                # Proyectar
                current_grad -= np.dot(current_grad, mem_grad) / np.dot(mem_grad, mem_grad) * mem_grad
        return current_grad
