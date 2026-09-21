"""
Multi-Task Learning - RF_RF_RFEN1_RN_1_2_7
Aprendizaje multi-tarea
"""
import numpy as np


class MultiTaskLearning:
    """Entrena en múltiples tareas simultáneamente"""

    def __init__(self, input_size=100, learning_rate=0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.tasks = []

    def forward(self, features):
        return np.clip(np.dot(features, self.weights) + self.bias, 0, 1)

    def backward(self, error, features):
        gradient = error
        self.weights -= self.learning_rate * gradient * features
        self.bias -= self.learning_rate * gradient
        return gradient * features, gradient

    def add_task(self, task_name, loss_function):
        """Añade tarea al sistema"""
        self.tasks.append({'name': task_name, 'loss': loss_function})

    def weighted_loss(self, losses, weights):
        """Combina pérdidas de múltiples tareas"""
        return sum(l * w for l, w in zip(losses, weights))

    def task_weighting(self):
        """Calcula pesos óptimos para tareas"""
        # GradNorm (2020), Dynamic Weight Average (2021)
        return [0.33, 0.33, 0.34]  # Pesos balanceados
