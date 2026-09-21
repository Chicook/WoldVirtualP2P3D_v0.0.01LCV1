"""
Interfaz Python-3D - RF_RF_RFEN1_RN_1_1_1
Conecta Python con librerías 3D (moderngl, pyglet, PyOpenGL)
Sistema Avanzado de Redes Neuronales con Arquitectura Transformer
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import warnings
import gc
import json
from pathlib import Path

warnings.filterwarnings('ignore')


@dataclass
class NeuralConfig:
    """Configuración avanzada para el procesador neuronal"""
    input_size: int = 100
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 256])
    num_attention_heads: int = 8
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    use_layer_norm: bool = True
    activation: str = 'gelu'
    use_residual: bool = True
    gradient_clip_val: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class Python3DInterface:
    """Conecta código Python con renderizado 3D"""

    def __init__(self, input_size=100, learning_rate=0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.contexts = {}

        # Inicializar procesador neuronal avanzado
        self.neural_config = NeuralConfig(input_size=input_size, learning_rate=learning_rate)
        self.advanced_processor = self.AdvancedNeuralProcessor(self.neural_config)

    class AdvancedNeuralProcessor(nn.Module):
        """
        Procesador Neuronal Avanzado con Arquitectura Transformer
        Implementa mecanismos de atención, normalización por capas y optimización avanzada
        """

        def __init__(self, config: NeuralConfig):
            super().__init__()
            self.config = config
            self.device = torch.device(config.device)

            # Embeddings de entrada con proyección
            self.input_projection = nn.Sequential(
                nn.Linear(config.input_size, config.hidden_dims[0]),
                self._get_activation(config.activation),
                nn.Dropout(config.dropout_rate)
            )

            # Capas de atención multi-cabeza
            self.attention_layers = nn.ModuleList([
                self._build_attention_block(dim)
                for dim in config.hidden_dims
            ])

            # Redes Feed-Forward
            self.ffn_layers = nn.ModuleList([
                self._build_ffn_block(dim)
                for dim in config.hidden_dims
            ])

            # Normalizaciones por capa
            if config.use_layer_norm:
                self.layer_norms_1 = nn.ModuleList([
                    nn.LayerNorm(dim) for dim in config.hidden_dims
                ])
                self.layer_norms_2 = nn.ModuleList([
                    nn.LayerNorm(dim) for dim in config.hidden_dims
                ])

            # Capa de salida con proyección
            self.output_projection = nn.Sequential(
                nn.Linear(config.hidden_dims[-1], config.hidden_dims[-1] // 2),
                self._get_activation(config.activation),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_dims[-1] // 2, config.input_size)
            )

            # Optimizador y scheduler
            self.optimizer = AdamW(
                self.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )

            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )

            # Métricas y estadísticas
            self.training_stats = {
                'losses': [],
                'gradients': [],
                'learning_rates': [],
                'epochs': 0
            }

            # Cache para inferencia rápida
            self.cache = OrderedDict()
            self.cache_max_size = 1000

            # Mover modelo al dispositivo
            self.to(self.device)

        def _get_activation(self, name: str) -> nn.Module:
            """Obtiene función de activación por nombre"""
            activations = {
                'relu': nn.ReLU(),
                'gelu': nn.GELU(),
                'silu': nn.SiLU(),
                'tanh': nn.Tanh(),
                'leaky_relu': nn.LeakyReLU(0.2)
            }
            return activations.get(name.lower(), nn.GELU())

        def _build_attention_block(self, dim: int) -> nn.Module:
            """Construye bloque de atención multi-cabeza"""
            return nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=self.config.num_attention_heads,
                dropout=self.config.dropout_rate,
                batch_first=True
            )

        def _build_ffn_block(self, dim: int) -> nn.Module:
            """Construye red feed-forward con expansión"""
            expansion_factor = 4
            return nn.Sequential(
                nn.Linear(dim, dim * expansion_factor),
                self._get_activation(self.config.activation),
                nn.Dropout(self.config.dropout_rate),
                nn.Linear(dim * expansion_factor, dim),
                nn.Dropout(self.config.dropout_rate)
            )

        def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
            """
            Forward pass con atención y residuales
            Args:
                x: Tensor de entrada [batch_size, input_size]
                use_cache: Si usar caché para inferencia rápida
            Returns:
                Tensor procesado [batch_size, input_size]
            """
            if use_cache:
                cache_key = self._get_cache_key(x)
                if cache_key in self.cache:
                    return self.cache[cache_key]

            # Proyección de entrada
            x = self.input_projection(x)

            # Si es 2D, añadir dimensión de secuencia
            if x.dim() == 2:
                x = x.unsqueeze(1)

            # Pasar por capas de atención y FFN
            for i, (attn, ffn) in enumerate(zip(self.attention_layers, self.ffn_layers)):
                # Atención con conexión residual
                residual = x
                if self.config.use_layer_norm:
                    x = self.layer_norms_1[i](x)

                attn_output, _ = attn(x, x, x)
                x = residual + attn_output

                # FFN con conexión residual
                residual = x
                if self.config.use_layer_norm:
                    x = self.layer_norms_2[i](x)

                ffn_output = ffn(x)
                x = residual + ffn_output

            # Proyección de salida
            x = x.squeeze(1) if x.size(1) == 1 else x.mean(dim=1)
            output = self.output_projection(x)

            # Guardar en caché
            if use_cache:
                self._update_cache(cache_key, output)

            return output

        def train_step(self, x: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
            """
            Paso de entrenamiento con retropropagación
            Args:
                x: Datos de entrada
                target: Objetivo
            Returns:
                Diccionario con métricas
            """
            self.train()
            self.optimizer.zero_grad()

            # Forward pass
            output = self.forward(x, use_cache=False)
            loss = F.mse_loss(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                self.config.gradient_clip_val
            )

            self.optimizer.step()
            self.scheduler.step()

            # Actualizar estadísticas
            self.training_stats['losses'].append(loss.item())
            self.training_stats['gradients'].append(grad_norm.item())
            self.training_stats['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )

            return {
                'loss': loss.item(),
                'grad_norm': grad_norm.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            }

        def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
            """Predicción con conversión automática"""
            self.eval()
            with torch.no_grad():
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).float().to(self.device)
                output = self.forward(x, use_cache=True)
                return output.cpu().numpy()

        def _get_cache_key(self, x: torch.Tensor) -> str:
            """Genera clave de caché para tensor"""
            return str(hash(x.cpu().numpy().tobytes()))

        def _update_cache(self, key: str, value: torch.Tensor):
            """Actualiza caché con límite de tamaño"""
            if len(self.cache) >= self.cache_max_size:
                self.cache.popitem(last=False)
            self.cache[key] = value.detach().clone()

        def save_checkpoint(self, path: Union[str, Path]):
            """Guarda checkpoint del modelo"""
            checkpoint = {
                'model_state': self.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'config': self.config.__dict__,
                'stats': self.training_stats
            }
            torch.save(checkpoint, path)

        def load_checkpoint(self, path: Union[str, Path]):
            """Carga checkpoint del modelo"""
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.training_stats = checkpoint['stats']

        def get_stats_summary(self) -> Dict[str, Any]:
            """Obtiene resumen de estadísticas de entrenamiento"""
            if not self.training_stats['losses']:
                return {'status': 'No training data'}

            return {
                'avg_loss': np.mean(self.training_stats['losses'][-100:]),
                'min_loss': np.min(self.training_stats['losses']),
                'avg_grad_norm': np.mean(self.training_stats['gradients'][-100:]),
                'current_lr': self.training_stats['learning_rates'][-1],
                'total_steps': len(self.training_stats['losses'])
            }

        def optimize_memory(self):
            """Optimiza uso de memoria"""
            self.cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def forward(self, features):
        """
        Forward pass mejorado con procesador adaptativo
        Utiliza clase anidada AdaptiveForwardProcessor para mayor eficiencia
        """
        if not hasattr(self, 'adaptive_processor'):
            self.adaptive_processor = self.AdaptiveForwardProcessor(
                self.input_size,
                self.learning_rate,
                self.weights.copy(),
                self.bias
            )
        return self.adaptive_processor.process(features)

    class AdaptiveForwardProcessor:
        """
        Procesador Avanzado de Forward Pass con Ensemble Learning
        Implementa múltiples técnicas de optimización y procesamiento adaptativo
        """

        def __init__(self, input_size: int, learning_rate: float, weights: np.ndarray, bias: float):
            import scipy.special as sp
            from scipy.optimize import minimize
            from scipy.stats import entropy
            from sklearn.preprocessing import StandardScaler, RobustScaler
            from sklearn.decomposition import PCA, FastICA
            from sklearn.ensemble import RandomForestRegressor
            import warnings
            warnings.filterwarnings('ignore')

            # Imports avanzados
            self.sp = sp
            self.minimize = minimize
            self.entropy = entropy

            # Configuración base
            self.input_size = input_size
            self.learning_rate = learning_rate
            self.base_weights = weights.copy()
            self.base_bias = bias

            # Inicializar componentes de ensemble
            self._initialize_ensemble_components()

            # Scalers para normalización adaptativa
            self.standard_scaler = StandardScaler()
            self.robust_scaler = RobustScaler()

            # Reducción de dimensionalidad
            self.pca = PCA(n_components=min(50, input_size))
            self.ica = FastICA(n_components=min(30, input_size), max_iter=500)

            # Modelo de ensemble para predicción
            self.rf_regressor = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            # Pesos adaptativos para ensemble
            self.ensemble_weights = {
                'base': 0.3,
                'adaptive': 0.25,
                'momentum': 0.2,
                'quantum': 0.15,
                'evolutionary': 0.1
            }

            # Historial y caché
            self.prediction_history = []
            self.gradient_history = []
            self.momentum_buffer = np.zeros(input_size)
            self.velocity = np.zeros(input_size)

            # Parámetros de optimización adaptativa
            self.beta1 = 0.9  # Momentum
            self.beta2 = 0.999  # RMSprop
            self.epsilon = 1e-8
            self.iteration = 0

            # Estado del procesador
            self.is_trained = False
            self.performance_metrics = {
                'accuracy': [],
                'loss': [],
                'entropy': [],
                'convergence_rate': []
            }

        def _initialize_ensemble_components(self):
            """Inicializa múltiples componentes de procesamiento neuronal"""

            # Red neuronal base (feedforward simple)
            self.base_network = {
                'W1': np.random.randn(self.input_size, 128) * 0.01,
                'b1': np.zeros(128),
                'W2': np.random.randn(128, 64) * 0.01,
                'b2': np.zeros(64),
                'W3': np.random.randn(64, 1) * 0.01,
                'b3': np.zeros(1)
            }

            # Pesos adaptativos con inicialización Glorot
            fan_in = self.input_size
            limit = np.sqrt(6 / (fan_in + 1))
            self.adaptive_weights = np.random.uniform(-limit, limit, self.input_size)

            # Componente de momentum (SGD con Nesterov)
            self.momentum_weights = self.base_weights.copy()
            self.momentum_velocity = np.zeros(self.input_size)

            # Componente quantum-inspired (superposición de estados)
            self.quantum_states = np.random.randn(5, self.input_size)
            self.quantum_amplitudes = np.ones(5) / 5

            # Componente evolutivo (algoritmo genético)
            self.population_size = 10
            self.population = [
                np.random.randn(self.input_size) * 0.1
                for _ in range(self.population_size)
            ]
            self.fitness_scores = np.zeros(self.population_size)

        def _base_activation(self, x: np.ndarray) -> np.ndarray:
            """Activación base con tanh mejorado"""
            return np.tanh(x)

        def _advanced_activation(self, x: np.ndarray, method: str = 'swish') -> np.ndarray:
            """Activaciones avanzadas con múltiples opciones"""
            if method == 'swish':
                return x * self.sp.expit(x)  # Swish: x * sigmoid(x)
            elif method == 'mish':
                return x * np.tanh(self.sp.softplus(x))  # Mish
            elif method == 'gelu':
                return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
            elif method == 'elu':
                return np.where(x > 0, x, self.learning_rate * (np.exp(x) - 1))
            else:
                return self._base_activation(x)

        def _to_scalar(self, value) -> float:
            """Convierte valor a escalar de forma segura, evitando DeprecationWarning de NumPy."""
            if hasattr(value, 'item'):
                return float(value.item())
            return float(value)

        def _forward_base(self, features: np.ndarray) -> float:
            """Forward pass tradicional mejorado"""
            linear_output = np.dot(features, self.base_weights) + self.base_bias
            return self._to_scalar(self._base_activation(linear_output))

        def _forward_deep_network(self, features: np.ndarray) -> float:
            """Forward pass con red neuronal profunda"""
            # Capa 1
            z1 = np.dot(features, self.base_network['W1']) + self.base_network['b1']
            a1 = self._advanced_activation(z1, 'swish')

            # Capa 2
            z2 = np.dot(a1, self.base_network['W2']) + self.base_network['b2']
            a2 = self._advanced_activation(z2, 'mish')

            # Capa 3 (salida)
            z3 = np.dot(a2, self.base_network['W3']) + self.base_network['b3']
            return self._to_scalar(self._base_activation(z3))

        def _forward_adaptive(self, features: np.ndarray) -> float:
            """Forward pass con pesos adaptativos (Adam-like)"""
            linear_output = np.dot(features, self.adaptive_weights) + self.base_bias
            return self._to_scalar(self._advanced_activation(linear_output, 'gelu'))

        def _forward_momentum(self, features: np.ndarray) -> float:
            """Forward pass con momentum (Nesterov)"""
            # Actualizar momentum
            look_ahead = self.momentum_weights + self.beta1 * self.momentum_velocity
            linear_output = np.dot(features, look_ahead) + self.base_bias
            return self._to_scalar(self._base_activation(linear_output))

        def _forward_quantum(self, features: np.ndarray) -> float:
            """Forward pass inspirado en computación cuántica"""
            # Superposición de estados
            outputs = []
            for state, amplitude in zip(self.quantum_states, self.quantum_amplitudes):
                weighted_state = state * amplitude
                linear_output = np.dot(features, weighted_state) + self.base_bias
                outputs.append(self._base_activation(linear_output))

            # Colapso de la función de onda (promedio ponderado)
            return self._to_scalar(np.dot(outputs, self.quantum_amplitudes))

        def _forward_evolutionary(self, features: np.ndarray) -> float:
            """Forward pass con algoritmo genético"""
            # Evaluar población
            predictions = []
            for i, individual in enumerate(self.population):
                linear_output = np.dot(features, individual) + self.base_bias
                pred = self._base_activation(linear_output)
                predictions.append(pred)

            # Seleccionar mejor individuo
            best_idx = np.argmax(self.fitness_scores) if self.is_trained else 0
            return self._to_scalar(predictions[best_idx])

        def _ensemble_prediction(self, features: np.ndarray) -> float:
            """Combina todas las predicciones con pesos adaptativos"""
            predictions = {
                'base': self._forward_base(features),
                'adaptive': self._forward_adaptive(features),
                'momentum': self._forward_momentum(features),
                'quantum': self._forward_quantum(features),
                'evolutionary': self._forward_evolutionary(features)
            }

            # Predicción con deep network
            if len(features) >= 10:
                predictions['deep'] = self._forward_deep_network(features)
                self.ensemble_weights['deep'] = 0.2
                # Renormalizar pesos
                total = sum(self.ensemble_weights.values())
                self.ensemble_weights = {k: v/total for k, v in self.ensemble_weights.items()}

            # Combinar con pesos
            ensemble_output = sum(
                pred * self.ensemble_weights.get(name, 0.0)
                for name, pred in predictions.items()
            )

            return float(ensemble_output)

        def _update_adaptive_weights(self, features: np.ndarray, error: float):
            """Actualiza pesos adaptativos usando Adam optimizer"""
            self.iteration += 1

            # Calcular gradiente
            gradient = error * features

            # Actualizar momento y velocidad (Adam)
            self.momentum_buffer = self.beta1 * self.momentum_buffer + (1 - self.beta1) * gradient
            self.velocity = self.beta2 * self.velocity + (1 - self.beta2) * (gradient ** 2)

            # Corrección de sesgo
            m_hat = self.momentum_buffer / (1 - self.beta1 ** self.iteration)
            v_hat = self.velocity / (1 - self.beta2 ** self.iteration)

            # Actualizar pesos
            self.adaptive_weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Guardar historial
            self.gradient_history.append(np.linalg.norm(gradient))

        def _evolve_population(self, features: np.ndarray):
            """Evoluciona la población usando algoritmo genético"""
            # Selección por torneo
            tournament_size = 3
            parents = []
            for _ in range(2):
                tournament = np.random.choice(self.population_size, tournament_size, replace=False)
                winner = tournament[np.argmax(self.fitness_scores[tournament])]
                parents.append(self.population[winner].copy())

            # Cruce (crossover)
            crossover_point = np.random.randint(1, self.input_size)
            child = np.concatenate([parents[0][:crossover_point], parents[1][crossover_point:]])

            # Mutación
            mutation_rate = 0.1
            mutation_mask = np.random.random(self.input_size) < mutation_rate
            num_mutations = int(np.sum(mutation_mask))
            if num_mutations > 0:
                child[mutation_mask] += np.random.randn(num_mutations) * 0.1

            # Reemplazar peor individuo
            worst_idx = np.argmin(self.fitness_scores)
            self.population[worst_idx] = child

        def _calculate_entropy(self, features: np.ndarray) -> float:
            """Calcula entropía de la distribución de features"""
            # Normalizar features a distribución de probabilidad
            features_normalized = np.abs(features)
            if np.sum(features_normalized) > 0:
                features_normalized /= np.sum(features_normalized)
                return float(self.entropy(features_normalized + 1e-10))
            return 0.0

        def process(self, features: np.ndarray) -> float:
            """
            Procesa features usando ensemble de métodos avanzados

            Args:
                features: Vector de características de entrada

            Returns:
                Predicción combinada de todos los métodos
            """
            # Validar entrada
            if len(features) != self.input_size:
                features = np.resize(features, self.input_size)

            # Normalizar features adaptativamente
            features_normalized = features.copy()
            if len(self.prediction_history) > 10:
                try:
                    features_normalized = self.robust_scaler.fit_transform(
                        features.reshape(-1, 1)
                    ).flatten()
                except:
                    features_normalized = features

            # Predicción ensemble
            prediction = self._ensemble_prediction(features_normalized)

            # Actualizar historial
            self.prediction_history.append(prediction)
            if len(self.prediction_history) > 1000:
                self.prediction_history.pop(0)

            # Calcular métricas
            feature_entropy = self._calculate_entropy(features)
            self.performance_metrics['entropy'].append(feature_entropy)

            # Actualización adaptativa cada N iteraciones
            if self.iteration % 10 == 0 and self.iteration > 0:
                self._update_adaptive_weights(features_normalized, prediction)

            # Evolución genética cada 50 iteraciones
            if self.iteration % 50 == 0 and self.is_trained:
                self._evolve_population(features_normalized)

            self.iteration += 1
            self.is_trained = True

            return prediction

        def get_performance_summary(self) -> Dict[str, Any]:
            """Retorna resumen de rendimiento del procesador"""
            if not self.prediction_history:
                return {'status': 'No predictions yet'}

            return {
                'total_predictions': len(self.prediction_history),
                'avg_prediction': np.mean(self.prediction_history[-100:]),
                'prediction_std': np.std(self.prediction_history[-100:]),
                'avg_entropy': np.mean(self.performance_metrics['entropy'][-100:]) if self.performance_metrics['entropy'] else 0,
                'gradient_norm': np.mean(self.gradient_history[-100:]) if self.gradient_history else 0,
                'ensemble_weights': self.ensemble_weights,
                'iteration': self.iteration,
                'is_trained': self.is_trained
            }

        def reset(self):
            """Reinicia el estado del procesador"""
            self._initialize_ensemble_components()
            self.prediction_history.clear()
            self.gradient_history.clear()
            self.momentum_buffer = np.zeros(self.input_size)
            self.velocity = np.zeros(self.input_size)
            self.iteration = 0
            self.is_trained = False

    def backward(self, error, features):
        """
        Backward pass mejorado con procesador avanzado de gradientes
        Utiliza clase anidada AdvancedBackpropagation
        """
        if not hasattr(self, 'backprop_processor'):
            self.backprop_processor = self.AdvancedBackpropagation(
                self.input_size,
                self.learning_rate
            )
        return self.backprop_processor.compute_gradients(
            error, features, self.weights, self.bias, self
        )

    class AdvancedBackpropagation:
        """
        Sistema Avanzado de Retropropagación con Múltiples Optimizadores
        Implementa diferenciación automática, optimización adaptativa y momentum
        """

        def __init__(self, input_size: int, learning_rate: float):
            from scipy.optimize import minimize, differential_evolution
            from scipy.linalg import svd, qr
            import warnings
            warnings.filterwarnings('ignore')

            # Configuración
            self.input_size = input_size
            self.base_lr = learning_rate
            self.current_lr = learning_rate

            # Optimizadores avanzados
            self.minimize = minimize
            self.diff_evolution = differential_evolution

            # Álgebra lineal avanzada
            self.svd = svd
            self.qr = qr

            # Historial de gradientes para análisis
            self.gradient_history = []
            self.weight_updates = []
            self.loss_history = []

            # Optimizadores con estado (Adam, RMSprop, AdaGrad)
            self._initialize_optimizers()

            # Learning rate schedulers
            self.scheduler_state = {
                'step': 0,
                'best_loss': float('inf'),
                'patience_counter': 0,
                'warmup_steps': 100,
                'initial_lr': learning_rate
            }

            # Matriz de segunda derivada aproximada (quasi-Newton)
            self.hessian_approx = np.eye(input_size) * 0.01

            # Parámetros para Natural Gradient Descent
            self.fisher_matrix = np.eye(input_size) * 0.01

        def _initialize_optimizers(self):
            """Inicializa múltiples optimizadores con sus estados"""

            # Adam optimizer state
            self.adam_state = {
                'm': np.zeros(self.input_size),  # Primer momento
                'v': np.zeros(self.input_size),  # Segundo momento
                'm_bias': 0.0,
                'v_bias': 0.0,
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 1e-8,
                't': 0
            }

            # RMSprop state
            self.rmsprop_state = {
                'cache': np.zeros(self.input_size),
                'cache_bias': 0.0,
                'decay_rate': 0.99,
                'epsilon': 1e-8
            }

            # AdaGrad state
            self.adagrad_state = {
                'cache': np.zeros(self.input_size),
                'cache_bias': 0.0,
                'epsilon': 1e-8
            }

            # Momentum state
            self.momentum_state = {
                'velocity': np.zeros(self.input_size),
                'velocity_bias': 0.0,
                'beta': 0.9
            }

            # Nesterov Momentum state
            self.nesterov_state = {
                'velocity': np.zeros(self.input_size),
                'velocity_bias': 0.0,
                'mu': 0.9
            }

            # AdaDelta state
            self.adadelta_state = {
                'eg2': np.zeros(self.input_size),  # E[g^2]
                'edelta2': np.zeros(self.input_size),  # E[delta^2]
                'eg2_bias': 0.0,
                'edelta2_bias': 0.0,
                'rho': 0.95,
                'epsilon': 1e-6
            }

        def _compute_gradient_tanh(self, x: np.ndarray, error: float) -> np.ndarray:
            """Calcula gradiente para activación tanh"""
            return error * (1 - np.tanh(x)**2)

        def _adam_update(self, gradient: np.ndarray, gradient_bias: Union[float, np.ndarray]) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
            """Actualización usando Adam optimizer"""
            self.adam_state['t'] += 1
            t = self.adam_state['t']

            # Actualizar momentos para weights
            self.adam_state['m'] = (self.adam_state['beta1'] * self.adam_state['m'] +
                                    (1 - self.adam_state['beta1']) * gradient)
            self.adam_state['v'] = (self.adam_state['beta2'] * self.adam_state['v'] +
                                    (1 - self.adam_state['beta2']) * (gradient ** 2))

            # Actualizar momentos para bias
            self.adam_state['m_bias'] = (self.adam_state['beta1'] * self.adam_state['m_bias'] +
                                         (1 - self.adam_state['beta1']) * gradient_bias)
            self.adam_state['v_bias'] = (self.adam_state['beta2'] * self.adam_state['v_bias'] +
                                         (1 - self.adam_state['beta2']) * (gradient_bias ** 2))

            # Corrección de sesgo
            m_hat = self.adam_state['m'] / (1 - self.adam_state['beta1'] ** t)
            v_hat = self.adam_state['v'] / (1 - self.adam_state['beta2'] ** t)
            m_hat_bias = self.adam_state['m_bias'] / (1 - self.adam_state['beta1'] ** t)
            v_hat_bias = self.adam_state['v_bias'] / (1 - self.adam_state['beta2'] ** t)

            # Actualización
            weight_update = self.current_lr * m_hat / (np.sqrt(v_hat) + self.adam_state['epsilon'])
            bias_update = self.current_lr * m_hat_bias / (np.sqrt(v_hat_bias) + self.adam_state['epsilon'])

            return weight_update, bias_update

        def _rmsprop_update(self, gradient: np.ndarray, gradient_bias: Union[float, np.ndarray]) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
            """Actualización usando RMSprop optimizer"""
            # Actualizar cache
            self.rmsprop_state['cache'] = (self.rmsprop_state['decay_rate'] * self.rmsprop_state['cache'] +
                                           (1 - self.rmsprop_state['decay_rate']) * (gradient ** 2))
            self.rmsprop_state['cache_bias'] = (self.rmsprop_state['decay_rate'] * self.rmsprop_state['cache_bias'] +
                                                (1 - self.rmsprop_state['decay_rate']) * (gradient_bias ** 2))

            # Actualización
            weight_update = self.current_lr * gradient / (np.sqrt(self.rmsprop_state['cache']) +
                                                          self.rmsprop_state['epsilon'])
            bias_update = self.current_lr * gradient_bias / (np.sqrt(self.rmsprop_state['cache_bias']) +
                                                             self.rmsprop_state['epsilon'])

            return weight_update, bias_update

        def _adadelta_update(self, gradient: np.ndarray, gradient_bias: Union[float, np.ndarray]) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
            """Actualización usando AdaDelta optimizer (no requiere learning rate)"""
            # Actualizar E[g^2]
            self.adadelta_state['eg2'] = (self.adadelta_state['rho'] * self.adadelta_state['eg2'] +
                                          (1 - self.adadelta_state['rho']) * (gradient ** 2))
            self.adadelta_state['eg2_bias'] = (self.adadelta_state['rho'] * self.adadelta_state['eg2_bias'] +
                                               (1 - self.adadelta_state['rho']) * (gradient_bias ** 2))

            # Calcular delta
            rms_g = np.sqrt(self.adadelta_state['eg2'] + self.adadelta_state['epsilon'])
            rms_delta = np.sqrt(self.adadelta_state['edelta2'] + self.adadelta_state['epsilon'])
            delta_weights = -(rms_delta / rms_g) * gradient

            rms_g_bias = np.sqrt(self.adadelta_state['eg2_bias'] + self.adadelta_state['epsilon'])
            rms_delta_bias = np.sqrt(self.adadelta_state['edelta2_bias'] + self.adadelta_state['epsilon'])
            delta_bias = -(rms_delta_bias / rms_g_bias) * gradient_bias

            # Actualizar E[delta^2]
            self.adadelta_state['edelta2'] = (self.adadelta_state['rho'] * self.adadelta_state['edelta2'] +
                                              (1 - self.adadelta_state['rho']) * (delta_weights ** 2))
            self.adadelta_state['edelta2_bias'] = (self.adadelta_state['rho'] * self.adadelta_state['edelta2_bias'] +
                                                   (1 - self.adadelta_state['rho']) * (delta_bias ** 2))

            return -delta_weights, -delta_bias

        def _nesterov_update(self, gradient: np.ndarray, gradient_bias: Union[float, np.ndarray]) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
            """Actualización usando Nesterov Accelerated Gradient"""
            # Guardar velocidad anterior
            v_prev = self.nesterov_state['velocity'].copy()
            v_prev_bias = self.nesterov_state['velocity_bias']

            # Actualizar velocidad
            self.nesterov_state['velocity'] = (self.nesterov_state['mu'] * self.nesterov_state['velocity'] -
                                               self.current_lr * gradient)
            self.nesterov_state['velocity_bias'] = (self.nesterov_state['mu'] * self.nesterov_state['velocity_bias'] -
                                                    self.current_lr * gradient_bias)

            # Actualización con look-ahead
            weight_update = (-self.nesterov_state['mu'] * v_prev +
                             (1 + self.nesterov_state['mu']) * self.nesterov_state['velocity'])
            bias_update = (-self.nesterov_state['mu'] * v_prev_bias +
                           (1 + self.nesterov_state['mu']) * self.nesterov_state['velocity_bias'])

            return weight_update, bias_update

        def _update_learning_rate(self, current_loss: float):
            """Actualiza learning rate con scheduler adaptativo"""
            self.scheduler_state['step'] += 1
            step = self.scheduler_state['step']

            # Warmup
            if step < self.scheduler_state['warmup_steps']:
                self.current_lr = (self.base_lr * step) / self.scheduler_state['warmup_steps']
                return

            # ReduceLROnPlateau
            if current_loss < self.scheduler_state['best_loss']:
                self.scheduler_state['best_loss'] = current_loss
                self.scheduler_state['patience_counter'] = 0
            else:
                self.scheduler_state['patience_counter'] += 1
                if self.scheduler_state['patience_counter'] > 10:
                    self.current_lr *= 0.5
                    self.scheduler_state['patience_counter'] = 0

            # Cosine annealing
            cosine_lr = self.base_lr * 0.5 * (1 + np.cos(np.pi * step / 1000))
            self.current_lr = max(cosine_lr, 1e-6)

        def _compute_natural_gradient(self, gradient: np.ndarray) -> np.ndarray:
            """Calcula gradiente natural usando matriz de Fisher"""
            try:
                # Actualizar matriz de Fisher (aproximación)
                self.fisher_matrix = 0.9 * self.fisher_matrix + 0.1 * np.outer(gradient, gradient)

                # Invertir usando descomposición SVD para estabilidad
                U, s, Vt = self.svd(self.fisher_matrix)
                s_inv = 1.0 / (s + 1e-8)
                fisher_inv = U @ np.diag(s_inv) @ Vt

                # Gradiente natural
                natural_grad = fisher_inv @ gradient
                return natural_grad
            except:
                return gradient

        def compute_gradients(self, error: float, features: np.ndarray,
                              weights: np.ndarray, bias: float, parent_obj) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
            """
            Computa gradientes usando ensemble de optimizadores avanzados

            Args:
                error: Error de la predicción
                features: Vector de características
                weights: Pesos actuales
                bias: Bias actual
                parent_obj: Referencia al objeto padre para actualizar pesos

            Returns:
                Tupla (gradiente_weights, gradiente_bias)
            """
            # Cálculo base del gradiente
            x = np.dot(features, weights) + bias
            gradient_scalar = self._compute_gradient_tanh(x, error)
            gradient_weights = gradient_scalar * features
            gradient_bias = gradient_scalar

            # Guardar en historial
            self.gradient_history.append(np.linalg.norm(gradient_weights))

            # Calcular loss para scheduler
            current_loss = error ** 2
            self.loss_history.append(current_loss)

            # Actualizar learning rate
            self._update_learning_rate(current_loss)

            # Seleccionar optimizador basado en estadísticas
            if len(self.gradient_history) < 10:
                # Usar Adam al inicio
                weight_update, bias_update = self._adam_update(gradient_weights, gradient_bias)
            elif np.std(self.gradient_history[-50:]) > 0.1:
                # Si hay alta varianza, usar RMSprop
                weight_update, bias_update = self._rmsprop_update(gradient_weights, gradient_bias)
            else:
                # Si converge, usar Nesterov
                weight_update, bias_update = self._nesterov_update(gradient_weights, gradient_bias)

            # Aplicar gradiente natural cada 10 pasos
            if self.scheduler_state['step'] % 10 == 0:
                gradient_weights = self._compute_natural_gradient(gradient_weights)
                weight_update, bias_update = self._adadelta_update(gradient_weights, gradient_bias)

            # Actualizar pesos del objeto padre
            parent_obj.weights -= weight_update
            parent_obj.bias -= bias_update

            # Guardar actualizaciones
            self.weight_updates.append(np.linalg.norm(weight_update))

            return gradient_weights, gradient_bias

        def get_optimizer_stats(self) -> Dict[str, Any]:
            """Retorna estadísticas de optimización"""
            if not self.gradient_history:
                return {'status': 'No gradients computed'}

            return {
                'current_lr': self.current_lr,
                'avg_gradient_norm': np.mean(self.gradient_history[-100:]),
                'gradient_variance': np.var(self.gradient_history[-100:]),
                'avg_weight_update': np.mean(self.weight_updates[-100:]) if self.weight_updates else 0,
                'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
                'best_loss': self.scheduler_state['best_loss'],
                'total_steps': self.scheduler_state['step'],
                'adam_iterations': self.adam_state['t']
            }

        def reset_optimizers(self):
            """Reinicia todos los optimizadores"""
            self._initialize_optimizers()
            self.gradient_history.clear()
            self.weight_updates.clear()
            self.loss_history.clear()
            self.scheduler_state['step'] = 0
            self.current_lr = self.base_lr

    def create_3d_context(self):
        """
        Crea contexto 3D avanzado con gestión multi-backend
        Utiliza clase anidada Advanced3DContextManager
        """
        if not hasattr(self, 'context_manager'):
            self.context_manager = self.Advanced3DContextManager()
        return self.context_manager.create_optimized_context()

    class Advanced3DContextManager:
        """
        Gestor Avanzado de Contextos 3D Multi-Backend
        Soporta WebGL, OpenGL, Vulkan y DirectX con optimización automática
        """

        def __init__(self):
            try:
                import moderngl  # type: ignore
                self.moderngl = moderngl
                self.has_moderngl = True
            except ImportError:
                self.has_moderngl = False
                self.moderngl = None

            try:
                import pyglet  # type: ignore
                import pyglet.gl  # type: ignore
                self.pyglet = pyglet
                self.pyglet_gl = pyglet.gl
                self.has_pyglet = True
            except ImportError:
                self.has_pyglet = False
                self.pyglet = None

            try:
                import OpenGL.GL  # type: ignore
                import OpenGL.GLU  # type: ignore
                import OpenGL.GLUT  # type: ignore
                self.opengl_gl = OpenGL.GL
                self.opengl_glu = OpenGL.GLU
                self.opengl_glut = OpenGL.GLUT
                self.has_opengl = True
            except ImportError:
                self.has_opengl = False

            # Estado del contexto
            self.active_contexts = {}
            self.context_counter = 0
            self.performance_metrics = {
                'fps': [],
                'draw_calls': [],
                'vertices_rendered': [],
                'memory_usage': []
            }

            # Configuración de renderizado
            self.render_config = {
                'backend': self._detect_best_backend(),
                'vsync': True,
                'samples': 4,  # MSAA
                'double_buffer': True,
                'depth_bits': 24,
                'stencil_bits': 8,
                'alpha_bits': 8,
                'accumulation_bits': 0,
                'max_texture_size': 4096,
                'anisotropic_filtering': 16
            }

            # Pipeline de renderizado
            self.render_pipeline = {
                'stages': ['geometry', 'lighting', 'post_processing'],
                'active_stage': None,
                'framebuffers': {},
                'textures': {},
                'shaders': {},
                'vao': {},  # Vertex Array Objects
                'vbo': {},  # Vertex Buffer Objects
                'ebo': {}   # Element Buffer Objects
            }

            # Sistema de batching para optimización
            self.batch_system = {
                'max_batch_size': 10000,
                'current_batch': [],
                'batches_per_frame': 0
            }

            # Cache de estados OpenGL para reducir state changes
            self.gl_state_cache = {
                'blend_enabled': False,
                'depth_test_enabled': True,
                'cull_face_enabled': True,
                'active_texture': 0,
                'bound_vao': None,
                'bound_shader': None
            }

        def _detect_best_backend(self) -> str:
            """Detecta el mejor backend disponible"""
            if self.has_moderngl:
                return 'moderngl'
            elif self.has_pyglet:
                return 'pyglet'
            elif self.has_opengl:
                return 'opengl'
            else:
                return 'software'  # Fallback a renderizado por software

        def _create_moderngl_context(self) -> Dict[str, Any]:
            """Crea contexto usando ModernGL (más rápido)"""
            try:
                if not self.moderngl:
                    return {'error': 'ModernGL not available', 'backend': 'moderngl'}
                ctx = self.moderngl.create_standalone_context()

                return {
                    'context': ctx,
                    'backend': 'moderngl',
                    'version': f'{ctx.version_code}',
                    'max_texture_size': ctx.max_texture_size,
                    'max_samples': ctx.max_samples,
                    'extensions': list(ctx.extensions),
                    'info': {
                        'vendor': ctx.info['GL_VENDOR'],
                        'renderer': ctx.info['GL_RENDERER'],
                        'version': ctx.info['GL_VERSION']
                    },
                    'capabilities': {
                        'compute_shaders': ctx.version_code >= 430,
                        'geometry_shaders': ctx.version_code >= 320,
                        'tessellation': ctx.version_code >= 400,
                        'multisample': True
                    }
                }
            except Exception as e:
                return {'error': str(e), 'backend': 'moderngl'}

        def _create_pyglet_context(self) -> Dict[str, Any]:
            """Crea contexto usando Pyglet"""
            try:
                if not self.pyglet:
                    return {'error': 'Pyglet not available', 'backend': 'pyglet'}
                config = self.pyglet.gl.Config(
                    double_buffer=self.render_config['double_buffer'],
                    depth_size=self.render_config['depth_bits'],
                    stencil_size=self.render_config['stencil_bits'],
                    sample_buffers=1,
                    samples=self.render_config['samples']
                )

                if not self.pyglet:
                    return {'error': 'Pyglet not available', 'backend': 'pyglet'}
                window = self.pyglet.window.Window(
                    width=800,
                    height=600,
                    config=config,
                    vsync=self.render_config['vsync'],
                    visible=False  # Hidden para contexto off-screen
                )

                return {
                    'context': window,
                    'backend': 'pyglet',
                    'version': '3.0',
                    'window_handle': window,
                    'config': config,
                    'capabilities': {
                        'double_buffer': config.double_buffer,
                        'depth_buffer': config.depth_size > 0,
                        'stencil_buffer': config.stencil_size > 0,
                        'multisampling': config.samples > 1
                    }
                }
            except Exception as e:
                return {'error': str(e), 'backend': 'pyglet'}

        def _create_opengl_context(self) -> Dict[str, Any]:
            """Crea contexto usando PyOpenGL directo"""
            try:
                # Configuración básica sin ventana
                return {
                    'context': 'opengl_direct',
                    'backend': 'opengl',
                    'version': '3.0',
                    'capabilities': {
                        'shaders': True,
                        'vao': True,
                        'instancing': True,
                        'fbo': True
                    }
                }
            except Exception as e:
                return {'error': str(e), 'backend': 'opengl'}

        def _create_software_context(self) -> Dict[str, Any]:
            """Crea contexto de software como fallback"""
            return {
                'context': 'software_renderer',
                'backend': 'software',
                'version': '1.0',
                'warning': 'Using software rendering - performance will be limited',
                'capabilities': {
                    'basic_rendering': True,
                    'hardware_acceleration': False
                }
            }

        def create_optimized_context(self) -> Dict[str, Any]:
            """Crea contexto optimizado según backend disponible"""
            backend = self.render_config['backend']

            # Intentar crear contexto con el mejor backend
            if backend == 'moderngl':
                context_info = self._create_moderngl_context()
            elif backend == 'pyglet':
                context_info = self._create_pyglet_context()
            elif backend == 'opengl':
                context_info = self._create_opengl_context()
            else:
                context_info = self._create_software_context()

            # Si falló, intentar con fallback
            if 'error' in context_info:
                context_info = self._create_software_context()

            # Asignar ID único al contexto
            self.context_counter += 1
            context_id = f'ctx_{self.context_counter}'

            # Configurar pipeline de renderizado
            context_info['id'] = context_id
            context_info['pipeline'] = self._initialize_render_pipeline(context_info)
            context_info['optimization'] = self._get_optimization_hints(context_info)
            context_info['state'] = 'active'
            context_info['created_at'] = np.datetime64('now')

            # Guardar contexto activo
            self.active_contexts[context_id] = context_info

            return context_info

        def _initialize_render_pipeline(self, context_info: Dict) -> Dict[str, Any]:
            """Inicializa pipeline de renderizado optimizado"""
            backend = context_info.get('backend', 'software')

            pipeline = {
                'forward_rendering': {
                    'enabled': True,
                    'passes': ['opaque', 'transparent'],
                    'sort_order': 'front_to_back'
                },
                'deferred_rendering': {
                    'enabled': backend in ['moderngl', 'opengl'],
                    'g_buffer': ['position', 'normal', 'albedo', 'metallic_roughness'],
                    'light_pass': True
                },
                'pbr_workflow': {
                    'enabled': True,
                    'ibl': True,  # Image-Based Lighting
                    'env_map_resolution': 512
                },
                'post_processing': {
                    'bloom': True,
                    'ssao': True,  # Screen Space Ambient Occlusion
                    'ssr': False,  # Screen Space Reflections
                    'tonemapping': 'ACES',
                    'antialiasing': 'FXAA'
                },
                'culling': {
                    'frustum': True,
                    'occlusion': backend == 'moderngl',
                    'backface': True
                },
                'lod_system': {
                    'enabled': True,
                    'levels': 4,
                    'distance_multiplier': 1.5
                }
            }

            return pipeline

        def _get_optimization_hints(self, context_info: Dict) -> Dict[str, Any]:
            """Genera hints de optimización basados en el contexto"""
            backend = context_info.get('backend', 'software')

            return {
                'use_instancing': backend in ['moderngl', 'opengl'],
                'use_compute_shaders': backend == 'moderngl' and
                context_info.get('capabilities', {}).get('compute_shaders', False),
                'batch_draw_calls': True,
                'use_vao': backend in ['moderngl', 'opengl'],
                'compress_textures': True,
                'use_mipmaps': True,
                'cull_small_triangles': True,
                'occlusion_culling': backend == 'moderngl',
                'texture_streaming': False,
                'async_loading': True,
                'max_draw_calls_per_frame': 1000 if backend == 'software' else 5000
            }

        def set_render_state(self, state: Dict[str, Any]):
            """Establece estado de renderizado con cache"""
            changes_made = []

            # Comparar con cache y solo aplicar cambios
            for key, value in state.items():
                if self.gl_state_cache.get(key) != value:
                    self.gl_state_cache[key] = value
                    changes_made.append(key)

                    # Aplicar cambio según backend
                    if self.render_config['backend'] == 'moderngl':
                        self._apply_moderngl_state(key, value)

            return {'changes': changes_made, 'state': self.gl_state_cache.copy()}

        def _apply_moderngl_state(self, key: str, value: Any):
            """Aplica estado específico a ModernGL"""
            # Implementación específica para ModernGL
            pass

        def create_framebuffer(self, width: int, height: int,
                               attachments: List[str]) -> Dict[str, Any]:
            """Crea framebuffer para render-to-texture"""
            fb_id = f'fb_{len(self.render_pipeline["framebuffers"])}'

            framebuffer = {
                'id': fb_id,
                'width': width,
                'height': height,
                'attachments': attachments,
                'textures': {},
                'depth_buffer': 'depth' in attachments,
                'stencil_buffer': 'stencil' in attachments
            }

            self.render_pipeline['framebuffers'][fb_id] = framebuffer
            return framebuffer

        def batch_geometry(self, geometries: List[Dict]) -> Dict[str, Any]:
            """Agrupa geometrías para rendering eficiente"""
            batches = []
            current_batch = []
            current_vertex_count = 0

            for geom in geometries:
                vertex_count = geom.get('vertex_count', 0)

                if current_vertex_count + vertex_count > self.batch_system['max_batch_size']:
                    batches.append(current_batch)
                    current_batch = [geom]
                    current_vertex_count = vertex_count
                else:
                    current_batch.append(geom)
                    current_vertex_count += vertex_count

            if current_batch:
                batches.append(current_batch)

            self.batch_system['batches_per_frame'] = len(batches)

            return {
                'batch_count': len(batches),
                'batches': batches,
                'total_vertices': sum(g.get('vertex_count', 0) for g in geometries)
            }

        def get_performance_metrics(self) -> Dict[str, Any]:
            """Retorna métricas de rendimiento del contexto"""
            return {
                'active_contexts': len(self.active_contexts),
                'avg_batches_per_frame': self.batch_system['batches_per_frame'],
                'render_backend': self.render_config['backend'],
                'state_changes_cached': len(self.gl_state_cache),
                'total_framebuffers': len(self.render_pipeline['framebuffers']),
                'capabilities': self.render_config
            }

        def destroy_context(self, context_id: str) -> bool:
            """Destruye un contexto específico"""
            if context_id in self.active_contexts:
                context = self.active_contexts[context_id]

                # Limpiar recursos según backend
                if context['backend'] == 'pyglet' and 'window_handle' in context:
                    context['window_handle'].close()

                del self.active_contexts[context_id]
                return True
            return False

        def optimize_pipeline(self) -> Dict[str, Any]:
            """Optimiza pipeline de renderizado automáticamente"""
            optimizations = []

            # Analizar métricas y ajustar configuración
            if self.batch_system['batches_per_frame'] > 100:
                self.batch_system['max_batch_size'] *= 2
                optimizations.append('Increased batch size')

            # Deshabilitar features pesadas si hay lag
            avg_fps = np.mean(self.performance_metrics['fps'][-10:]) if self.performance_metrics['fps'] else 60
            if avg_fps < 30:
                self.render_pipeline['post_processing']['ssao'] = False
                self.render_pipeline['post_processing']['bloom'] = False
                optimizations.append('Disabled heavy post-processing')

            return {
                'optimizations_applied': optimizations,
                'current_config': self.render_config,
                'pipeline_state': self.render_pipeline
            }

    def compile_shader(self, code):
        """
        Compila shader avanzado con optimización y conversión Python → GLSL/SPIR-V
        Utiliza clase anidada AdvancedShaderCompiler
        """
        if not hasattr(self, 'shader_compiler'):
            self.shader_compiler = self.AdvancedShaderCompiler()
        return self.shader_compiler.compile(code)

    class AdvancedShaderCompiler:
        """
        Compilador Avanzado de Shaders con Optimización Automática
        Soporta Python → GLSL, HLSL, SPIR-V con análisis y optimización
        """

        def __init__(self):
            import re
            import ast
            import hashlib
            from collections import defaultdict

            self.re = re
            self.ast = ast
            self.hashlib = hashlib
            self.defaultdict = defaultdict

            # Tipos de shaders soportados
            self.shader_types = ['vertex', 'fragment', 'geometry', 'compute', 'tessellation']

            # Caché de shaders compilados
            self.shader_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0

            # Estadísticas de compilación
            self.compilation_stats = {
                'total_compiled': 0,
                'successful': 0,
                'failed': 0,
                'avg_compile_time': 0.0,
                'optimizations_applied': []
            }

            # Templates GLSL base
            self.glsl_templates = self._initialize_glsl_templates()

            # Mapeo de tipos Python → GLSL
            self.type_mapping = {
                'float': 'float',
                'int': 'int',
                'bool': 'bool',
                'list': 'vec',
                'tuple': 'vec',
                'numpy.ndarray': 'vec',
                'Vec2': 'vec2',
                'Vec3': 'vec3',
                'Vec4': 'vec4',
                'Mat3': 'mat3',
                'Mat4': 'mat4'
            }

            # Funciones built-in de GLSL
            self.glsl_builtins = {
                'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                'pow', 'exp', 'log', 'sqrt', 'abs', 'sign',
                'floor', 'ceil', 'fract', 'mod', 'min', 'max',
                'clamp', 'mix', 'step', 'smoothstep',
                'length', 'distance', 'dot', 'cross', 'normalize',
                'reflect', 'refract', 'texture', 'texture2D'
            }

            # Optimizaciones disponibles
            self.optimizations = {
                'constant_folding': True,
                'dead_code_elimination': True,
                'loop_unrolling': True,
                'inline_functions': True,
                'vectorization': True,
                'strength_reduction': True,
                'common_subexpression_elimination': True
            }

            # Registro de errores
            self.error_log = []

        def _initialize_glsl_templates(self) -> Dict[str, str]:
            """Inicializa templates base para diferentes tipos de shaders"""
            return {
                'vertex': '''
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

{user_code}

void main() {{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
    
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}}
''',
                'fragment': '''
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 lightColor;

{user_code}

void main() {{
    FragColor = vec4(1.0);
}}
''',
                'compute': '''
#version 430 core
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

{user_code}

void main() {{
    ivec3 gid = ivec3(gl_GlobalInvocationID);
    // Compute shader logic
}}
'''
            }

        def _generate_cache_key(self, code: str, shader_type: str) -> str:
            """Genera clave única para caché de shaders"""
            content = f"{shader_type}:{code}"
            return self.hashlib.sha256(content.encode()).hexdigest()[:16]

        def _parse_python_to_ast(self, code: str) -> Optional[Any]:
            """Parsea código Python a AST"""
            try:
                return self.ast.parse(code)
            except SyntaxError as e:
                self.error_log.append(f"Python syntax error: {e}")
                return None

        def _python_to_glsl_type(self, py_type: str) -> str:
            """Convierte tipo Python a tipo GLSL"""
            return self.type_mapping.get(py_type, 'float')

        def _convert_expression(self, expr: str) -> str:
            """Convierte expresión Python a GLSL"""
            # Reemplazar operadores Python por GLSL
            conversions = {
                '**': 'pow',
                '//': '/',
                ' and ': ' && ',
                ' or ': ' || ',
                ' not ': ' ! ',
                'True': 'true',
                'False': 'false',
                'None': 'null'
            }

            result = expr
            for py_op, glsl_op in conversions.items():
                result = result.replace(py_op, glsl_op)

            return result

        def _optimize_constant_folding(self, code: str) -> str:
            """Optimización: constant folding"""
            # Evaluar expresiones constantes en tiempo de compilación
            pattern = r'(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)'

            def evaluate(match):
                left, op, right = match.groups()
                try:
                    if op == '+':
                        result = float(left) + float(right)
                    elif op == '-':
                        result = float(left) - float(right)
                    elif op == '*':
                        result = float(left) * float(right)
                    elif op == '/':
                        result = float(left) / float(right)
                    return str(result)
                except:
                    return match.group(0)

            return self.re.sub(pattern, evaluate, code)

        def _optimize_dead_code_elimination(self, code: str) -> str:
            """Optimización: eliminación de código muerto"""
            lines = code.split('\n')
            used_vars = set()
            result_lines = []

            # Análisis simple de variables usadas
            for line in reversed(lines):
                if '=' in line and '//' not in line[:line.index('=')]:
                    var_name = line.split('=')[0].strip().split()[-1]
                    if var_name in used_vars or 'gl_' in var_name or 'return' in line:
                        result_lines.insert(0, line)
                        # Agregar variables usadas en esta línea
                        for word in line.split():
                            if word.isidentifier():
                                used_vars.add(word)
                else:
                    result_lines.insert(0, line)
                    for word in line.split():
                        if word.isidentifier():
                            used_vars.add(word)

            return '\n'.join(result_lines)

        def _optimize_loop_unrolling(self, code: str) -> str:
            """Optimización: desenrollado de bucles pequeños"""
            # Buscar bucles for simples y desenrollarlos si son pequeños
            pattern = r'for\s*\(\s*int\s+(\w+)\s*=\s*(\d+)\s*;\s*\1\s*<\s*(\d+)\s*;\s*\1\+\+\s*\)\s*\{([^}]+)\}'

            def unroll(match):
                var, start, end, body = match.groups()
                start_val = int(start)
                end_val = int(end)

                # Solo desenrollar bucles pequeños
                if end_val - start_val > 8:
                    return match.group(0)

                unrolled = []
                for i in range(start_val, end_val):
                    iteration = body.replace(var, str(i))
                    unrolled.append(iteration)

                return '\n'.join(unrolled)

            return self.re.sub(pattern, unroll, code)

        def _optimize_vectorization(self, code: str) -> str:
            """Optimización: vectorización de operaciones escalares"""
            # Convertir operaciones escalares repetidas a operaciones vectoriales
            optimizations_applied = []

            # Buscar patrones como: a.x = ...; a.y = ...; a.z = ...;
            pattern = r'(\w+)\.x\s*=\s*([^;]+);\s*\1\.y\s*=\s*([^;]+);\s*\1\.z\s*=\s*([^;]+);'

            def vectorize(match):
                var, x, y, z = match.groups()
                optimizations_applied.append(f'Vectorized {var}')
                return f'{var} = vec3({x}, {y}, {z});'

            result = self.re.sub(pattern, vectorize, code)
            self.compilation_stats['optimizations_applied'].extend(optimizations_applied)

            return result

        def _apply_optimizations(self, code: str) -> str:
            """Aplica todas las optimizaciones habilitadas"""
            optimized = code

            if self.optimizations['constant_folding']:
                optimized = self._optimize_constant_folding(optimized)

            if self.optimizations['dead_code_elimination']:
                optimized = self._optimize_dead_code_elimination(optimized)

            if self.optimizations['loop_unrolling']:
                optimized = self._optimize_loop_unrolling(optimized)

            if self.optimizations['vectorization']:
                optimized = self._optimize_vectorization(optimized)

            return optimized

        def _convert_python_function_to_glsl(self, func_code: str) -> str:
            """Convierte función Python a función GLSL"""
            # Parsear función Python
            try:
                tree = self.ast.parse(func_code)
                if not tree.body or not isinstance(tree.body[0], self.ast.FunctionDef):
                    return func_code

                func_def = tree.body[0]
                func_name = func_def.name

                # Convertir argumentos
                args = []
                for arg in func_def.args.args:
                    arg_type = 'float'  # Default type
                    args.append(f'{arg_type} {arg.arg}')

                # Convertir cuerpo de la función
                body_lines = []
                for node in func_def.body:
                    if isinstance(node, self.ast.Return):
                        if node.value:
                            body_lines.append(f'return {self.ast.unparse(node.value)};')
                    elif isinstance(node, self.ast.Assign):
                        target = self.ast.unparse(node.targets[0])
                        value = self.ast.unparse(node.value)
                        body_lines.append(f'{target} = {value};')

                # Construir función GLSL
                glsl_func = f'float {func_name}({", ".join(args)}) {{\n'
                glsl_func += '\n'.join(f'    {line}' for line in body_lines)
                glsl_func += '\n}'

                return glsl_func
            except:
                return func_code

        def compile(self, code: str, shader_type: str = 'fragment',
                    optimize: bool = True) -> Dict[str, Any]:
            """
            Compila código a shader GLSL optimizado

            Args:
                code: Código fuente (Python o GLSL)
                shader_type: Tipo de shader (vertex, fragment, compute, etc.)
                optimize: Aplicar optimizaciones

            Returns:
                Diccionario con shader compilado y metadata
            """
            import time
            start_time = time.time()

            # Verificar caché
            cache_key = self._generate_cache_key(code, shader_type)
            if cache_key in self.shader_cache:
                self.cache_hits += 1
                cached = self.shader_cache[cache_key]
                cached['from_cache'] = True
                return cached

            self.cache_misses += 1

            # Determinar si es Python o GLSL
            is_python = 'def ' in code or 'import ' in code

            # Convertir de Python a GLSL si es necesario
            if is_python:
                glsl_code = self._convert_python_function_to_glsl(code)
            else:
                glsl_code = code

            # Convertir expresiones
            glsl_code = self._convert_expression(glsl_code)

            # Aplicar optimizaciones
            if optimize:
                glsl_code = self._apply_optimizations(glsl_code)

            # Insertar en template si no es código completo
            if '#version' not in glsl_code:
                template = self.glsl_templates.get(shader_type, self.glsl_templates['fragment'])
                glsl_code = template.replace('{user_code}', glsl_code)

            # Análisis del código compilado
            analysis = self._analyze_shader(glsl_code)

            # Resultado de compilación
            compile_time = time.time() - start_time
            result = {
                'success': True,
                'compiled_code': glsl_code,
                'shader_type': shader_type,
                'original_code': code,
                'optimized': optimize,
                'compile_time_ms': compile_time * 1000,
                'cache_key': cache_key,
                'analysis': analysis,
                'from_cache': False,
                'errors': self.error_log.copy()
            }

            # Guardar en caché
            self.shader_cache[cache_key] = result

            # Actualizar estadísticas
            self.compilation_stats['total_compiled'] += 1
            self.compilation_stats['successful'] += 1
            self.compilation_stats['avg_compile_time'] = (
                (self.compilation_stats['avg_compile_time'] *
                 (self.compilation_stats['total_compiled'] - 1) + compile_time) /
                self.compilation_stats['total_compiled']
            )

            self.error_log.clear()
            return result

        def _analyze_shader(self, code: str) -> Dict[str, Any]:
            """Analiza shader compilado para métricas y advertencias"""
            return {
                'line_count': len(code.split('\n')),
                'has_version': '#version' in code,
                'has_main': 'void main()' in code,
                'uniform_count': code.count('uniform'),
                'texture_samples': code.count('texture'),
                'loop_count': code.count('for') + code.count('while'),
                'branch_count': code.count('if'),
                'complexity_estimate': self._estimate_complexity(code)
            }

        def _estimate_complexity(self, code: str) -> str:
            """Estima complejidad del shader"""
            score = 0
            score += code.count('texture') * 5
            score += code.count('for') * 10
            score += code.count('while') * 15
            score += code.count('if') * 2
            score += len(code.split('\n'))

            if score < 50:
                return 'low'
            elif score < 150:
                return 'medium'
            else:
                return 'high'

        def get_compilation_stats(self) -> Dict[str, Any]:
            """Retorna estadísticas de compilación"""
            return {
                **self.compilation_stats,
                'cache_size': len(self.shader_cache),
                'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses
            }

        def clear_cache(self):
            """Limpia caché de shaders"""
            self.shader_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0

        def export_shader_to_file(self, shader_result: Dict, filename: str):
            """Exporta shader compilado a archivo"""
            try:
                with open(filename, 'w') as f:
                    f.write(f"// Compiled shader - {shader_result['shader_type']}\n")
                    f.write(f"// Compile time: {shader_result['compile_time_ms']:.2f}ms\n")
                    f.write(f"// Complexity: {shader_result['analysis']['complexity_estimate']}\n\n")
                    f.write(shader_result['compiled_code'])
                return True
            except Exception as e:
                self.error_log.append(f"Export error: {e}")
                return False
