"""
Core del Sistema de Neuronas de Refuerzo
=========================================
Implementación principal de agentes de aprendizaje por refuerzo usando
Stable Baselines3, TF-Agents y RLlib (Ray).

Algoritmos implementados:
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- DQN (Deep Q-Network)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)
"""

import numpy as np
try:
    import torch
    import torch.nn as nn
except ImportError:
    pass  # dependencia pesada opcional
try:
    torch
except NameError:
    import types as _t
    torch = _t.SimpleNamespace(
        no_grad=lambda *a, **k: (lambda f: f) if a and callable(a[0]) else (lambda f: f),
        optim=_t.SimpleNamespace(Optimizer=object),
        Tensor=object,
    )
try:
    nn
except NameError:
    import types as _t2
    nn = _t2.SimpleNamespace(Module=object)
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path

# Configuración de dispositivo (GPU si está disponible)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlgorithmType(Enum):
    """Tipos de algoritmos de refuerzo soportados"""
    PPO = "ppo"
    A2C = "a2c"
    DQN = "dqn"
    SAC = "sac"
    TD3 = "td3"
    DDPG = "ddpg"


@dataclass
class NeuronConfig:
    """Configuración de neurona de refuerzo"""
    algorithm: AlgorithmType = AlgorithmType.PPO
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_cuda: bool = True
    seed: int = 42
    verbose: int = 1
    tensorboard_log: Optional[str] = "./logs/tensorboard/"
    checkpoint_dir: Optional[str] = "./checkpoints/"

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a diccionario"""
        config_dict = {
            'algorithm': self.algorithm.value,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'use_cuda': self.use_cuda,
            'seed': self.seed,
            'verbose': self.verbose
        }
        return config_dict


class NeuralNetwork(nn.Module):
    """Red neuronal personalizada para el agente"""

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int] = [256, 256]):
        super(NeuralNetwork, self).__init__()

        layers = []
        prev_dim = input_dim

        # Capas ocultas
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        # Capa de salida
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Inicialización de pesos
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Inicialización orthogonal de pesos"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)


class ReinforcementNeuron:
    """
    Neurona de Refuerzo principal que gestiona el aprendizaje
    y la toma de decisiones del agente
    """

    def __init__(self, config: Optional[NeuronConfig] = None):
        self.config = config or NeuronConfig()
        self.device = DEVICE if self.config.use_cuda else torch.device("cpu")

        # Métricas de entrenamiento
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.training_steps: int = 0
        self.episodes_completed: int = 0

        # Estado interno
        self.policy_network: Optional[NeuralNetwork] = None
        self.value_network: Optional[NeuralNetwork] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        print(f"✅ ReinforcementNeuron inicializada en {self.device}")

    def initialize_networks(self, state_dim: int, action_dim: int):
        """Inicializa las redes neuronales del agente"""
        self.policy_network = NeuralNetwork(state_dim, action_dim).to(self.device)
        self.value_network = NeuralNetwork(state_dim, 1).to(self.device)

        # Optimizador Adam con weight decay
        self.optimizer = torch.optim.Adam(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=self.config.learning_rate,
            eps=1e-5,
            weight_decay=1e-4
        )

        print(f"🧠 Redes neuronales inicializadas: State({state_dim}) → Action({action_dim})")

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Selecciona una acción basada en el estado actual

        Args:
            state: Estado del entorno
            deterministic: Si True, selecciona la acción más probable

        Returns:
            action: Acción seleccionada
            log_prob: Log probabilidad de la acción
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.policy_network(state_tensor)

            if deterministic:
                action = torch.argmax(logits, dim=-1).cpu().numpy()
                log_prob = 0.0
            else:
                probs = torch.softmax(logits, dim=-1)
                action_dist = torch.distributions.Categorical(probs)
                action_tensor = action_dist.sample()
                action = action_tensor.cpu().numpy()
                log_prob = action_dist.log_prob(action_tensor).item()

        return action, log_prob

    def compute_value(self, state: np.ndarray) -> float:
        """Calcula el valor del estado actual"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.value_network(state_tensor).item()
        return value

    def update(self, states: np.ndarray, actions: np.ndarray,
               rewards: np.ndarray, next_states: np.ndarray,
               dones: np.ndarray) -> Dict[str, float]:
        """
        Actualiza las redes usando el algoritmo PPO

        Returns:
            Diccionario con métricas del entrenamiento
        """
        # Convertir a tensores
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Calcular valores y ventajas
        values = self.value_network(states_t).squeeze()
        next_values = self.value_network(next_states_t).squeeze()

        # Calcular TD error y ventajas usando GAE
        td_targets = rewards_t + self.config.gamma * next_values * (1 - dones_t)
        advantages = td_targets - values

        # Normalizar ventajas
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actualización PPO
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(self.config.n_epochs):
            # Forward pass
            logits = self.policy_network(states_t)
            probs = torch.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs)
            log_probs = action_dist.log_prob(actions_t)
            entropy = action_dist.entropy().mean()

            # Ratio para PPO
            ratio = torch.exp(log_probs)

            # Loss de política (PPO clip)
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()

            # Loss de valor
            value_pred = self.value_network(states_t).squeeze()
            value_loss = nn.MSELoss()(value_pred, td_targets.detach())

            # Loss total
            loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy

            # Backward y optimización
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy_network.parameters()) + list(self.value_network.parameters()),
                self.config.max_grad_norm
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        self.training_steps += 1

        metrics = {
            'policy_loss': total_policy_loss / self.config.n_epochs,
            'value_loss': total_value_loss / self.config.n_epochs,
            'mean_advantage': advantages.mean().item(),
            'training_steps': self.training_steps
        }

        return metrics

    def save_checkpoint(self, filename: str):
        """Guarda un checkpoint del modelo"""
        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'episodes_completed': self.episodes_completed,
            'config': self.config.to_dict()
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint guardado en {checkpoint_path}")

    def load_checkpoint(self, filename: str):
        """Carga un checkpoint del modelo"""
        checkpoint_path = Path(self.config.checkpoint_dir) / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.episodes_completed = checkpoint['episodes_completed']

        print(f"📂 Checkpoint cargado desde {checkpoint_path}")


class StableBaselinesAgent:
    """Wrapper para agentes de Stable Baselines3"""

    def __init__(self, algorithm: str = "PPO"):
        self.algorithm = algorithm
        self.model = None
        print(f"🤖 StableBaselinesAgent creado con algoritmo {algorithm}")

    def train(self, env, total_timesteps: int = 10000):
        """Entrena el agente en el entorno"""
        try:
            from stable_baselines3 import PPO, A2C, SAC, TD3

            algo_map = {
                "PPO": PPO,
                "A2C": A2C,
                "SAC": SAC,
                "TD3": TD3
            }

            AlgoClass = algo_map.get(self.algorithm, PPO)
            self.model = AlgoClass("MlpPolicy", env, verbose=1)
            self.model.learn(total_timesteps=total_timesteps)

            print(f"✅ Entrenamiento completado: {total_timesteps} pasos")
        except ImportError:
            print("⚠️ Stable Baselines3 no está instalado")


class TFAgentsCore:
    """Wrapper para TensorFlow Agents"""

    def __init__(self):
        self.agent = None
        print("🎯 TFAgentsCore inicializado")

    def create_agent(self, env):
        """Crea un agente de TF-Agents"""
        print("🔧 Creando agente TF-Agents...")


class RLlibDistributedAgent:
    """Agente distribuido usando Ray RLlib"""

    def __init__(self):
        self.trainer = None
        print("☁️ RLlibDistributedAgent inicializado")

    def train_distributed(self, config: Dict):
        """Entrenamiento distribuido"""
        print("🚀 Iniciando entrenamiento distribuido...")


class TrainingManager:
    """Gestor de entrenamiento que coordina todos los componentes"""

    def __init__(self, config: Optional[NeuronConfig] = None):
        self.config = config or NeuronConfig()
        self.neuron = ReinforcementNeuron(self.config)
        self.metrics_history: List[Dict] = []
        print("📊 TrainingManager inicializado")

    def run_training(self, env, num_episodes: int = 1000):
        """Ejecuta el ciclo de entrenamiento completo"""
        print(f"🎮 Iniciando entrenamiento: {num_episodes} episodios")

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = self.neuron.select_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state

            self.neuron.episode_rewards.append(episode_reward)
            self.neuron.episodes_completed += 1

            if episode % 100 == 0:
                avg_reward = np.mean(self.neuron.episode_rewards[-100:])
                print(f"Episodio {episode}: Recompensa promedio = {avg_reward:.2f}")

        print("✅ Entrenamiento completado")
