"""
Sistema de Gestión de Entornos de Metaverso
===========================================
Entornos específicos para entrenamiento de RL en metaversos 3D.
Integración con OpenSimulator y motores físicos.

Características:
- Entornos personalizados para metaversos
- Motor de física realista
- Capa de networking
- Gestión de estados
- Sistema de recompensas adaptativo
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time


class PhysicsType(Enum):
    """Tipos de física soportados"""
    BULLET = "bullet"
    ODE = "ode"
    PHYSX = "physx"
    CUSTOM = "custom"


@dataclass
class EnvironmentConfig:
    """Configuración del entorno de metaverso"""
    name: str = "WoldVirtual3D"
    max_episode_steps: int = 1000
    physics_engine: PhysicsType = PhysicsType.BULLET
    physics_timestep: float = 0.01
    render_mode: Optional[str] = None
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)

    # Límites del mundo
    world_bounds: Tuple[float, float, float] = (1000.0, 1000.0, 1000.0)

    # OpenSim specific
    opensim_enabled: bool = False
    opensim_host: str = "localhost"
    opensim_port: int = 9000

    # Recompensas
    step_penalty: float = -0.01
    goal_reward: float = 100.0
    collision_penalty: float = -10.0


@dataclass
class AgentState:
    """Estado del agente en el metaverso"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(4))  # Quaternion
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    health: float = 100.0
    energy: float = 100.0
    inventory: Dict[str, int] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        """Convierte el estado a array numpy"""
        return np.concatenate([
            self.position,
            self.rotation,
            self.velocity,
            self.angular_velocity,
            [self.health, self.energy]
        ])


class PhysicsEngine:
    """
    Motor de física para simulación realista
    """

    def __init__(self, physics_type: PhysicsType = PhysicsType.BULLET, timestep: float = 0.01):
        self.physics_type = physics_type
        self.timestep = timestep
        self.gravity = np.array([0.0, -9.81, 0.0])

        # Objetos físicos en la escena
        self.rigid_bodies: Dict[str, Dict] = {}
        self.colliders: Dict[str, Dict] = {}

        # Propiedades físicas
        self.air_density = 1.225  # kg/m³
        self.friction_coefficient = 0.5

        print(f"⚙️ PhysicsEngine inicializado ({physics_type.value})")

    def create_rigid_body(self, body_id: str, mass: float, position: np.ndarray,
                          rotation: np.ndarray = None, is_kinematic: bool = False):
        """
        Crea un cuerpo rígido

        Args:
            body_id: Identificador único
            mass: Masa en kg
            position: Posición inicial
            rotation: Rotación inicial (quaternion)
            is_kinematic: Si True, el objeto no es afectado por fuerzas
        """
        if rotation is None:
            rotation = np.array([0, 0, 0, 1])  # Quaternion identidad

        self.rigid_bodies[body_id] = {
            'mass': mass,
            'position': position.copy(),
            'rotation': rotation.copy(),
            'velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'force': np.zeros(3),
            'torque': np.zeros(3),
            'is_kinematic': is_kinematic,
            'colliding': False
        }

        print(f"📦 Cuerpo rígido creado: {body_id} (masa: {mass}kg)")

    def apply_force(self, body_id: str, force: np.ndarray, position: Optional[np.ndarray] = None):
        """Aplica una fuerza a un cuerpo rígido"""
        if body_id not in self.rigid_bodies:
            return

        body = self.rigid_bodies[body_id]

        if not body['is_kinematic']:
            body['force'] += force

            # Si la fuerza se aplica en un punto específico, calcular torque
            if position is not None:
                r = position - body['position']
                torque = np.cross(r, force)
                body['torque'] += torque

    def set_velocity(self, body_id: str, velocity: np.ndarray):
        """Establece la velocidad de un cuerpo"""
        if body_id in self.rigid_bodies:
            self.rigid_bodies[body_id]['velocity'] = velocity.copy()

    def step(self, dt: Optional[float] = None):
        """
        Avanza la simulación física un paso

        Args:
            dt: Delta time (si es None, usa self.timestep)
        """
        dt = dt or self.timestep

        for body_id, body in self.rigid_bodies.items():
            if body['is_kinematic']:
                continue

            # Calcular aceleración
            acceleration = body['force'] / body['mass'] + self.gravity

            # Integración semi-implícita de Euler
            body['velocity'] += acceleration * dt
            body['position'] += body['velocity'] * dt

            # Integración de rotación (simplificada)
            if np.linalg.norm(body['angular_velocity']) > 0:
                # Aquí iría integración de quaternion real
                pass

            # Aplicar fricción del aire
            drag_force = -0.5 * self.air_density * np.linalg.norm(body['velocity']) * body['velocity']
            body['velocity'] += (drag_force / body['mass']) * dt

            # Resetear fuerzas
            body['force'] = np.zeros(3)
            body['torque'] = np.zeros(3)

    def check_collision(self, body_id1: str, body_id2: str, radius1: float = 0.5,
                        radius2: float = 0.5) -> bool:
        """
        Verifica colisión entre dos cuerpos (usando esferas por simplicidad)

        Args:
            body_id1: ID del primer cuerpo
            body_id2: ID del segundo cuerpo
            radius1: Radio de la esfera de colisión del primer cuerpo
            radius2: Radio de la esfera de colisión del segundo cuerpo

        Returns:
            True si hay colisión
        """
        if body_id1 not in self.rigid_bodies or body_id2 not in self.rigid_bodies:
            return False

        pos1 = self.rigid_bodies[body_id1]['position']
        pos2 = self.rigid_bodies[body_id2]['position']

        distance = np.linalg.norm(pos1 - pos2)
        collision = distance < (radius1 + radius2)

        if collision:
            self.rigid_bodies[body_id1]['colliding'] = True
            self.rigid_bodies[body_id2]['colliding'] = True

        return collision

    def resolve_collision(self, body_id1: str, body_id2: str):
        """Resuelve una colisión aplicando impulsos"""
        if body_id1 not in self.rigid_bodies or body_id2 not in self.rigid_bodies:
            return

        body1 = self.rigid_bodies[body_id1]
        body2 = self.rigid_bodies[body_id2]

        # Calcular vector de colisión
        collision_normal = body2['position'] - body1['position']
        distance = np.linalg.norm(collision_normal)

        if distance == 0:
            return

        collision_normal = collision_normal / distance

        # Calcular velocidad relativa
        relative_velocity = body1['velocity'] - body2['velocity']
        velocity_along_normal = np.dot(relative_velocity, collision_normal)

        # No resolver si los objetos se están separando
        if velocity_along_normal > 0:
            return

        # Calcular impulso
        restitution = 0.5  # Coeficiente de restitución
        impulse_scalar = -(1 + restitution) * velocity_along_normal
        impulse_scalar /= (1 / body1['mass'] + 1 / body2['mass'])

        impulse = impulse_scalar * collision_normal

        # Aplicar impulso
        if not body1['is_kinematic']:
            body1['velocity'] += impulse / body1['mass']
        if not body2['is_kinematic']:
            body2['velocity'] -= impulse / body2['mass']

    def raycast(self, origin: np.ndarray, direction: np.ndarray,
                max_distance: float = 100.0) -> Optional[Tuple[str, float, np.ndarray]]:
        """
        Lanza un rayo y detecta el primer objeto que intersecta

        Returns:
            Tupla de (body_id, distancia, punto_intersección) o None
        """
        direction = direction / np.linalg.norm(direction)

        closest_hit = None
        closest_distance = max_distance

        for body_id, body in self.rigid_bodies.items():
            # Intersección rayo-esfera simplificada
            to_sphere = body['position'] - origin
            projection = np.dot(to_sphere, direction)

            if projection < 0:
                continue

            closest_point = origin + direction * projection
            distance_to_center = np.linalg.norm(body['position'] - closest_point)

            sphere_radius = 0.5  # Radio por defecto

            if distance_to_center < sphere_radius:
                hit_distance = projection - np.sqrt(sphere_radius**2 - distance_to_center**2)

                if hit_distance < closest_distance:
                    closest_distance = hit_distance
                    hit_point = origin + direction * hit_distance
                    closest_hit = (body_id, hit_distance, hit_point)

        return closest_hit


class NetworkingLayer:
    """
    Capa de networking para entornos multi-agente
    """

    def __init__(self):
        self.agents: Dict[str, Dict] = {}
        self.message_queue: List[Dict] = []
        print("🌐 NetworkingLayer inicializado")

    def register_agent(self, agent_id: str, agent_data: Dict):
        """Registra un nuevo agente en la red"""
        self.agents[agent_id] = agent_data
        print(f"👤 Agente registrado: {agent_id}")

    def send_message(self, from_agent: str, to_agent: str, message_type: str, data: Any):
        """Envía un mensaje entre agentes"""
        message = {
            'from': from_agent,
            'to': to_agent,
            'type': message_type,
            'data': data,
            'timestamp': time.time()
        }
        self.message_queue.append(message)

    def broadcast_message(self, from_agent: str, message_type: str, data: Any):
        """Envía un mensaje a todos los agentes"""
        for agent_id in self.agents.keys():
            if agent_id != from_agent:
                self.send_message(from_agent, agent_id, message_type, data)

    def get_messages(self, agent_id: str) -> List[Dict]:
        """Obtiene los mensajes pendientes para un agente"""
        messages = [msg for msg in self.message_queue if msg['to'] == agent_id]
        # Limpiar mensajes procesados
        self.message_queue = [msg for msg in self.message_queue if msg['to'] != agent_id]
        return messages

    def sync_state(self, agent_id: str, state: Dict):
        """Sincroniza el estado de un agente con los demás"""
        if agent_id in self.agents:
            self.agents[agent_id].update(state)
            self.broadcast_message(agent_id, 'state_update', state)


class StateManager:
    """
    Gestor de estados del entorno
    """

    def __init__(self):
        self.global_state: Dict[str, Any] = {}
        self.agent_states: Dict[str, AgentState] = {}
        self.episode_stats: Dict[str, float] = {
            'total_reward': 0.0,
            'steps': 0,
            'collisions': 0,
            'goals_reached': 0
        }
        print("📊 StateManager inicializado")

    def update_agent_state(self, agent_id: str, state: AgentState):
        """Actualiza el estado de un agente"""
        self.agent_states[agent_id] = state

    def get_observation(self, agent_id: str) -> np.ndarray:
        """Obtiene la observación del entorno para un agente"""
        if agent_id not in self.agent_states:
            return np.zeros(16)  # Vector de estado por defecto

        state = self.agent_states[agent_id]
        return state.to_array()

    def reset_episode_stats(self):
        """Resetea las estadísticas del episodio"""
        self.episode_stats = {
            'total_reward': 0.0,
            'steps': 0,
            'collisions': 0,
            'goals_reached': 0
        }


class OpenSimIntegration:
    """
    Integración específica con OpenSimulator
    """

    def __init__(self, host: str = "localhost", port: int = 9000):
        self.host = host
        self.port = port
        self.connected = False
        self.region_name = "WoldVirtual"
        print(f"🌍 OpenSimIntegration configurado para {host}:{port}")

    def connect(self) -> bool:
        """Conecta con el servidor OpenSim"""
        print(f"🔌 Conectando a OpenSim...")
        # Aquí iría la lógica de conexión real con OpenSim
        self.connected = True
        print("✅ Conectado a OpenSim")
        return True

    def create_region(self, name: str, size: Tuple[int, int] = (256, 256)) -> bool:
        """Crea una región en OpenSim"""
        if not self.connected:
            return False

        print(f"🗺️ Creando región: {name} ({size[0]}x{size[1]})")
        self.region_name = name
        return True

    def spawn_avatar(self, avatar_id: str, position: Tuple[float, float, float]) -> bool:
        """Genera un avatar en el mundo"""
        if not self.connected:
            return False

        print(f"👤 Spawneando avatar {avatar_id} en {position}")
        return True

    def create_object(self, object_type: str, position: Tuple[float, float, float],
                      properties: Optional[Dict] = None) -> Optional[str]:
        """Crea un objeto en el mundo"""
        if not self.connected:
            return None

        import uuid
        object_id = str(uuid.uuid4())
        print(f"🎲 Objeto creado: {object_type} ({object_id})")
        return object_id


class MetaverseEnvironment:
    """
    Entorno principal de metaverso para RL
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()

        # Componentes del entorno
        self.physics = PhysicsEngine(self.config.physics_engine, self.config.physics_timestep)
        self.networking = NetworkingLayer()
        self.state_manager = StateManager()

        # OpenSim (opcional)
        self.opensim = None
        if self.config.opensim_enabled:
            self.opensim = OpenSimIntegration(self.config.opensim_host, self.config.opensim_port)
            self.opensim.connect()

        # Estado del episodio
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

        # Objetivo y agente
        self.agent_id = "agent_0"
        self.goal_position = np.array([50.0, 0.0, 50.0])

        print(f"🌟 MetaverseEnvironment '{self.config.name}' inicializado")

    def reset(self) -> np.ndarray:
        """Resetea el entorno al estado inicial"""
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

        # Crear agente físico
        initial_position = np.array([0.0, 1.0, 0.0])
        self.physics.create_rigid_body(self.agent_id, mass=70.0, position=initial_position)

        # Crear estado del agente
        agent_state = AgentState(position=initial_position)
        self.state_manager.update_agent_state(self.agent_id, agent_state)

        # Resetear estadísticas
        self.state_manager.reset_episode_stats()

        print("🔄 Entorno reseteado")

        return self.state_manager.get_observation(self.agent_id)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Ejecuta una acción en el entorno

        Args:
            action: Vector de acción (ej: [forward, turn, jump])

        Returns:
            observation, reward, done, info
        """
        # Aplicar acción al agente
        force = np.array([action[0], 0.0, action[1]]) * 100.0  # Fuerza de movimiento
        self.physics.apply_force(self.agent_id, force)

        # Si hay salto
        if len(action) > 2 and action[2] > 0.5:
            jump_force = np.array([0.0, 300.0, 0.0])
            self.physics.apply_force(self.agent_id, jump_force)

        # Avanzar simulación física
        self.physics.step()

        # Actualizar estado del agente
        body = self.physics.rigid_bodies[self.agent_id]
        agent_state = AgentState(
            position=body['position'],
            velocity=body['velocity']
        )
        self.state_manager.update_agent_state(self.agent_id, agent_state)

        # Calcular recompensa
        reward = self._calculate_reward(agent_state)

        # Verificar condiciones de terminación
        self.done = self._check_done(agent_state)

        # Incrementar contador
        self.current_step += 1
        self.episode_reward += reward

        # Info adicional
        info = {
            'episode_step': self.current_step,
            'episode_reward': self.episode_reward,
            'distance_to_goal': np.linalg.norm(agent_state.position - self.goal_position)
        }

        observation = self.state_manager.get_observation(self.agent_id)

        return observation, reward, self.done, info

    def _calculate_reward(self, agent_state: AgentState) -> float:
        """Calcula la recompensa del paso actual"""
        reward = self.config.step_penalty

        # Distancia al objetivo
        distance_to_goal = np.linalg.norm(agent_state.position - self.goal_position)

        # Recompensa por acercarse al objetivo
        reward -= distance_to_goal * 0.01

        # Recompensa por alcanzar el objetivo
        if distance_to_goal < 2.0:
            reward += self.config.goal_reward
            self.state_manager.episode_stats['goals_reached'] += 1

        # Penalización por colisión
        if self.physics.rigid_bodies[self.agent_id]['colliding']:
            reward += self.config.collision_penalty
            self.state_manager.episode_stats['collisions'] += 1

        return reward

    def _check_done(self, agent_state: AgentState) -> bool:
        """Verifica si el episodio ha terminado"""
        # Límite de pasos
        if self.current_step >= self.config.max_episode_steps:
            return True

        # Objetivo alcanzado
        distance_to_goal = np.linalg.norm(agent_state.position - self.goal_position)
        if distance_to_goal < 2.0:
            return True

        # Fuera de límites
        if np.any(np.abs(agent_state.position) > self.config.world_bounds):
            return True

        return False

    def render(self, mode: str = "human"):
        """Renderiza el entorno (placeholder)"""
        if mode == "human":
            print(f"Step: {self.current_step}, Reward: {self.episode_reward:.2f}")


print("✅ Módulo de entornos de metaverso cargado")
