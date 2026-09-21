"""
Neurona Especializada 8: Sistema de IA para NPCs y Comportamientos
Inteligencia artificial avanzada para personajes no jugadores
WoldVirtual3DlucIA v0.6.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import time


class BehaviorState(Enum):
    """Estados de comportamiento de NPC"""
    IDLE = "idle"
    PATROL = "patrol"
    CHASE = "chase"
    FLEE = "flee"
    INTERACT = "interact"
    ATTACK = "attack"
    SEARCH = "search"
    FOLLOW = "follow"


class EmotionalState(Enum):
    """Estados emocionales"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    FEAR = "fear"
    CURIOUS = "curious"
    BORED = "bored"


@dataclass
class NPCProfile:
    """Perfil de personalidad de NPC"""
    npc_id: str
    name: str
    aggression: float = 0.5  # 0-1
    courage: float = 0.5  # 0-1
    sociability: float = 0.5  # 0-1
    intelligence: float = 0.5  # 0-1
    curiosity: float = 0.5  # 0-1
    memory_span: int = 10  # Cuántos eventos recuerda


@dataclass
class NPCState:
    """Estado actual de un NPC"""
    npc_id: str
    position: np.ndarray
    rotation: float  # Ángulo en radianes
    current_behavior: BehaviorState = BehaviorState.IDLE
    emotional_state: EmotionalState = EmotionalState.NEUTRAL
    target: Optional[str] = None
    health: float = 100.0
    energy: float = 100.0
    memory: List[Dict] = field(default_factory=list)


class AIController:
    """
    Controlador de IA principal para NPCs
    Implementa máquina de estados finita y árboles de decisión
    """

    def __init__(self, profile: NPCProfile):
        self.profile = profile
        self.state = NPCState(
            npc_id=profile.npc_id,
            position=np.array([0.0, 0.0, 0.0]),
            rotation=0.0
        )
        self.behavior_tree = self._build_behavior_tree()
        self.perception_radius = 20.0
        self.decision_cooldown = 0.5
        self.last_decision_time = 0.0

    def _build_behavior_tree(self) -> Dict:
        """Construye árbol de decisiones"""
        return {
            "root": {
                "type": "selector",
                "children": [
                    "check_danger",
                    "check_target",
                    "check_energy",
                    "default_behavior"
                ]
            },
            "check_danger": {
                "type": "condition",
                "condition": self._is_in_danger,
                "on_true": "flee_behavior",
                "on_false": None
            },
            "check_target": {
                "type": "condition",
                "condition": self._has_target,
                "on_true": "chase_behavior",
                "on_false": None
            },
            "check_energy": {
                "type": "condition",
                "condition": self._is_low_energy,
                "on_true": "rest_behavior",
                "on_false": None
            },
            "default_behavior": {
                "type": "action",
                "action": self._idle_or_patrol
            }
        }

    def update(self, delta_time: float, world_state: Dict) -> Dict:
        """
        Actualiza IA del NPC

        Args:
            delta_time: Tiempo transcurrido
            world_state: Estado del mundo (otros NPCs, jugadores, objetos)

        Returns:
            Acciones a realizar
        """
        current_time = time.time()

        # Actualizar percepción
        perceived_entities = self._perceive_environment(world_state)

        # Tomar decisiones periódicamente
        if current_time - self.last_decision_time > self.decision_cooldown:
            self._make_decision(perceived_entities)
            self.last_decision_time = current_time

        # Ejecutar comportamiento actual
        actions = self._execute_behavior(delta_time, perceived_entities)

        # Actualizar estado emocional
        self._update_emotional_state(perceived_entities)

        # Decrementar energía
        self.state.energy = max(0, self.state.energy - delta_time * 0.5)

        return actions

    def _perceive_environment(self, world_state: Dict) -> List[Dict]:
        """
        Percibe entidades cercanas

        Args:
            world_state: Estado del mundo

        Returns:
            Lista de entidades percibidas
        """
        perceived = []

        # Percibir otros agentes
        for agent_id, agent_data in world_state.get("agents", {}).items():
            if agent_id == self.state.npc_id:
                continue

            agent_pos = np.array(agent_data.get("position", [0, 0, 0]))
            distance = np.linalg.norm(agent_pos - self.state.position)

            if distance <= self.perception_radius:
                perceived.append({
                    "type": "agent",
                    "id": agent_id,
                    "position": agent_pos,
                    "distance": distance,
                    "data": agent_data
                })

        # Percibir objetos de interés
        for obj_id, obj_data in world_state.get("objects", {}).items():
            obj_pos = np.array(obj_data.get("position", [0, 0, 0]))
            distance = np.linalg.norm(obj_pos - self.state.position)

            if distance <= self.perception_radius:
                perceived.append({
                    "type": "object",
                    "id": obj_id,
                    "position": obj_pos,
                    "distance": distance,
                    "data": obj_data
                })

        return perceived

    def _make_decision(self, perceived_entities: List[Dict]):
        """Toma decisión basada en árbol de comportamiento"""
        # Evaluar árbol de decisión
        decision = self._evaluate_behavior_tree("root", perceived_entities)

        if decision:
            self.state.current_behavior = decision

    def _evaluate_behavior_tree(self, node_name: str, context: List[Dict]) -> Optional[BehaviorState]:
        """Evalúa nodo del árbol de comportamiento"""
        node = self.behavior_tree.get(node_name)

        if not node:
            return None

        node_type = node.get("type")

        if node_type == "selector":
            # Selector: prueba hijos hasta que uno tenga éxito
            for child_name in node.get("children", []):
                result = self._evaluate_behavior_tree(child_name, context)
                if result:
                    return result

        elif node_type == "condition":
            condition_func = node.get("condition")
            if condition_func and condition_func(context):
                return self._evaluate_behavior_tree(node.get("on_true"), context)
            elif node.get("on_false"):
                return self._evaluate_behavior_tree(node.get("on_false"), context)

        elif node_type == "action":
            action_func = node.get("action")
            if action_func:
                return action_func(context)

        return None

    def _is_in_danger(self, context: List[Dict]) -> bool:
        """Detecta si hay peligro"""
        for entity in context:
            if entity.get("type") == "agent":
                agent_data = entity.get("data", {})
                if agent_data.get("hostile", False) and entity["distance"] < 10.0:
                    # Evaluar según coraje
                    return random.random() > self.profile.courage
        return False

    def _has_target(self, context: List[Dict]) -> bool:
        """Comprueba si hay un objetivo"""
        return self.state.target is not None

    def _is_low_energy(self, context: List[Dict]) -> bool:
        """Comprueba si la energía es baja"""
        return self.state.energy < 30.0

    def _idle_or_patrol(self, context: List[Dict]) -> BehaviorState:
        """Decide entre idle y patrullar"""
        if random.random() < 0.3:
            return BehaviorState.PATROL
        return BehaviorState.IDLE

    def _execute_behavior(self, delta_time: float, context: List[Dict]) -> Dict:
        """Ejecuta el comportamiento actual"""
        behavior = self.state.current_behavior

        if behavior == BehaviorState.IDLE:
            return self._behavior_idle(delta_time)
        elif behavior == BehaviorState.PATROL:
            return self._behavior_patrol(delta_time)
        elif behavior == BehaviorState.CHASE:
            return self._behavior_chase(delta_time, context)
        elif behavior == BehaviorState.FLEE:
            return self._behavior_flee(delta_time, context)
        elif behavior == BehaviorState.INTERACT:
            return self._behavior_interact(delta_time, context)

        return {"action": "none"}

    def _behavior_idle(self, delta_time: float) -> Dict:
        """Comportamiento idle"""
        # Ocasionalmente mirar alrededor
        if random.random() < 0.01:
            self.state.rotation += random.uniform(-0.5, 0.5)

        return {
            "action": "idle",
            "animation": "idle",
            "position": self.state.position,
            "rotation": self.state.rotation
        }

    def _behavior_patrol(self, delta_time: float) -> Dict:
        """Comportamiento de patrulla"""
        # Movimiento aleatorio
        speed = 2.0 * delta_time

        # Cambiar dirección ocasionalmente
        if random.random() < 0.05:
            self.state.rotation += random.uniform(-1.0, 1.0)

        # Mover hacia adelante
        direction = np.array([
            np.cos(self.state.rotation),
            0.0,
            np.sin(self.state.rotation)
        ])

        self.state.position += direction * speed

        return {
            "action": "move",
            "animation": "walk",
            "position": self.state.position,
            "rotation": self.state.rotation,
            "velocity": direction * speed
        }

    def _behavior_chase(self, delta_time: float, context: List[Dict]) -> Dict:
        """Comportamiento de persecución"""
        if not self.state.target:
            self.state.current_behavior = BehaviorState.IDLE
            return {"action": "none"}

        # Buscar objetivo en contexto
        target_entity = None
        for entity in context:
            if entity["id"] == self.state.target:
                target_entity = entity
                break

        if not target_entity:
            self.state.target = None
            return {"action": "none"}

        # Moverse hacia objetivo
        target_pos = target_entity["position"]
        direction = target_pos - self.state.position
        distance = np.linalg.norm(direction)

        if distance > 0.1:
            direction = direction / distance
            speed = 4.0 * delta_time
            self.state.position += direction * speed

            # Actualizar rotación
            self.state.rotation = np.arctan2(direction[2], direction[0])

        return {
            "action": "chase",
            "animation": "run",
            "position": self.state.position,
            "rotation": self.state.rotation
        }

    def _behavior_flee(self, delta_time: float, context: List[Dict]) -> Dict:
        """Comportamiento de huida"""
        # Encontrar amenaza más cercana
        closest_threat = None
        min_distance = float('inf')

        for entity in context:
            if entity.get("type") == "agent" and entity.get("data", {}).get("hostile"):
                if entity["distance"] < min_distance:
                    min_distance = entity["distance"]
                    closest_threat = entity

        if closest_threat:
            # Huir en dirección opuesta
            threat_pos = closest_threat["position"]
            direction = self.state.position - threat_pos
            distance = np.linalg.norm(direction)

            if distance > 0.1:
                direction = direction / distance
                speed = 5.0 * delta_time  # Más rápido que caminar
                self.state.position += direction * speed

                self.state.rotation = np.arctan2(direction[2], direction[0])

        return {
            "action": "flee",
            "animation": "run",
            "position": self.state.position,
            "rotation": self.state.rotation
        }

    def _behavior_interact(self, delta_time: float, context: List[Dict]) -> Dict:
        """Comportamiento de interacción"""
        return {
            "action": "interact",
            "animation": "talk",
            "position": self.state.position,
            "rotation": self.state.rotation
        }

    def _update_emotional_state(self, context: List[Dict]):
        """Actualiza estado emocional"""
        # Contar amenazas y aliados
        threats = sum(1 for e in context if e.get("data", {}).get("hostile", False))
        friends = sum(1 for e in context if e.get("data", {}).get("friendly", False))

        # Determinar emoción
        if threats > 0 and self.state.health < 50:
            self.state.emotional_state = EmotionalState.FEAR
        elif threats > 0:
            self.state.emotional_state = EmotionalState.ANGRY
        elif friends > 0:
            self.state.emotional_state = EmotionalState.HAPPY
        elif len(context) == 0:
            self.state.emotional_state = EmotionalState.BORED
        else:
            self.state.emotional_state = EmotionalState.NEUTRAL

    def add_memory(self, event: Dict):
        """Añade evento a la memoria"""
        self.state.memory.append({
            "timestamp": time.time(),
            "event": event
        })

        # Limitar tamaño de memoria
        if len(self.state.memory) > self.profile.memory_span:
            self.state.memory.pop(0)


class NPCBehaviorEngine:
    """Motor de comportamiento para múltiples NPCs"""

    def __init__(self):
        self.npcs = {}
        self.ai_controllers = {}

    def create_npc(
        self,
        npc_id: str,
        name: str,
        position: Tuple[float, float, float],
        profile: Optional[NPCProfile] = None
    ) -> str:
        """
        Crea un nuevo NPC

        Args:
            npc_id: ID único
            name: Nombre del NPC
            position: Posición inicial
            profile: Perfil de personalidad

        Returns:
            ID del NPC creado
        """
        if profile is None:
            profile = NPCProfile(
                npc_id=npc_id,
                name=name,
                aggression=random.uniform(0.2, 0.8),
                courage=random.uniform(0.3, 0.9),
                sociability=random.uniform(0.3, 0.9),
                intelligence=random.uniform(0.4, 0.9),
                curiosity=random.uniform(0.3, 0.8)
            )

        controller = AIController(profile)
        controller.state.position = np.array(position)

        self.npcs[npc_id] = {
            "profile": profile,
            "controller": controller
        }
        self.ai_controllers[npc_id] = controller

        return npc_id

    def update_all(self, delta_time: float, world_state: Dict) -> Dict[str, Dict]:
        """
        Actualiza todos los NPCs

        Args:
            delta_time: Tiempo transcurrido
            world_state: Estado del mundo

        Returns:
            Acciones de cada NPC
        """
        actions = {}

        for npc_id, controller in self.ai_controllers.items():
            actions[npc_id] = controller.update(delta_time, world_state)

        return actions

    def get_npc_state(self, npc_id: str) -> Optional[NPCState]:
        """Obtiene estado de un NPC"""
        controller = self.ai_controllers.get(npc_id)
        return controller.state if controller else None


class PathfindingSystem:
    """
    Sistema de pathfinding usando A*
    Para navegación de NPCs en el mundo
    """

    def __init__(self, grid_size: Tuple[int, int], cell_size: float = 1.0):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.obstacle_map = np.zeros(grid_size, dtype=bool)

    def set_obstacle(self, x: int, y: int, is_obstacle: bool = True):
        """Marca celda como obstáculo"""
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            self.obstacle_map[y, x] = is_obstacle

    def find_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Encuentra camino usando A*

        Args:
            start: Posición inicial (x, y)
            goal: Posición objetivo (x, y)

        Returns:
            Lista de puntos del camino o None
        """
        # Convertir posiciones a celdas
        start_cell = self._world_to_grid(start)
        goal_cell = self._world_to_grid(goal)

        # Verificar validez
        if not self._is_valid_cell(start_cell) or not self._is_valid_cell(goal_cell):
            return None

        # A* algorithm
        open_set = {start_cell}
        came_from = {}

        g_score = {start_cell: 0}
        f_score = {start_cell: self._heuristic(start_cell, goal_cell)}

        while open_set:
            # Encontrar nodo con menor f_score
            current = min(open_set, key=lambda cell: f_score.get(cell, float('inf')))

            if current == goal_cell:
                # Reconstruir camino
                path = self._reconstruct_path(came_from, current)
                # Convertir a coordenadas del mundo
                return [self._grid_to_world(cell) for cell in path]

            open_set.remove(current)

            # Explorar vecinos
            for neighbor in self._get_neighbors(current):
                if not self._is_valid_cell(neighbor) or self.obstacle_map[neighbor[1], neighbor[0]]:
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_cell)

                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return None  # No se encontró camino

    def _world_to_grid(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convierte coordenadas del mundo a celda de grid"""
        return (
            int(pos[0] / self.cell_size),
            int(pos[1] / self.cell_size)
        )

    def _grid_to_world(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        """Convierte celda de grid a coordenadas del mundo"""
        return (
            (cell[0] + 0.5) * self.cell_size,
            (cell[1] + 0.5) * self.cell_size
        )

    def _is_valid_cell(self, cell: Tuple[int, int]) -> bool:
        """Verifica si celda es válida"""
        return (0 <= cell[0] < self.grid_size[0] and
                0 <= cell[1] < self.grid_size[1])

    def _get_neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Obtiene celdas vecinas (4-direcciones)"""
        x, y = cell
        return [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1)
        ]

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Heurística Manhattan para A*"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(
        self,
        came_from: Dict,
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruye camino desde came_from"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
