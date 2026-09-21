"""
Neurona Especializada 7: Motor de Física y Colisiones 3D
Simulación física realista compatible con OpenSim/Bullet Physics
WoldVirtual3DlucIA v0.6.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class PhysicsShape(Enum):
    """Tipos de formas físicas"""
    SPHERE = "sphere"
    BOX = "box"
    CAPSULE = "capsule"
    CYLINDER = "cylinder"
    MESH = "mesh"
    CONVEX_HULL = "convex_hull"


class PhysicsType(Enum):
    """Tipos de cuerpos físicos"""
    STATIC = "static"  # No se mueve
    DYNAMIC = "dynamic"  # Afectado por fuerzas
    KINEMATIC = "kinematic"  # Controlado por código


@dataclass
class PhysicsBody:
    """Cuerpo físico en la simulación"""
    body_id: str
    shape: PhysicsShape
    physics_type: PhysicsType
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.3  # Bounciness
    dimensions: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    enabled: bool = True


@dataclass
class CollisionInfo:
    """Información de colisión"""
    body_a: str
    body_b: str
    contact_point: np.ndarray
    contact_normal: np.ndarray
    penetration_depth: float
    timestamp: float


class PhysicsEngine:
    """
    Motor de física 3D
    Compatible con Bullet Physics (usado en OpenSimulator)
    """

    def __init__(self, gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)):
        self.gravity = np.array(gravity)
        self.bodies = {}
        self.collisions = []
        self.time_step = 1.0 / 60.0
        self.damping_linear = 0.04
        self.damping_angular = 0.05

    def add_body(self, body: PhysicsBody):
        """Añade un cuerpo a la simulación"""
        self.bodies[body.body_id] = body

    def remove_body(self, body_id: str):
        """Elimina un cuerpo de la simulación"""
        if body_id in self.bodies:
            del self.bodies[body_id]

    def step(self, delta_time: Optional[float] = None):
        """
        Avanza la simulación un paso

        Args:
            delta_time: Tiempo a simular (usa time_step por defecto)
        """
        dt = delta_time or self.time_step

        # Limpiar colisiones previas
        self.collisions.clear()

        # Actualizar todos los cuerpos dinámicos
        for body in self.bodies.values():
            if body.physics_type == PhysicsType.DYNAMIC and body.enabled:
                self._integrate_body(body, dt)

        # Detección y resolución de colisiones
        self._detect_collisions()
        self._resolve_collisions(dt)

        # Aplicar damping
        for body in self.bodies.values():
            if body.physics_type == PhysicsType.DYNAMIC:
                body.velocity *= (1.0 - self.damping_linear)
                body.angular_velocity *= (1.0 - self.damping_angular)

    def _integrate_body(self, body: PhysicsBody, dt: float):
        """
        Integra físicas de un cuerpo (método Euler)

        Args:
            body: Cuerpo a integrar
            dt: Delta time
        """
        # Aplicar gravedad
        acceleration = self.gravity

        # Actualizar velocidad
        body.velocity += acceleration * dt

        # Actualizar posición
        body.position += body.velocity * dt

        # Actualizar rotación (simplificado)
        if np.linalg.norm(body.angular_velocity) > 1e-6:
            angle = np.linalg.norm(body.angular_velocity) * dt
            axis = body.angular_velocity / np.linalg.norm(body.angular_velocity)

            # Crear quaternion de rotación
            half_angle = angle * 0.5
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)

            rotation_quat = np.array([
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
                cos_half
            ])

            # Multiplicar quaternions
            body.rotation = self._multiply_quaternions(body.rotation, rotation_quat)
            body.rotation /= np.linalg.norm(body.rotation)

    def _multiply_quaternions(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiplica dos quaternions"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

    def _detect_collisions(self):
        """Detecta colisiones entre cuerpos"""
        body_list = list(self.bodies.values())

        for i in range(len(body_list)):
            for j in range(i + 1, len(body_list)):
                body_a = body_list[i]
                body_b = body_list[j]

                # Saltar si ambos son estáticos
                if (body_a.physics_type == PhysicsType.STATIC and
                        body_b.physics_type == PhysicsType.STATIC):
                    continue

                # Detectar colisión
                collision = self._check_collision(body_a, body_b)
                if collision:
                    self.collisions.append(collision)

    def _check_collision(
        self,
        body_a: PhysicsBody,
        body_b: PhysicsBody
    ) -> Optional[CollisionInfo]:
        """
        Comprueba colisión entre dos cuerpos

        Returns:
            CollisionInfo si hay colisión, None si no
        """
        # Implementación simplificada para esferas y cajas
        if body_a.shape == PhysicsShape.SPHERE and body_b.shape == PhysicsShape.SPHERE:
            return self._sphere_sphere_collision(body_a, body_b)
        elif body_a.shape == PhysicsShape.BOX and body_b.shape == PhysicsShape.BOX:
            return self._box_box_collision(body_a, body_b)
        elif body_a.shape == PhysicsShape.SPHERE and body_b.shape == PhysicsShape.BOX:
            return self._sphere_box_collision(body_a, body_b)
        elif body_a.shape == PhysicsShape.BOX and body_b.shape == PhysicsShape.SPHERE:
            collision = self._sphere_box_collision(body_b, body_a)
            if collision:
                # Invertir normal
                collision.body_a, collision.body_b = collision.body_b, collision.body_a
                collision.contact_normal = -collision.contact_normal
            return collision

        return None

    def _sphere_sphere_collision(
        self,
        sphere_a: PhysicsBody,
        sphere_b: PhysicsBody
    ) -> Optional[CollisionInfo]:
        """Detecta colisión esfera-esfera"""
        radius_a = sphere_a.dimensions[0] / 2.0
        radius_b = sphere_b.dimensions[0] / 2.0

        # Vector entre centros
        delta = sphere_b.position - sphere_a.position
        distance = np.linalg.norm(delta)

        # Comprobar colisión
        if distance < (radius_a + radius_b):
            # Hay colisión
            normal = delta / (distance + 1e-10)
            penetration = (radius_a + radius_b) - distance
            contact_point = sphere_a.position + normal * radius_a

            return CollisionInfo(
                body_a=sphere_a.body_id,
                body_b=sphere_b.body_id,
                contact_point=contact_point,
                contact_normal=normal,
                penetration_depth=penetration,
                timestamp=0.0
            )

        return None

    def _box_box_collision(
        self,
        box_a: PhysicsBody,
        box_b: PhysicsBody
    ) -> Optional[CollisionInfo]:
        """Detecta colisión caja-caja (AABB simplificado)"""
        # Usar AABB (Axis-Aligned Bounding Box)
        half_size_a = np.array(box_a.dimensions) / 2.0
        half_size_b = np.array(box_b.dimensions) / 2.0

        min_a = box_a.position - half_size_a
        max_a = box_a.position + half_size_a
        min_b = box_b.position - half_size_b
        max_b = box_b.position + half_size_b

        # Comprobar solapamiento en cada eje
        overlap_x = min(max_a[0], max_b[0]) - max(min_a[0], min_b[0])
        overlap_y = min(max_a[1], max_b[1]) - max(min_a[1], min_b[1])
        overlap_z = min(max_a[2], max_b[2]) - max(min_a[2], min_b[2])

        if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
            # Hay colisión, encontrar eje de menor penetración
            penetrations = [overlap_x, overlap_y, overlap_z]
            min_pen_axis = np.argmin(penetrations)
            penetration = penetrations[min_pen_axis]

            # Normal en el eje de menor penetración
            normal = np.zeros(3)
            normal[min_pen_axis] = 1.0 if box_b.position[min_pen_axis] > box_a.position[min_pen_axis] else -1.0

            # Punto de contacto aproximado
            contact_point = (box_a.position + box_b.position) / 2.0

            return CollisionInfo(
                body_a=box_a.body_id,
                body_b=box_b.body_id,
                contact_point=contact_point,
                contact_normal=normal,
                penetration_depth=penetration,
                timestamp=0.0
            )

        return None

    def _sphere_box_collision(
        self,
        sphere: PhysicsBody,
        box: PhysicsBody
    ) -> Optional[CollisionInfo]:
        """Detecta colisión esfera-caja"""
        radius = sphere.dimensions[0] / 2.0
        half_size = np.array(box.dimensions) / 2.0

        # Punto más cercano de la caja a la esfera
        closest_point = np.clip(
            sphere.position,
            box.position - half_size,
            box.position + half_size
        )

        # Vector y distancia
        delta = sphere.position - closest_point
        distance = np.linalg.norm(delta)

        if distance < radius:
            # Hay colisión
            if distance > 1e-6:
                normal = delta / distance
            else:
                # Esfera dentro de caja, usar eje de mayor penetración
                penetrations = np.abs(sphere.position - box.position) - half_size
                max_pen_axis = np.argmax(penetrations)
                normal = np.zeros(3)
                normal[max_pen_axis] = 1.0 if sphere.position[max_pen_axis] > box.position[max_pen_axis] else -1.0

            penetration = radius - distance

            return CollisionInfo(
                body_a=sphere.body_id,
                body_b=box.body_id,
                contact_point=closest_point,
                contact_normal=normal,
                penetration_depth=penetration,
                timestamp=0.0
            )

        return None

    def _resolve_collisions(self, dt: float):
        """Resuelve todas las colisiones detectadas"""
        for collision in self.collisions:
            body_a = self.bodies.get(collision.body_a)
            body_b = self.bodies.get(collision.body_b)

            if not body_a or not body_b:
                continue

            # Separar cuerpos
            self._separate_bodies(body_a, body_b, collision)

            # Resolver impulso
            self._apply_collision_impulse(body_a, body_b, collision)

    def _separate_bodies(
        self,
        body_a: PhysicsBody,
        body_b: PhysicsBody,
        collision: CollisionInfo
    ):
        """Separa cuerpos que se están penetrando"""
        penetration = collision.penetration_depth
        normal = collision.contact_normal

        # Calcular masas inversas
        inv_mass_a = 0.0 if body_a.physics_type == PhysicsType.STATIC else 1.0 / body_a.mass
        inv_mass_b = 0.0 if body_b.physics_type == PhysicsType.STATIC else 1.0 / body_b.mass
        total_inv_mass = inv_mass_a + inv_mass_b

        if total_inv_mass > 1e-6:
            # Mover cuerpos proporcionalmente a sus masas
            correction = normal * penetration
            body_a.position -= correction * (inv_mass_a / total_inv_mass)
            body_b.position += correction * (inv_mass_b / total_inv_mass)

    def _apply_collision_impulse(
        self,
        body_a: PhysicsBody,
        body_b: PhysicsBody,
        collision: CollisionInfo
    ):
        """Aplica impulso de colisión para resolver velocidades"""
        # Velocidades relativas
        rel_velocity = body_b.velocity - body_a.velocity

        # Velocidad a lo largo de la normal
        vel_along_normal = np.dot(rel_velocity, collision.contact_normal)

        # No resolver si se están separando
        if vel_along_normal > 0:
            return

        # Coeficiente de restitución (promedio)
        restitution = (body_a.restitution + body_b.restitution) / 2.0

        # Calcular magnitud de impulso
        inv_mass_a = 0.0 if body_a.physics_type == PhysicsType.STATIC else 1.0 / body_a.mass
        inv_mass_b = 0.0 if body_b.physics_type == PhysicsType.STATIC else 1.0 / body_b.mass

        impulse_magnitude = -(1.0 + restitution) * vel_along_normal
        impulse_magnitude /= (inv_mass_a + inv_mass_b)

        # Aplicar impulso
        impulse = collision.contact_normal * impulse_magnitude

        if body_a.physics_type == PhysicsType.DYNAMIC:
            body_a.velocity -= impulse * inv_mass_a

        if body_b.physics_type == PhysicsType.DYNAMIC:
            body_b.velocity += impulse * inv_mass_b

    def apply_force(self, body_id: str, force: Tuple[float, float, float]):
        """Aplica fuerza a un cuerpo"""
        body = self.bodies.get(body_id)
        if body and body.physics_type == PhysicsType.DYNAMIC:
            acceleration = np.array(force) / body.mass
            body.velocity += acceleration * self.time_step

    def apply_impulse(
        self,
        body_id: str,
        impulse: Tuple[float, float, float],
        point: Optional[Tuple[float, float, float]] = None
    ):
        """Aplica impulso instantáneo a un cuerpo"""
        body = self.bodies.get(body_id)
        if body and body.physics_type == PhysicsType.DYNAMIC:
            body.velocity += np.array(impulse) / body.mass

            # Si hay punto de aplicación, calcular torque
            if point is not None:
                r = np.array(point) - body.position
                torque = np.cross(r, np.array(impulse))
                # Simplificado: asumir momento de inercia unitario
                body.angular_velocity += torque / body.mass


class CollisionDetector:
    """Detector de colisiones optimizado"""

    def __init__(self):
        self.spatial_hash = {}
        self.cell_size = 10.0

    def update_spatial_hash(self, bodies: Dict[str, PhysicsBody]):
        """Actualiza hash espacial para detección eficiente"""
        self.spatial_hash.clear()

        for body_id, body in bodies.items():
            cell = self._get_cell(body.position)

            if cell not in self.spatial_hash:
                self.spatial_hash[cell] = []

            self.spatial_hash[cell].append(body_id)

    def _get_cell(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Obtiene celda de hash espacial"""
        return (
            int(position[0] / self.cell_size),
            int(position[1] / self.cell_size),
            int(position[2] / self.cell_size)
        )

    def get_nearby_bodies(
        self,
        position: np.ndarray,
        bodies: Dict[str, PhysicsBody]
    ) -> List[str]:
        """Obtiene cuerpos cercanos a una posición"""
        cell = self._get_cell(position)
        nearby = []

        # Comprobar celda y adyacentes
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    check_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    if check_cell in self.spatial_hash:
                        nearby.extend(self.spatial_hash[check_cell])

        return nearby


class DynamicsSimulator:
    """Simulador de dinámicas complejas"""

    def __init__(self):
        self.physics_engine = PhysicsEngine()
        self.collision_detector = CollisionDetector()

    def simulate_ragdoll(
        self,
        skeleton: Dict,
        initial_impulse: Tuple[float, float, float]
    ) -> Dict:
        """
        Simula física de ragdoll para avatar

        Args:
            skeleton: Datos del esqueleto
            initial_impulse: Impulso inicial

        Returns:
            Transformaciones simuladas
        """
        # Crear cuerpos físicos para cada hueso
        ragdoll_bodies = {}

        for bone_name, bone_data in skeleton.items():
            body = PhysicsBody(
                body_id=bone_name,
                shape=PhysicsShape.CAPSULE,
                physics_type=PhysicsType.DYNAMIC,
                position=np.array(bone_data["position"]),
                mass=1.0
            )

            self.physics_engine.add_body(body)
            ragdoll_bodies[bone_name] = body

        # Aplicar impulso inicial
        if "pelvis" in ragdoll_bodies:
            self.physics_engine.apply_impulse("pelvis", initial_impulse)

        # Simular varios pasos
        transforms = {}
        for _ in range(60):  # 1 segundo a 60 FPS
            self.physics_engine.step()

        # Recoger transformaciones finales
        for bone_name, body in ragdoll_bodies.items():
            transforms[bone_name] = {
                "position": body.position.tolist(),
                "rotation": body.rotation.tolist()
            }

        return transforms
