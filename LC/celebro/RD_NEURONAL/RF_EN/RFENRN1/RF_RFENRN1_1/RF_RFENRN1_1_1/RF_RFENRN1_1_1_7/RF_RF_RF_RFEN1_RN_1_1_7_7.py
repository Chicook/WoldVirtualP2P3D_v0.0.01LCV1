"""
Neurona 7: Simulación Física y Cloth (Physics Simulation)
Algoritmos avanzados para simulación de física de ropa, pelo y soft bodies
en avatares 3D de OpenSimulator.

Librerías: numpy, scipy
Técnicas: Mass-Spring Systems, Verlet Integration, Collision Detection, Constraints
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass, field
try:
    from scipy.spatial import cKDTree
except ImportError:
    pass  # dependencia pesada opcional


logger = logging.getLogger(__name__)


@dataclass
class Particle:
    """Partícula en el sistema de física."""
    position: np.ndarray
    previous_position: np.ndarray
    velocity: np.ndarray
    mass: float
    is_pinned: bool = False
    forces: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        """Conversión de listas a arrays."""
        if isinstance(self.position, list):
            self.position = np.array(self.position, dtype=np.float32)
        if isinstance(self.previous_position, list):
            self.previous_position = np.array(self.previous_position, dtype=np.float32)
        if isinstance(self.velocity, list):
            self.velocity = np.array(self.velocity, dtype=np.float32)


@dataclass
class Spring:
    """Spring constraint entre dos partículas."""
    particle_a: int
    particle_b: int
    rest_length: float
    stiffness: float = 0.8
    damping: float = 0.05


@dataclass
class ClothMesh:
    """Representación de una malla de tela."""
    particles: List[Particle]
    springs: List[Spring]
    faces: List[Tuple[int, int, int]]
    width: int
    height: int


class PhysicsSimulationNeuron:
    """
    Neurona especializada en simulación física para avatares 3D.

    Funcionalidades:
    - Simulación de ropa (cloth simulation)
    - Simulación de pelo y cabello
    - Soft body dynamics
    - Detección de colisiones
    - Sistema de constraints
    - Integración temporal (Verlet)
    - Wind y fuerzas externas
    - Self-collision handling
    """

    def __init__(self):
        """Inicializa la neurona de simulación física."""
        self.name = "PhysicsSimulationNeuron"
        self.version = "1.0.0"
        self.gravity = np.array([0.0, -9.81, 0.0])
        self.timestep = 1.0 / 60.0  # 60 FPS
        self.substeps = 5  # Substeps para mayor precisión
        logger.info(f"{self.name} v{self.version} inicializada")

    def process(self, avatar_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa y simula física para un avatar.

        Args:
            avatar_data: Diccionario con datos del avatar

        Returns:
            Diccionario con simulación física configurada
        """
        try:
            cloth_data = avatar_data.get('cloth', [])
            body_mesh = avatar_data.get('mesh', None)
            simulation_params = avatar_data.get('physics_params', {})

            if not cloth_data and not body_mesh:
                return {'error': 'No se encontraron datos para simular'}

            results = {}

            # Configurar parámetros de simulación
            self._configure_simulation(simulation_params)

            # Procesar elementos de ropa
            cloth_simulations = []
            for i, cloth_item in enumerate(cloth_data):
                logger.info(f"Configurando simulación de ropa {i}")

                cloth_mesh = self._create_cloth_mesh(cloth_item)

                # Configurar pins (puntos fijos)
                self._configure_pins(cloth_mesh, cloth_item.get('pins', []))

                # Simular algunos frames de muestra
                simulation_frames = self._simulate_cloth(cloth_mesh, num_frames=10)

                cloth_simulations.append({
                    'index': i,
                    'name': cloth_item.get('name', f'cloth_{i}'),
                    'particle_count': len(cloth_mesh.particles),
                    'spring_count': len(cloth_mesh.springs),
                    'sample_frames': simulation_frames,
                    'configuration': self._cloth_to_dict(cloth_mesh)
                })

            # Configurar colisionador del cuerpo
            body_collider = None
            if body_mesh:
                body_collider = self._create_body_collider(body_mesh)

            # Generar código de simulación optimizado
            simulation_code = self._generate_simulation_code()

            return {
                'success': True,
                'cloth_simulations': cloth_simulations,
                'body_collider': body_collider,
                'simulation_code': simulation_code,
                'physics_params': {
                    'gravity': self.gravity.tolist(),
                    'timestep': self.timestep,
                    'substeps': self.substeps
                }
            }

        except Exception as e:
            logger.error(f"Error en {self.name}: {e}")
            return {'error': str(e)}

    def _configure_simulation(self, params: Dict[str, Any]):
        """Configura los parámetros de simulación."""
        if 'gravity' in params:
            self.gravity = np.array(params['gravity'])
        if 'timestep' in params:
            self.timestep = params['timestep']
        if 'substeps' in params:
            self.substeps = params['substeps']

    def _create_cloth_mesh(self, cloth_item: Dict[str, Any]) -> ClothMesh:
        """
        Crea una malla de tela desde datos de entrada.

        Args:
            cloth_item: Datos del elemento de ropa

        Returns:
            ClothMesh configurado
        """
        width = cloth_item.get('width', 10)
        height = cloth_item.get('height', 10)
        spacing = cloth_item.get('spacing', 0.1)

        particles = []
        springs = []
        faces = []

        # Crear partículas en una cuadrícula
        for y in range(height):
            for x in range(width):
                pos = np.array([x * spacing, y * spacing, 0.0], dtype=np.float32)

                particle = Particle(
                    position=pos.copy(),
                    previous_position=pos.copy(),
                    velocity=np.zeros(3, dtype=np.float32),
                    mass=1.0
                )
                particles.append(particle)

        # Crear springs entre partículas
        # Springs estructurales (horizontal y vertical)
        for y in range(height):
            for x in range(width):
                idx = y * width + x

                # Spring horizontal
                if x < width - 1:
                    neighbor_idx = y * width + (x + 1)
                    spring = Spring(
                        particle_a=idx,
                        particle_b=neighbor_idx,
                        rest_length=spacing,
                        stiffness=0.9
                    )
                    springs.append(spring)

                # Spring vertical
                if y < height - 1:
                    neighbor_idx = (y + 1) * width + x
                    spring = Spring(
                        particle_a=idx,
                        particle_b=neighbor_idx,
                        rest_length=spacing,
                        stiffness=0.9
                    )
                    springs.append(spring)

        # Springs de shear (diagonales)
        for y in range(height - 1):
            for x in range(width - 1):
                idx = y * width + x

                # Diagonal \
                neighbor_idx = (y + 1) * width + (x + 1)
                spring = Spring(
                    particle_a=idx,
                    particle_b=neighbor_idx,
                    rest_length=spacing * np.sqrt(2),
                    stiffness=0.7
                )
                springs.append(spring)

                # Diagonal /
                neighbor_idx = (y + 1) * width + x
                other_idx = y * width + (x + 1)
                spring = Spring(
                    particle_a=neighbor_idx,
                    particle_b=other_idx,
                    rest_length=spacing * np.sqrt(2),
                    stiffness=0.7
                )
                springs.append(spring)

        # Springs de bend (para resistencia a doblado)
        for y in range(height):
            for x in range(width):
                idx = y * width + x

                # Bend horizontal
                if x < width - 2:
                    neighbor_idx = y * width + (x + 2)
                    spring = Spring(
                        particle_a=idx,
                        particle_b=neighbor_idx,
                        rest_length=spacing * 2,
                        stiffness=0.5
                    )
                    springs.append(spring)

                # Bend vertical
                if y < height - 2:
                    neighbor_idx = (y + 2) * width + x
                    spring = Spring(
                        particle_a=idx,
                        particle_b=neighbor_idx,
                        rest_length=spacing * 2,
                        stiffness=0.5
                    )
                    springs.append(spring)

        # Crear caras para rendering
        for y in range(height - 1):
            for x in range(width - 1):
                idx = y * width + x

                # Dos triángulos por quad
                faces.append((idx, idx + 1, idx + width))
                faces.append((idx + 1, idx + width + 1, idx + width))

        return ClothMesh(
            particles=particles,
            springs=springs,
            faces=faces,
            width=width,
            height=height
        )

    def _configure_pins(self, cloth_mesh: ClothMesh, pin_indices: List[int]):
        """
        Configura partículas pinned (fijas) en la malla de tela.

        Args:
            cloth_mesh: Malla de tela
            pin_indices: Índices de partículas a fijar
        """
        for idx in pin_indices:
            if 0 <= idx < len(cloth_mesh.particles):
                cloth_mesh.particles[idx].is_pinned = True

    def _simulate_cloth(self, cloth_mesh: ClothMesh, num_frames: int) -> List[Dict[str, Any]]:
        """
        Simula la tela por un número de frames.

        Args:
            cloth_mesh: Malla de tela a simular
            num_frames: Número de frames a simular

        Returns:
            Lista de estados por frame
        """
        frames = []
        dt = self.timestep / self.substeps

        for frame in range(num_frames):
            # Ejecutar substeps para mayor precisión
            for _ in range(self.substeps):
                # 1. Aplicar fuerzas
                self._apply_forces(cloth_mesh)

                # 2. Integración de Verlet
                self._verlet_integration(cloth_mesh, dt)

                # 3. Satisfacer constraints (springs)
                self._satisfy_constraints(cloth_mesh)

                # 4. Auto-colisión (simplificada)
                self._handle_self_collision(cloth_mesh)

            # Guardar estado del frame
            positions = [p.position.tolist() for p in cloth_mesh.particles]
            frames.append({
                'frame': frame,
                'positions': positions
            })

        return frames

    def _apply_forces(self, cloth_mesh: ClothMesh):
        """Aplica fuerzas a todas las partículas."""
        for particle in cloth_mesh.particles:
            if not particle.is_pinned:
                # Gravedad
                particle.forces = self.gravity * particle.mass

                # Wind (viento simple)
                wind = np.array([0.5, 0.0, 0.2], dtype=np.float32)
                particle.forces += wind * 0.1

    def _verlet_integration(self, cloth_mesh: ClothMesh, dt: float):
        """
        Integración de Verlet para actualizar posiciones.

        Args:
            cloth_mesh: Malla de tela
            dt: Delta time
        """
        damping = 0.99  # Factor de amortiguamiento

        for particle in cloth_mesh.particles:
            if particle.is_pinned:
                continue

            # Verlet integration
            temp = particle.position.copy()

            # x_new = x + (x - x_old) * damping + a * dt^2
            acceleration = particle.forces / particle.mass
            particle.position = (
                particle.position +
                (particle.position - particle.previous_position) * damping +
                acceleration * dt * dt
            )

            particle.previous_position = temp

            # Resetear fuerzas
            particle.forces = np.zeros(3, dtype=np.float32)

    def _satisfy_constraints(self, cloth_mesh: ClothMesh):
        """Satisface los constraints de springs."""
        iterations = 3  # Número de iteraciones para convergencia

        for _ in range(iterations):
            for spring in cloth_mesh.springs:
                p1 = cloth_mesh.particles[spring.particle_a]
                p2 = cloth_mesh.particles[spring.particle_b]

                if p1.is_pinned and p2.is_pinned:
                    continue

                # Calcular dirección y distancia actual
                delta = p2.position - p1.position
                distance = np.linalg.norm(delta)

                if distance < 1e-6:
                    continue

                # Calcular corrección
                diff = (distance - spring.rest_length) / distance
                correction = delta * diff * spring.stiffness

                # Aplicar corrección
                if not p1.is_pinned and not p2.is_pinned:
                    p1.position += correction * 0.5
                    p2.position -= correction * 0.5
                elif p1.is_pinned:
                    p2.position -= correction
                elif p2.is_pinned:
                    p1.position += correction

    def _handle_self_collision(self, cloth_mesh: ClothMesh):
        """
        Maneja auto-colisiones simplificadas.

        Args:
            cloth_mesh: Malla de tela
        """
        # Implementación simplificada: verificar distancias mínimas
        min_distance = 0.05

        positions = np.array([p.position for p in cloth_mesh.particles])

        # Usar KDTree para búsqueda eficiente de vecinos cercanos
        tree = cKDTree(positions)

        for i, particle in enumerate(cloth_mesh.particles):
            if particle.is_pinned:
                continue

            # Encontrar partículas cercanas
            indices = tree.query_ball_point(particle.position, min_distance)

            for j in indices:
                if i >= j:  # Evitar procesar dos veces
                    continue

                other = cloth_mesh.particles[j]
                if other.is_pinned:
                    continue

                # Calcular repulsión
                delta = particle.position - other.position
                distance = np.linalg.norm(delta)

                if distance < min_distance and distance > 1e-6:
                    correction = delta / distance * (min_distance - distance) * 0.5
                    particle.position += correction
                    other.position -= correction

    def _create_body_collider(self, body_mesh: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea un colisionador simplificado del cuerpo.

        Args:
            body_mesh: Datos de la malla del cuerpo

        Returns:
            Diccionario con datos del colisionador
        """
        vertices = np.array(body_mesh.get('vertices', []))

        if len(vertices) == 0:
            return None

        # Crear esferas de colisión en puntos clave
        # Esto es una aproximación simple
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        center = (bounds_min + bounds_max) / 2
        radius = np.linalg.norm(bounds_max - bounds_min) / 2

        return {
            'type': 'sphere',
            'center': center.tolist(),
            'radius': float(radius),
            'vertices_count': len(vertices)
        }

    def _generate_simulation_code(self) -> Dict[str, str]:
        """
        Genera código optimizado para simulación en tiempo real.

        Returns:
            Diccionario con código de simulación
        """
        cpp_code = """
// Cloth Simulation - Optimized C++ Code
// Para OpenSimulator / Unity / Unreal Engine

#include <vector>
#include <cmath>

struct Particle {
    float position[3];
    float previous_position[3];
    float mass;
    bool is_pinned;
};

struct Spring {
    int particle_a;
    int particle_b;
    float rest_length;
    float stiffness;
};

class ClothSimulator {
private:
    std::vector<Particle> particles;
    std::vector<Spring> springs;
    float gravity[3] = {0.0f, -9.81f, 0.0f};
    float timestep = 1.0f / 60.0f;
    int substeps = 5;
    
public:
    void simulate() {
        float dt = timestep / substeps;
        
        for (int step = 0; step < substeps; ++step) {
            applyForces();
            verletIntegration(dt);
            satisfyConstraints();
        }
    }
    
    void applyForces() {
        for (auto& p : particles) {
            if (!p.is_pinned) {
                p.position[1] += gravity[1] * p.mass * timestep * timestep;
            }
        }
    }
    
    void verletIntegration(float dt) {
        float damping = 0.99f;
        
        for (auto& p : particles) {
            if (p.is_pinned) continue;
            
            float temp[3];
            for (int i = 0; i < 3; ++i) {
                temp[i] = p.position[i];
                p.position[i] = p.position[i] + 
                              (p.position[i] - p.previous_position[i]) * damping;
                p.previous_position[i] = temp[i];
            }
        }
    }
    
    void satisfyConstraints() {
        for (int iter = 0; iter < 3; ++iter) {
            for (const auto& spring : springs) {
                Particle& p1 = particles[spring.particle_a];
                Particle& p2 = particles[spring.particle_b];
                
                if (p1.is_pinned && p2.is_pinned) continue;
                
                float delta[3];
                float distance = 0.0f;
                
                for (int i = 0; i < 3; ++i) {
                    delta[i] = p2.position[i] - p1.position[i];
                    distance += delta[i] * delta[i];
                }
                
                distance = std::sqrt(distance);
                float diff = (distance - spring.rest_length) / distance;
                
                for (int i = 0; i < 3; ++i) {
                    float correction = delta[i] * diff * spring.stiffness;
                    
                    if (!p1.is_pinned && !p2.is_pinned) {
                        p1.position[i] += correction * 0.5f;
                        p2.position[i] -= correction * 0.5f;
                    } else if (p1.is_pinned) {
                        p2.position[i] -= correction;
                    } else if (p2.is_pinned) {
                        p1.position[i] += correction;
                    }
                }
            }
        }
    }
};
"""

        return {
            'language': 'cpp',
            'code': cpp_code,
            'description': 'Simulador de ropa optimizado en C++ para máximo rendimiento'
        }

    def _cloth_to_dict(self, cloth_mesh: ClothMesh) -> Dict[str, Any]:
        """Convierte una ClothMesh a diccionario."""
        return {
            'width': cloth_mesh.width,
            'height': cloth_mesh.height,
            'particle_count': len(cloth_mesh.particles),
            'spring_count': len(cloth_mesh.springs),
            'face_count': len(cloth_mesh.faces),
            'particles': [{
                'position': p.position.tolist(),
                'is_pinned': p.is_pinned,
                'mass': p.mass
            } for p in cloth_mesh.particles[:10]],  # Solo primeras 10 para ejemplo
            'springs': [{
                'particle_a': s.particle_a,
                'particle_b': s.particle_b,
                'rest_length': s.rest_length,
                'stiffness': s.stiffness
            } for s in cloth_mesh.springs[:20]]  # Solo primeros 20 para ejemplo
        }
