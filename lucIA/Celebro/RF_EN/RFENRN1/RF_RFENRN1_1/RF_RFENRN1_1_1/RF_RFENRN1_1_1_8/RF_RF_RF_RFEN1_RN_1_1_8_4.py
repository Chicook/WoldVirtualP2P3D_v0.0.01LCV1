"""
Neurona Especializada 4: Sistema de Animación y Rigging
Animaciones procedurales y motion capture para avatares 3D
WoldVirtual3DlucIA v0.6.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class AnimationType(Enum):
    """Tipos de animaciones soportadas"""
    IDLE = "idle"
    WALK = "walk"
    RUN = "run"
    JUMP = "jump"
    DANCE = "dance"
    GESTURE = "gesture"
    EMOTE = "emote"
    CUSTOM = "custom"


class InterpolationMode(Enum):
    """Modos de interpolación"""
    LINEAR = "linear"
    CUBIC = "cubic"
    BEZIER = "bezier"
    STEP = "step"


@dataclass
class Keyframe:
    """Fotograma clave de animación"""
    time: float
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # Quaternion
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class BoneAnimation:
    """Animación de un hueso"""
    bone_name: str
    keyframes: List[Keyframe] = field(default_factory=list)
    interpolation: InterpolationMode = InterpolationMode.LINEAR


@dataclass
class AnimationClip:
    """Clip de animación completo"""
    name: str
    duration: float
    bone_animations: Dict[str, BoneAnimation] = field(default_factory=dict)
    loop: bool = True
    fps: int = 30


class AnimationEngine:
    """
    Motor de animación avanzado
    Compatible con OpenSim/Second Life BVH format
    """

    def __init__(self):
        self.animation_library = {}
        self.current_animations = {}
        self.blend_factor = 0.0

    def create_animation_clip(
        self,
        name: str,
        duration: float,
        fps: int = 30,
        loop: bool = True
    ) -> AnimationClip:
        """Crea un nuevo clip de animación"""
        clip = AnimationClip(
            name=name,
            duration=duration,
            fps=fps,
            loop=loop
        )
        self.animation_library[name] = clip
        return clip

    def add_bone_animation(
        self,
        clip: AnimationClip,
        bone_name: str,
        keyframes: List[Keyframe]
    ):
        """Añade animación a un hueso específico"""
        bone_anim = BoneAnimation(
            bone_name=bone_name,
            keyframes=sorted(keyframes, key=lambda k: k.time)
        )
        clip.bone_animations[bone_name] = bone_anim

    def evaluate_animation(
        self,
        clip: AnimationClip,
        time: float
    ) -> Dict[str, Tuple]:
        """
        Evalúa la animación en un tiempo específico

        Args:
            clip: Clip de animación
            time: Tiempo en segundos

        Returns:
            Dict con transformaciones por hueso
        """
        # Manejar loop
        if clip.loop:
            time = time % clip.duration
        else:
            time = min(time, clip.duration)

        bone_transforms = {}

        for bone_name, bone_anim in clip.bone_animations.items():
            transform = self._evaluate_bone_animation(bone_anim, time)
            bone_transforms[bone_name] = transform

        return bone_transforms

    def _evaluate_bone_animation(
        self,
        bone_anim: BoneAnimation,
        time: float
    ) -> Tuple:
        """Evalúa animación de un hueso en un tiempo específico"""
        keyframes = bone_anim.keyframes

        if len(keyframes) == 0:
            return ((0, 0, 0), (0, 0, 0, 1), (1, 1, 1))

        if len(keyframes) == 1:
            kf = keyframes[0]
            return (kf.position, kf.rotation, kf.scale)

        # Encontrar keyframes antes y después
        before_kf = keyframes[0]
        after_kf = keyframes[-1]

        for i in range(len(keyframes) - 1):
            if keyframes[i].time <= time <= keyframes[i + 1].time:
                before_kf = keyframes[i]
                after_kf = keyframes[i + 1]
                break

        # Interpolar
        if before_kf.time == after_kf.time:
            t = 0.0
        else:
            t = (time - before_kf.time) / (after_kf.time - before_kf.time)

        position = self._interpolate_position(
            before_kf.position,
            after_kf.position,
            t,
            bone_anim.interpolation
        )

        rotation = self._interpolate_rotation(
            before_kf.rotation,
            after_kf.rotation,
            t
        )

        scale = self._interpolate_position(  # Misma lógica para escala
            before_kf.scale,
            after_kf.scale,
            t,
            bone_anim.interpolation
        )

        return (position, rotation, scale)

    def _interpolate_position(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float],
        t: float,
        mode: InterpolationMode
    ) -> Tuple[float, float, float]:
        """Interpola entre dos posiciones"""
        p1 = np.array(pos1)
        p2 = np.array(pos2)

        if mode == InterpolationMode.LINEAR:
            result = p1 + t * (p2 - p1)
        elif mode == InterpolationMode.CUBIC:
            # Interpolación cúbica suave
            t_smooth = t * t * (3.0 - 2.0 * t)
            result = p1 + t_smooth * (p2 - p1)
        else:
            result = p1 + t * (p2 - p1)

        return tuple(result)

    def _interpolate_rotation(
        self,
        rot1: Tuple[float, float, float, float],
        rot2: Tuple[float, float, float, float],
        t: float
    ) -> Tuple[float, float, float, float]:
        """
        Interpola rotaciones usando SLERP (Spherical Linear Interpolation)
        Quaternions en formato (x, y, z, w)
        """
        q1 = np.array(rot1)
        q2 = np.array(rot2)

        # Normalizar quaternions
        q1 = q1 / (np.linalg.norm(q1) + 1e-10)
        q2 = q2 / (np.linalg.norm(q2) + 1e-10)

        # Producto punto
        dot = np.dot(q1, q2)

        # Si dot es negativo, invertir uno de los quaternions
        if dot < 0.0:
            q2 = -q2
            dot = -dot

        # Clamp dot
        dot = np.clip(dot, -1.0, 1.0)

        # SLERP
        if dot > 0.9995:
            # Quaternions muy cercanos, usar interpolación lineal
            result = q1 + t * (q2 - q1)
        else:
            theta = np.arccos(dot)
            sin_theta = np.sin(theta)

            w1 = np.sin((1.0 - t) * theta) / sin_theta
            w2 = np.sin(t * theta) / sin_theta

            result = w1 * q1 + w2 * q2

        # Normalizar resultado
        result = result / (np.linalg.norm(result) + 1e-10)

        return tuple(result)

    def blend_animations(
        self,
        clip1: AnimationClip,
        clip2: AnimationClip,
        time1: float,
        time2: float,
        blend_factor: float
    ) -> Dict[str, Tuple]:
        """
        Mezcla dos animaciones

        Args:
            clip1, clip2: Clips a mezclar
            time1, time2: Tiempos de evaluación
            blend_factor: Factor de mezcla (0=clip1, 1=clip2)

        Returns:
            Transformaciones mezcladas
        """
        transforms1 = self.evaluate_animation(clip1, time1)
        transforms2 = self.evaluate_animation(clip2, time2)

        # Obtener todos los huesos
        all_bones = set(transforms1.keys()) | set(transforms2.keys())

        blended = {}

        for bone in all_bones:
            t1 = transforms1.get(bone, ((0, 0, 0), (0, 0, 0, 1), (1, 1, 1)))
            t2 = transforms2.get(bone, ((0, 0, 0), (0, 0, 0, 1), (1, 1, 1)))

            # Mezclar posición
            pos = self._interpolate_position(
                t1[0], t2[0],
                blend_factor,
                InterpolationMode.LINEAR
            )

            # Mezclar rotación
            rot = self._interpolate_rotation(
                t1[1], t2[1],
                blend_factor
            )

            # Mezclar escala
            scale = self._interpolate_position(
                t1[2], t2[2],
                blend_factor,
                InterpolationMode.LINEAR
            )

            blended[bone] = (pos, rot, scale)

        return blended

    def generate_procedural_walk(
        self,
        duration: float = 2.0,
        step_length: float = 0.5
    ) -> AnimationClip:
        """Genera animación procedural de caminar"""
        clip = self.create_animation_clip("procedural_walk", duration, fps=30, loop=True)

        # Animar pelvis (movimiento vertical)
        pelvis_keyframes = [
            Keyframe(0.0, position=(0, 0.9, 0)),
            Keyframe(duration * 0.25, position=(0, 0.88, 0)),
            Keyframe(duration * 0.5, position=(0, 0.9, 0)),
            Keyframe(duration * 0.75, position=(0, 0.88, 0)),
            Keyframe(duration, position=(0, 0.9, 0))
        ]
        self.add_bone_animation(clip, "pelvis", pelvis_keyframes)

        # Animar pierna izquierda
        left_leg_keyframes = [
            Keyframe(0.0, rotation=self._euler_to_quaternion(0, 0, 0)),
            Keyframe(duration * 0.5, rotation=self._euler_to_quaternion(30, 0, 0)),
            Keyframe(duration, rotation=self._euler_to_quaternion(0, 0, 0))
        ]
        self.add_bone_animation(clip, "thigh_l", left_leg_keyframes)

        # Animar pierna derecha (opuesta)
        right_leg_keyframes = [
            Keyframe(0.0, rotation=self._euler_to_quaternion(30, 0, 0)),
            Keyframe(duration * 0.5, rotation=self._euler_to_quaternion(0, 0, 0)),
            Keyframe(duration, rotation=self._euler_to_quaternion(30, 0, 0))
        ]
        self.add_bone_animation(clip, "thigh_r", right_leg_keyframes)

        return clip

    def _euler_to_quaternion(self, pitch: float, yaw: float, roll: float) -> Tuple:
        """Convierte ángulos de Euler (grados) a quaternion"""
        # Convertir a radianes
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        roll = np.radians(roll)

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return (x, y, z, w)


class RiggingSystem:
    """Sistema de rigging para avatares"""

    def __init__(self):
        self.skeleton_templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict:
        """Inicializa plantillas de esqueletos"""
        return {
            "humanoid": {
                "bones": [
                    "root", "pelvis", "spine", "chest", "neck", "head",
                    "shoulder_l", "upper_arm_l", "forearm_l", "hand_l",
                    "shoulder_r", "upper_arm_r", "forearm_r", "hand_r",
                    "thigh_l", "calf_l", "foot_l",
                    "thigh_r", "calf_r", "foot_r"
                ],
                "hierarchy": {
                    "root": ["pelvis"],
                    "pelvis": ["spine", "thigh_l", "thigh_r"],
                    "spine": ["chest"],
                    "chest": ["neck", "shoulder_l", "shoulder_r"],
                    "neck": ["head"],
                    "shoulder_l": ["upper_arm_l"],
                    "upper_arm_l": ["forearm_l"],
                    "forearm_l": ["hand_l"],
                    "shoulder_r": ["upper_arm_r"],
                    "upper_arm_r": ["forearm_r"],
                    "forearm_r": ["hand_r"],
                    "thigh_l": ["calf_l"],
                    "calf_l": ["foot_l"],
                    "thigh_r": ["calf_r"],
                    "calf_r": ["foot_r"]
                }
            }
        }

    def create_rig(
        self,
        template_name: str,
        scale: float = 1.0
    ) -> Dict:
        """
        Crea un rig desde una plantilla

        Args:
            template_name: Nombre de la plantilla
            scale: Escala del rig

        Returns:
            Datos del rig
        """
        template = self.skeleton_templates.get(template_name)

        if not template:
            raise ValueError(f"Template {template_name} no encontrado")

        rig = {
            "bones": {},
            "hierarchy": template["hierarchy"],
            "scale": scale
        }

        # Inicializar huesos con posiciones por defecto
        for bone_name in template["bones"]:
            rig["bones"][bone_name] = {
                "position": [0, 0, 0],
                "rotation": [0, 0, 0, 1],
                "scale": [1, 1, 1],
                "length": 0.1 * scale
            }

        return rig

    def calculate_inverse_kinematics(
        self,
        bone_chain: List[str],
        target_position: Tuple[float, float, float],
        max_iterations: int = 10
    ) -> Dict[str, Tuple]:
        """
        Calcula IK (Inverse Kinematics) para una cadena de huesos
        Implementa algoritmo FABRIK simplificado

        Args:
            bone_chain: Lista de nombres de huesos en la cadena
            target_position: Posición objetivo
            max_iterations: Iteraciones máximas

        Returns:
            Rotaciones calculadas por hueso
        """
        # Implementación simplificada de FABRIK
        # En producción usar implementación completa

        rotations = {}

        for bone in bone_chain:
            # Calcular dirección hacia target
            direction = np.array(target_position)
            direction = direction / (np.linalg.norm(direction) + 1e-10)

            # Convertir dirección a rotación (simplificado)
            angle = np.arctan2(direction[2], direction[0])
            quat = self._angle_axis_to_quaternion(angle, (0, 1, 0))

            rotations[bone] = quat

        return rotations

    def _angle_axis_to_quaternion(
        self,
        angle: float,
        axis: Tuple[float, float, float]
    ) -> Tuple:
        """Convierte ángulo-eje a quaternion"""
        axis = np.array(axis)
        axis = axis / (np.linalg.norm(axis) + 1e-10)

        half_angle = angle / 2.0
        s = np.sin(half_angle)

        return (
            axis[0] * s,
            axis[1] * s,
            axis[2] * s,
            np.cos(half_angle)
        )


class MotionCaptureBridge:
    """Puente para integración de motion capture"""

    def __init__(self):
        self.capture_data = []
        self.animation_engine = AnimationEngine()

    def import_bvh(self, bvh_data: str) -> AnimationClip:
        """
        Importa datos BVH (BioVision Hierarchy)
        Compatible con OpenSim/Second Life

        Args:
            bvh_data: Datos BVH como string

        Returns:
            AnimationClip generado
        """
        # Parser BVH simplificado
        lines = bvh_data.strip().split('\n')

        # Crear clip básico
        clip = AnimationClip(
            name="imported_bvh",
            duration=5.0,
            fps=30
        )

        # En producción, implementar parser completo de BVH
        # Esto es una versión simplificada

        return clip

    def export_to_bvh(self, clip: AnimationClip) -> str:
        """
        Exporta animación a formato BVH

        Args:
            clip: Clip a exportar

        Returns:
            String con datos BVH
        """
        bvh_lines = []

        bvh_lines.append("HIERARCHY")
        bvh_lines.append("ROOT root")
        bvh_lines.append("{")
        bvh_lines.append("  OFFSET 0.0 0.0 0.0")
        bvh_lines.append("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")

        # Exportar jerarquía de huesos
        for bone_name in clip.bone_animations.keys():
            bvh_lines.append(f"  JOINT {bone_name}")
            bvh_lines.append("  {")
            bvh_lines.append("    OFFSET 0.0 0.0 0.0")
            bvh_lines.append("    CHANNELS 3 Zrotation Xrotation Yrotation")
            bvh_lines.append("  }")

        bvh_lines.append("}")

        # Sección de motion
        bvh_lines.append("MOTION")
        frames = int(clip.duration * clip.fps)
        bvh_lines.append(f"Frames: {frames}")
        bvh_lines.append(f"Frame Time: {1.0/clip.fps}")

        # Exportar keyframes
        for frame in range(frames):
            time = frame / clip.fps
            transforms = self.animation_engine.evaluate_animation(clip, time)

            frame_data = []
            for bone_name in clip.bone_animations.keys():
                if bone_name in transforms:
                    pos, rot, scale = transforms[bone_name]
                    frame_data.extend([f"{p:.6f}" for p in pos])
                    # Convertir quaternion a Euler para BVH
                    euler = self._quaternion_to_euler(rot)
                    frame_data.extend([f"{e:.6f}" for e in euler])

            bvh_lines.append(" ".join(frame_data))

        return "\n".join(bvh_lines)

    def _quaternion_to_euler(self, quat: Tuple) -> Tuple:
        """Convierte quaternion a ángulos de Euler (grados)"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return (np.degrees(pitch), np.degrees(yaw), np.degrees(roll))
