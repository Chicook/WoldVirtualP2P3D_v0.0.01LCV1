"""
Sistema de Neuronas Especializadas en Metaversos 3D
Módulo RF_RFENRN1_1_1_8 - WoldVirtual3DlucIA v0.6.0

Este módulo contiene neuronas especializadas en:
- Generación de avatares 3D compatibles con OpenSim/Second Life
- Creación de entornos virtuales inmersivos
- Procesamiento de geometría y meshes
- Sistemas de animación y física
- Networking para metaversos descentralizados
"""

from .RF_RF_RF_RFEN1_RN_1_1_8_1 import (
    AvatarGenerator3D,
    AvatarMorphologyEngine,
    ProceduralAvatarBuilder
)

from .RF_RF_RF_RFEN1_RN_1_1_8_2 import (
    MeshProcessor,
    GeometryOptimizer,
    LODGenerator
)

from .RF_RF_RF_RFEN1_RN_1_1_8_3 import (
    TextureGenerator,
    MaterialSystem,
    PBRMaterialBuilder
)

from .RF_RF_RF_RFEN1_RN_1_1_8_4 import (
    AnimationEngine,
    RiggingSystem,
    MotionCaptureBridge
)

from .RF_RF_RF_RFEN1_RN_1_1_8_5 import (
    TerrainGenerator,
    IslandBuilder,
    ProceduralLandscape
)

from .RF_RF_RF_RFEN1_RN_1_1_8_6 import (
    OpenSimProtocol,
    LLSDParser,
    MetaverseNetworking
)

from .RF_RF_RF_RFEN1_RN_1_1_8_7 import (
    PhysicsEngine,
    CollisionDetector,
    DynamicsSimulator
)

from .RF_RF_RF_RFEN1_RN_1_1_8_8 import (
    NPCBehaviorEngine,
    AIController,
    PathfindingSystem
)

from .RF_RF_RF_RFEN1_RN_1_1_8_9 import (
    RenderingOptimizer,
    ShaderManager,
    PerformanceMonitor
)

from .RF_RF_RF_RFEN1_RN_1_1_8_10 import (
    SecondLifeIntegration,
    OpenSimConnector,
    MetaverseUnifier
)

__version__ = "0.6.0"
__author__ = "WoldVirtual3DlucIA Neural Network"

__all__ = [
    # Módulo 1 - Avatar Generation
    "AvatarGenerator3D",
    "AvatarMorphologyEngine",
    "ProceduralAvatarBuilder",

    # Módulo 2 - Mesh Processing
    "MeshProcessor",
    "GeometryOptimizer",
    "LODGenerator",

    # Módulo 3 - Textures & Materials
    "TextureGenerator",
    "MaterialSystem",
    "PBRMaterialBuilder",

    # Módulo 4 - Animation & Rigging
    "AnimationEngine",
    "RiggingSystem",
    "MotionCaptureBridge",

    # Módulo 5 - Terrain & Islands
    "TerrainGenerator",
    "IslandBuilder",
    "ProceduralLandscape",

    # Módulo 6 - Networking
    "OpenSimProtocol",
    "LLSDParser",
    "MetaverseNetworking",

    # Módulo 7 - Physics
    "PhysicsEngine",
    "CollisionDetector",
    "DynamicsSimulator",

    # Módulo 8 - AI & NPCs
    "NPCBehaviorEngine",
    "AIController",
    "PathfindingSystem",

    # Módulo 9 - Rendering
    "RenderingOptimizer",
    "ShaderManager",
    "PerformanceMonitor",

    # Módulo 10 - Integration
    "SecondLifeIntegration",
    "OpenSimConnector",
    "MetaverseUnifier"
]
