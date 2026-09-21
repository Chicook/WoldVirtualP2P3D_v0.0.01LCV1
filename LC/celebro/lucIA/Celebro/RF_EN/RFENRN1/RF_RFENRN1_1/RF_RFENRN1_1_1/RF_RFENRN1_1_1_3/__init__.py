"""
RF_RFENRN1_1_1_3 - Neuronas de generación 3D/procedural y optimización (preview)
Cada clase define pesos mínimos y un forward simple para el reporte ANSI.
"""

from .RF_RF_RF_RFEN1_RN_1_1_3_1 import SirenSDFNeuron
from .RF_RF_RF_RFEN1_RN_1_1_3_2 import NeRFMiniNeuron
from .RF_RF_RF_RFEN1_RN_1_1_3_3 import HashGridNGPNeuron
from .RF_RF_RF_RFEN1_RN_1_1_3_4 import GaussianSplatNeuron
from .RF_RF_RF_RFEN1_RN_1_1_3_5 import DiffusionLiteNeuron
from .RF_RF_RF_RFEN1_RN_1_1_3_6 import ProceduralNoiseNeuron
from .RF_RF_RF_RFEN1_RN_1_1_3_7 import MarchingCubesNeuron
from .RF_RF_RF_RFEN1_RN_1_1_3_8 import WaveFunctionCollapseNeuron
from .RF_RF_RF_RFEN1_RN_1_1_3_9 import OptimizerPackNeuron
from .RF_RF_RF_RFEN1_RN_1_1_3_10 import DiffRenderHookNeuron

__all__ = [
    "SirenSDFNeuron",
    "NeRFMiniNeuron",
    "HashGridNGPNeuron",
    "GaussianSplatNeuron",
    "DiffusionLiteNeuron",
    "ProceduralNoiseNeuron",
    "MarchingCubesNeuron",
    "WaveFunctionCollapseNeuron",
    "OptimizerPackNeuron",
    "DiffRenderHookNeuron",
]
