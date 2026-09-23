import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import LC.celebro.red_neuronal.ENRN as enrn_pkg
import LC.celebro.red_neuronal.RF_SL.RFSL1 as rfsl_pkg
import LC.celebro.red_neuronal.RF_EN as rfen_pkg
import LC.celebro.red_neuronal.RNP as rnp_pkg
import LC.celebro.red_neuronal.SLRN as slrn_pkg
from LC.celebro.red_neuronal.SLRN import SupervisedLearningNeuralConfig
from LC.celebro.CMFG.neural_math import (
    NeuralMathPrecision2026,
    SemanticEncoderSHA,
    SemanticEncoder4D,
)
from LC.celebro.CMFG.PSNRCV import PSNRL_DIR

logger = logging.getLogger("WoldVirtualP2P3D.PSNRCV")
from LC.Constructor.LUCIA_20260923_031455.celebro.CMFG.PSNRCV_ConversorRespuestaPesos_ConversorRespuestaPesos import ConversorRespuestaPesos  # CLASSPACK
