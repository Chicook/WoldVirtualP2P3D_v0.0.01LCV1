"""
lucIA.CORE.utils - Utilidades matemáticas y métricas de optimización
"""

import numpy as np
from typing import List, Union


def check_convergence(loss_history: List[float], patience: int = 10, min_delta: float = 1e-4) -> bool:
    """
    Verifica convergencia de pérdida basado en paciencia y cambio mínimo relativo/absoluto.
    """
    if not loss_history or len(loss_history) <= patience:
        return False
    
    recent = loss_history[-patience:]
    best_loss = min(loss_history[:-patience]) if len(loss_history) > patience else loss_history[0]
    recent_best = min(recent)
    
    # Si la mejora no supera min_delta durante 'patience' épocas, ha convergido
    improvement = best_loss - recent_best
    return bool(improvement < min_delta)


def normalize_data(data: np.ndarray, method: str = 'z_score') -> np.ndarray:
    """
    Normaliza un arreglo numpy según el método especificado.
    """
    if method == 'z_score':
        std = np.std(data)
        if std == 0:
            return data - np.mean(data)
        return (data - np.mean(data)) / (std + 1e-8)
    elif method == 'min_max':
        min_v = np.min(data)
        max_v = np.max(data)
        diff = max_v - min_v
        if diff == 0:
            return np.zeros_like(data)
        return (data - min_v) / (diff + 1e-8)
    return data
