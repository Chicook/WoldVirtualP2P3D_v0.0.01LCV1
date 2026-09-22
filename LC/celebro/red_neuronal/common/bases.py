"""bases.py — NeuronaBase común."""
from __future__ import annotations
import numpy as np
from typing import Dict

class NeuronaBase:
    def __init__(self, input_size, output_size, nombre="Neurona"):
        self.input_size = int(input_size); self.output_size = int(output_size)
        self.nombre = str(nombre)
        self.pesos = None; self.sesgo = None
        self.historial_activaciones = []; self.historial_gradientes = []; self.pasos = 0
    def inicializar_pesos(self): raise NotImplementedError
    def forward(self, e): raise NotImplementedError
    def resetear_historial(self): self.historial_activaciones = []; self.historial_gradientes = []
    def info(self): return f"{self.nombre}(in={self.input_size},out={self.output_size},pasos={self.pasos})"
    def params_count(self):
        t = 0
        if self.pesos is not None: t += self.pesos.size
        if self.sesgo is not None: t += self.sesgo.size
        return t
