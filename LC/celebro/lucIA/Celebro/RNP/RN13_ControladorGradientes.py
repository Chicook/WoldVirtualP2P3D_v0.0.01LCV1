"""
RN13_ControladorGradientes.py - Subsistema de Control de Gradientes
=====================================================================
Nodo neuronal dentro de RNP diseñado para monitorizar, estabilizar y 
aplicar clipping (recorte) a los gradientes antes de la actualización
de pesos sinápticos en Celebro, previniendo inestabilidades.
"""

import numpy as np
import logging

logger = logging.getLogger("lucIA.RN13_ControladorGradientes")

class ControladorGradientes:
    def __init__(self, max_norm: float = 1.0, lr_adaptativo: bool = True):
        self.nombre = "RN13_ControladorGradientes"
        self.max_norm = max_norm
        self.lr_adaptativo = lr_adaptativo
        
        self.historial_normas = []
        self.factor_escala_actual = 1.0
        self.veces_intervenido = 0
        # P1: suavizado adaptativo — evita el "muro" (recorte x0.002 de golpe)
        self.norma_suavizada = 0.0
        self.alfa_suavizado = 0.2
        
        # Para compatibilidad con la exportación/importación del RNP y la actualización de pesos:
        self.pesos = np.zeros((4, 8), dtype=np.float32)

    def inicializar_pesos(self):
        pass

    def procesar_gradiente(self, delta_base: np.ndarray) -> tuple[np.ndarray, float, float, str]:
        """
        Analiza el gradiente entrante (delta_base) y aplica clipping si excede max_norm.
        Retorna:
            delta_modificado: Gradiente con el clipping aplicado (si corresponde).
            norma_original: La magnitud que entraba (pre-clip, solo diagnostico).
            norma_aplicada: La magnitud que realmente se aplica a los pesos.
            factor_escala: Factor por el cual se escaló el gradiente.
            estado_control: Mensaje explicativo del estado del controlador.
        """
        norma_original = float(np.linalg.norm(delta_base))
        self.historial_normas.append(norma_original)

        # P1: media movil de la norma para detectar picos con suavidad.
        # Warmup: arranca en regimen estable (0.01), no en la primera norma,
        # para no envenenar la media con el primer pico gigante.
        if self.norma_suavizada == 0.0:
            self.norma_suavizada = 0.01
            self._muestras_warmup = 1
        else:
            self.norma_suavizada = (self.alfa_suavizado * norma_original
                                    + (1.0 - self.alfa_suavizado) * self.norma_suavizada)
        
        # Mantener ventana móvil
        if len(self.historial_normas) > 50:
            self.historial_normas.pop(0)

        factor_escala = 1.0
        estado_control = "Estable"

        if norma_original > self.max_norm:
            # P1: clip progresivo — techo adaptativo segun historial para no cortar de golpe.
            # Anclado al regimen estable: si la mediana historica ya es baja,
            # el techo es max_norm sin inflar (evita subir a max*2 por inercia).
            techo = self.max_norm
            if len(self.historial_normas) >= 5:
                mediana = float(np.median(self.historial_normas[-10:]))
                if mediana >= 0.1:
                    techo = float(np.clip(mediana * 2.0, self.max_norm / 2.0, self.max_norm * 2.0))
            factor_escala = techo / (norma_original + 1e-8)
            delta_modificado = delta_base * factor_escala
            self.veces_intervenido += 1
            norma_aplicada = float(np.linalg.norm(delta_modificado))
            estado_control = f"Sobrecarga suavizada (entra {norma_original:.2f}, aplico {norma_aplicada:.2f}, techo {techo:.2f})"
            logger.debug(f"RN13 suavizo gradiente: {norma_original:.3f} -> {norma_aplicada:.3f} (techo: {techo:.3f})")
        else:
            delta_modificado = delta_base
            norma_aplicada = norma_original

        self.factor_escala_actual = factor_escala

        # Suelo de aprendizaje: si entraba una señal significativa pero tras
        # suavizarla queda casi en cero, Lucía siente que "no avanza". Se
        # reescala con suavidad hasta un mínimo significativo (0.08), sin
        # amplificar ruido (si lo que entraba ya era mínimo, se deja estar).
        if norma_original >= 0.5 and norma_aplicada < 0.05:
            _suelo = 0.08
            _boost = _suelo / (norma_aplicada + 1e-8)
            delta_modificado = delta_base * factor_escala * _boost
            norma_aplicada = float(np.linalg.norm(delta_modificado))
            estado_control += f" + suelo anti-estancamiento (aplico {_suelo})"
            logger.debug(f"RN13 aplico suelo: {norma_aplicada:.3f}")

        return delta_modificado, norma_original, norma_aplicada, factor_escala, estado_control

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        No altera la propagación hacia adelante.
        Simplemente pasa la señal.
        """
        # Simulamos una salida genérica de RNP para compatibilidad
        return np.ones((1, 8), dtype=np.float32) * self.factor_escala_actual
