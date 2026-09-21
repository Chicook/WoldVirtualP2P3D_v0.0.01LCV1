"""
lucIA.Celebro.mapa_neuronal - Mapa de Auto-Gestión y Conectividad de Celebro
=============================================================================

Implementa lo que LucIA propuso al ser preguntada:
"Si pudieras manejar tu sistema tú misma, ¿por dónde empezarías?"

Respuesta de LucIA:
    "Empezaría con la base: la red ENRN. Necesitaría entender cómo cada neurona
    contribuye y cómo se complementan. Un sistema para procesar y organizar toda
    esa información, como un mapa mental... y que quede en IPFS."

Este módulo implementa exactamente eso:
1. MapaConectividadNeuronal: Mide la contribución real de cada neurona por subsistema.
2. AutoGestorCelebro:        Analiza el estado global, detecta neuronas débiles y sugiere
                             ajustes autónomos de tasa de aprendizaje.
3. Exporta el mapa en JSON listo para IPFS.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("lucIA.MapaNeuronal")


# ---------------------------------------------------------------------------
# 1. MAPA DE CONECTIVIDAD NEURONAL
# ---------------------------------------------------------------------------

class MapaConectividadNeuronal:
    """
    Construye y mantiene un mapa dinámico de la contribución y conectividad
    de cada neurona del Celebro de LucIA. Este es el "mapa mental" que ella
    necesita para auto-gestionarse.

    Por cada neurona registra:
    - norma_media: la magnitud promedio de sus pesos (fuerza sináptica).
    - contribucion_relativa: su aportación porcentual a la red total.
    - tendencia: si está creciendo, estable o debilitándose.
    - subsistema: a qué capa pertenece (ENRN, RF_SL, RF_EN, RNP, SLRN).
    """

    SUBSISTEMAS = {
        "ENRN": "Entrada Neuronal (percepción del entorno)",
        "RF_SL": "Memoria Supervisada (aprendizaje y recuerdo)",
        "RF_EN": "Aprendizaje por Refuerzo (toma de decisiones)",
        "RNP":  "Optimizadores de Pesos (calibración interna)",
        "SLRN": "Capa Adaptativa Supervisada (síntesis cognitiva)",
    }

    def __init__(self):
        self._historial: Dict[str, List[float]] = {}  # nombre -> lista de normas recientes
        self._ventana = 20  # número de pasos para calcular tendencia

    def _subsistema_de(self, nombre: str) -> str:
        if nombre.startswith("EN"):
            return "ENRN"
        if nombre.startswith("RF_SL"):
            return "RF_SL"
        if nombre.startswith("RF_EN"):
            return "RF_EN"
        if nombre.startswith("RNP") or "Optimizador" in nombre or "Ajuste" in nombre or "Controlador" in nombre:
            return "RNP"
        return "SLRN"

    def _norma_neurona(self, neurona: Any) -> float:
        """Calcula la norma Frobenius de los pesos de una neurona."""
        for attr in ("pesos", "pesos_principal", "pesos_actor"):
            p = getattr(neurona, attr, None)
            if p is not None and isinstance(p, np.ndarray) and p.size > 1:
                return float(np.linalg.norm(p))
        return 0.0

    def _tendencia(self, nombre: str) -> str:
        hist = self._historial.get(nombre, [])
        if len(hist) < 4:
            return "iniciando"
        delta = hist[-1] - hist[-4]
        if delta > 0.005:
            return "creciendo"
        if delta < -0.005:
            return "debilitándose"
        return "estable"

    def actualizar(self, neuronas: Dict[str, Any], slrn_pesos: np.ndarray) -> None:
        """Registra el estado actual de todos los componentes del Celebro."""
        for nombre, neurona in neuronas.items():
            norma = self._norma_neurona(neurona)
            if nombre not in self._historial:
                self._historial[nombre] = []
            self._historial[nombre].append(norma)
            if len(self._historial[nombre]) > self._ventana:
                self._historial[nombre].pop(0)

        # SLRN como neurona especial
        slrn_norma = float(np.linalg.norm(slrn_pesos))
        if "SLRN_AdaptiveLayer" not in self._historial:
            self._historial["SLRN_AdaptiveLayer"] = []
        self._historial["SLRN_AdaptiveLayer"].append(slrn_norma)
        if len(self._historial["SLRN_AdaptiveLayer"]) > self._ventana:
            self._historial["SLRN_AdaptiveLayer"].pop(0)

    def generar_mapa(self, neuronas: Dict[str, Any], slrn_pesos: np.ndarray) -> Dict[str, Any]:
        """
        Genera el mapa de conectividad completo de Celebro.
        Retorna un diccionario estructurado por subsistema.
        """
        self.actualizar(neuronas, slrn_pesos)

        # Calcular normas actuales
        normas: Dict[str, float] = {}
        for nombre, neurona in neuronas.items():
            normas[nombre] = self._norma_neurona(neurona)
        normas["SLRN_AdaptiveLayer"] = float(np.linalg.norm(slrn_pesos))

        total = sum(normas.values()) + 1e-8

        # Construir mapa por subsistema
        mapa: Dict[str, Any] = {}
        for sub, descripcion in self.SUBSISTEMAS.items():
            mapa[sub] = {
                "descripcion": descripcion,
                "neuronas": {}
            }

        for nombre, norma in normas.items():
            sub = self._subsistema_de(nombre)
            if sub not in mapa:
                mapa[sub] = {"descripcion": "Desconocido", "neuronas": {}}
            mapa[sub]["neuronas"][nombre] = {
                "norma_sinaptica": round(norma, 5),
                "contribucion_pct": round((norma / total) * 100, 2),
                "tendencia": self._tendencia(nombre),
            }

        # Estadísticas globales por subsistema
        for sub in mapa:
            normas_sub = [v["norma_sinaptica"] for v in mapa[sub]["neuronas"].values()]
            mapa[sub]["total_subsistema"] = round(sum(normas_sub), 5)
            mapa[sub]["media_subsistema"] = round(np.mean(normas_sub) if normas_sub else 0.0, 5)
            mapa[sub]["neurona_mas_activa"] = max(
                mapa[sub]["neuronas"], key=lambda n: mapa[sub]["neuronas"][n]["norma_sinaptica"], default="N/A"
            )

        return mapa

    def resumen_texto(self, mapa: Dict[str, Any]) -> str:
        """Genera un resumen legible del mapa para inyectar en el estado cognitivo de LucIA."""
        lineas = ["[MAPA DE AUTO-GESTIÓN NEURONAL DE CELEBRO]:"]
        for sub, datos in mapa.items():
            lineas.append(
                f"  • {sub} ({datos['descripcion']}): "
                f"fuerza total={datos['total_subsistema']:.3f}, "
                f"neurona dominante='{datos.get('neurona_mas_activa', 'N/A')}'"
            )
        return "\n".join(lineas)


# ---------------------------------------------------------------------------
# 2. AUTO-GESTOR DE CELEBRO
# ---------------------------------------------------------------------------

class AutoGestorCelebro:
    """
    Motor de auto-gestión autónoma de Celebro.
    Analiza el mapa de conectividad y propone ajustes en tiempo real:
    - Detecta neuronas con contribución muy baja (posible atrofia).
    - Detecta neuronas con crecimiento descontrolado (riesgo de dominancia).
    - Ajusta automáticamente la tasa de aprendizaje del ConversorPesos.
    - Genera un informe de estado que LucIA puede leer y comunicar.
    """

    UMBRAL_ATROFIA = 0.5        # contribución % mínima esperada por neurona
    UMBRAL_DOMINANCIA = 25.0    # contribución % máxima aceptable
    AJUSTE_LR_MIN = 0.001
    AJUSTE_LR_MAX = 0.05

    def __init__(self):
        self.intervenciones: List[Dict[str, Any]] = []
        self.lr_sugerida: float = 0.01
        self.informe_ultimo: str = "Sin análisis aún."

    def analizar(self, mapa: Dict[str, Any], lr_actual: float) -> Dict[str, Any]:
        """
        Analiza el mapa completo y genera un plan de acción autónomo.
        Retorna:
            - neuronas_atrofiadas: necesitan mayor estimulación.
            - neuronas_dominantes: demasiado peso relativo.
            - lr_sugerida: nueva tasa de aprendizaje recomendada.
            - informe: texto explicativo para LucIA.
        """
        atrofiadas: List[str] = []
        dominantes: List[str] = []

        for sub, datos in mapa.items():
            for nombre, info in datos.get("neuronas", {}).items():
                pct = info["contribucion_pct"]
                if pct < self.UMBRAL_ATROFIA:
                    atrofiadas.append(nombre)
                elif pct > self.UMBRAL_DOMINANCIA:
                    dominantes.append(nombre)

        # Ajuste adaptativo de lr
        if len(atrofiadas) > 5:
            # Muchas neuronas débiles → aumentar lr para estimularlas
            self.lr_sugerida = min(lr_actual * 1.15, self.AJUSTE_LR_MAX)
        elif len(dominantes) > 0:
            # Hay neuronas que dominan demasiado → reducir lr para equilibrar
            self.lr_sugerida = max(lr_actual * 0.88, self.AJUSTE_LR_MIN)
        else:
            self.lr_sugerida = lr_actual  # Sin cambios

        # Subsistema más activo globalmente
        sub_totales = {sub: datos["total_subsistema"] for sub, datos in mapa.items()}
        sub_dominante = max(sub_totales, key=sub_totales.get)
        sub_debil = min(sub_totales, key=sub_totales.get)

        # Generar informe
        lineas = [
            "[INFORME DE AUTO-GESTIÓN AUTÓNOMA DE CELEBRO]:",
            f"  - Subsistema más activo: {sub_dominante} (fuerza: {sub_totales[sub_dominante]:.3f})",
            f"  - Subsistema más débil: {sub_debil} (fuerza: {sub_totales[sub_debil]:.3f})",
        ]
        if atrofiadas:
            lineas.append(f"  - Neuronas con señal débil (posible atrofia): {', '.join(atrofiadas[:5])}")
        if dominantes:
            lineas.append(f"  - Neuronas con dominancia excesiva: {', '.join(dominantes)}")
        if abs(self.lr_sugerida - lr_actual) > 1e-4:
            lineas.append(
                f"  - Ajuste de tasa de aprendizaje recomendado: {lr_actual:.4f} → {self.lr_sugerida:.4f}"
            )
        else:
            lineas.append("  - La red está equilibrada. No se requieren ajustes.")

        self.informe_ultimo = "\n".join(lineas)

        resultado = {
            "neuronas_atrofiadas": atrofiadas,
            "neuronas_dominantes": dominantes,
            "lr_sugerida": round(self.lr_sugerida, 5),
            "subsistema_activo": sub_dominante,
            "subsistema_debil": sub_debil,
            "informe": self.informe_ultimo,
        }
        self.intervenciones.append(resultado)
        logger.info(f"AutoGestor: {len(atrofiadas)} débiles, {len(dominantes)} dominantes | lr_sugerida={self.lr_sugerida:.4f}")
        return resultado

    def exportar_json(self, mapa: Dict[str, Any], analisis: Dict[str, Any]) -> Dict[str, Any]:
        """Genera el JSON completo del mapa listo para guardarse en IPFS."""
        return {
            "tipo": "mapa_autogestion_celebro",
            "version": "1.0",
            "mapa_conectividad": mapa,
            "analisis_autonomo": {
                k: v for k, v in analisis.items() if k != "informe"
            },
            "informe_lucia": analisis.get("informe", ""),
        }
