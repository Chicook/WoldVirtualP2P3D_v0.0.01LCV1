"""
test_pesos_vivos.py — Suite exhaustiva de pruebas para GestorPesosVivos de LucIA
================================================================================
Cubre la dinámica de pesos vivos, telemetría sináptica y checkpoints en PSNRL:
  1. Constantes ANSI para interfaces de terminal enriquecidas.
  2. Métricas dinámicas de pesos 2026:
     - Entropía de Shannon normalizada sobre vectores de pesos.
     - Deriva espectral Muon (alineación ortogonal W y delta).
     - Barras de progreso en bloques Unicode con porcentaje.
     - Índice de estabilidad y diagnóstico de salud sináptica.
  3. VisualizadorTerminalPesos: renderizado de paneles de turno y cierre de sesión.
  4. GestorPesosVivos:
     - Inyección de turnos en vivo con cálculo de derivas y GSNR por familia.
     - Historial de turnos y acumulador de deriva.
     - Generación de checkpoints binarios comprimidos .npz y metadata JSON en PSNRL.
     - Resumen consolidado de telemetría sináptica.
  5. Contrato de interfaz global: get_gestor_pesos_vivos, inyectar_pesos_turno, checkpoint_y_registrar.
  6. Pruebas de integración con conversor y subsistemas de neuronas (ENRN, RF_SL, etc.).
  7. Casos límite, estabilidad matemática y robustez ante tensores nulos o vacíos.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from LC.celebro.CMFG.pesos_vivos import (
    ANSI,
    GestorPesosVivos,
    MetricasDinamicasPesos,
    VisualizadorTerminalPesos,
    checkpoint_y_registrar,
    get_gestor_pesos_vivos,
    inyectar_pesos_turno,
)
from LC.celebro.test.conftest import (
    assert_tensor_valido,
    assert_norma_acotada,
    psnrl_tmp,
    vector_512,
    matriz_2d_32x32,
)


# ============================================================================
# 1. CONSTANTES ANSI DE TERMINAL
# ============================================================================

class TestANSIConstantes:
    """Verifica la definición de códigos ANSI para telemetría visual."""

    def test_constantes_ansi_definidas(self) -> None:
        """Comprueba que todos los colores y atributos básicos existan."""
        attrs = [
            "RESET", "BOLD", "DIM", "ITALIC", "CYAN", "BLUE",
            "GREEN", "YELLOW", "MAGENTA", "RED", "WHITE", "BG_DARK", "CLEAR",
        ]
        for a in attrs:
            assert hasattr(ANSI, a)
            val = getattr(ANSI, a)
            assert isinstance(val, str)
            assert val.startswith("\033[")

    def test_ansi_reset_estandar(self) -> None:
        """Verifica que ANSI.RESET corresponda al reset estándar."""
        assert ANSI.RESET == "\033[0m"


# ============================================================================
# 2. MÉTRICAS DINÁMICAS DE PESOS 2026
# ============================================================================

class TestMetricasDinamicasPesos:
    """Pruebas para los algoritmos de entropía, deriva Muon y estabilidad."""

    def test_entropia_shannon_vector_uniforme(self) -> None:
        """Verifica que un vector con valores idénticos tenga máxima entropía (cercana a 1.0)."""
        vec = np.ones(64, dtype=np.float32)
        h = MetricasDinamicasPesos.entropia_shannon(vec)
        assert 0.95 <= h <= 1.0

    def test_entropia_shannon_vector_nulo_o_casi_cero(self) -> None:
        """Comprueba que un vector de ceros retorne entropía 0.0 de forma segura."""
        vec = np.zeros(32, dtype=np.float32)
        h = MetricasDinamicasPesos.entropia_shannon(vec)
        assert h == 0.0

    def test_entropia_shannon_vector_pico_unico(self) -> None:
        """Verifica que un vector con un solo elemento activo tenga baja entropía."""
        vec = np.zeros(50, dtype=np.float32)
        vec[0] = 10.0
        h = MetricasDinamicasPesos.entropia_shannon(vec)
        assert h < 0.1

    def test_deriva_espectral_muon_matrices_identicas(self) -> None:
        """Comprueba que matrices colineales den deriva cercana a 1.0."""
        W = np.eye(8, dtype=np.float32)
        delta = np.eye(8, dtype=np.float32) * 0.5
        cos_sim = MetricasDinamicasPesos.deriva_espectral_muon(W, delta)
        assert 0.95 <= cos_sim <= 1.0

    def test_deriva_espectral_muon_matrices_ortogonales(self) -> None:
        """Verifica que matrices ortogonales den deriva cercana a 0.0."""
        W = np.zeros((4, 4), dtype=np.float32)
        W[0, 1] = 1.0
        delta = np.zeros((4, 4), dtype=np.float32)
        delta[1, 0] = 1.0
        cos_sim = MetricasDinamicasPesos.deriva_espectral_muon(W, delta)
        assert cos_sim < 0.05

    def test_barra_progreso_formato(self) -> None:
        """Comprueba el formateo de barras con bloques Unicode y porcentajes."""
        barra_mitad = MetricasDinamicasPesos.barra_progreso(valor=0.5, max_val=1.0, ancho=10)
        assert "█" in barra_mitad
        assert "░" in barra_mitad
        assert "50.0%" in barra_mitad

    def test_barra_progreso_limites(self) -> None:
        """Verifica acotamiento entre 0% y 100% ante valores fuera de rango."""
        barra_cero = MetricasDinamicasPesos.barra_progreso(valor=-1.0, max_val=1.0, ancho=10)
        assert "0.0%" in barra_cero
        barra_cien = MetricasDinamicasPesos.barra_progreso(valor=2.0, max_val=1.0, ancho=10)
        assert "100.0%" in barra_cien

    @pytest.mark.parametrize("gsnr,entropia,deriva,diagnostico_esperado", [
        (2.0, 0.9, 0.05, "OPTIMA"),
        (1.0, 0.5, 0.5, "ESTABLE"),
        (0.1, 0.1, 5.0, "ALERTA"),
    ])
    def test_calcular_indice_estabilidad(
        self, gsnr: float, entropia: float, deriva: float, diagnostico_esperado: str
    ) -> None:
        """Valida los rangos de clasificación del índice de salud sináptica."""
        estabilidad, diag = MetricasDinamicasPesos.calcular_indice_estabilidad(gsnr, entropia, deriva)
        assert 0.0 <= estabilidad <= 1.0
        assert diagnostico_esperado in diag


# ============================================================================
# 3. VISUALIZADOR DE PESOS EN TERMINAL
# ============================================================================

class TestVisualizadorTerminalPesos:
    """Verifica la generación de paneles y reportes formateados en texto."""

    def test_render_panel_turno_estructura(self) -> None:
        """Comprueba que el panel contenga cabecera, tabla de subsistemas y diagnóstico."""
        vis = VisualizadorTerminalPesos()
        stats_sim = {
            "ENRN": {"count": 10, "norma_media": 1.23, "delta_actividad": 0.45, "entropia": 0.8},
            "RF_SL": {"count": 10, "norma_media": 2.34, "delta_actividad": 0.56, "entropia": 0.75},
            "RF_EN": {"count": 10, "norma_media": 0.98, "delta_actividad": 0.32, "entropia": 0.7},
            "RNP": {"count": 10, "norma_media": 1.87, "delta_actividad": 0.65, "entropia": 0.85},
            "SLRN": {"count": 10, "norma_media": 1.45, "delta_actividad": 0.41, "entropia": 0.82},
        }
        panel = vis.render_panel_turno(
            turno_id=1,
            pregunta="¿Cuál es el rol de ENRN?",
            stats_familias=stats_sim,
            deriva_total=0.0123,
            gsnr_global=1.85,
            emocion=0.35,
            tono="analítico",
        )
        assert "MONITOR DE PESOS VIVOS CELEBRO 2026" in panel
        assert "TURNO #0001" in panel
        assert "ENRN" in panel
        assert "SLRN" in panel
        assert "Salud Sináptica" in panel

    def test_render_resumen_cierre_archivos(self, tmp_path: Path) -> None:
        """Verifica el panel de consolidación al cerrar sesión."""
        vis = VisualizadorTerminalPesos()
        f1 = tmp_path / "pesos_1.npz"
        f1.write_bytes(b"12345")
        resumen = vis.render_resumen_cierre(
            total_turnos=3, deriva_acumulada=0.045, archivos_guardados=[f1]
        )
        assert "CONSOLIDACION DE PESOS VIVOS COMPLETADA" in resumen
        assert "Turnos procesados en sesion" in resumen
        assert f1.name in resumen


# ============================================================================
# 4. GESTOR DE PESOS VIVOS (ORQUESTACIÓN Y TELEMETRÍA)
# ============================================================================

class TestGestorPesosVivos:
    """Pruebas del ciclo de inyección y checkpoints de GestorPesosVivos."""

    def test_inicializacion_gestor(self) -> None:
        """Comprueba estado inicial de un nuevo gestor de sesión."""
        gestor = GestorPesosVivos(session_id="SESION_TEST_2026")
        assert gestor.session_id == "SESION_TEST_2026"
        assert gestor.turno_contador == 0
        assert gestor.deriva_sesion == 0.0
        assert gestor.historial_turnos == []

    def test_inyectar_turno_con_conversor_simulado(self) -> None:
        """Verifica la inyección de un turno con mock de conversor."""
        gestor = GestorPesosVivos()

        mock_conv = MagicMock()
        mock_n_enrn = MagicMock()
        mock_n_enrn.pesos = np.random.randn(8, 4).astype(np.float32)
        mock_conv.neuronas = {
            "EN1_RN": mock_n_enrn,
        }

        info_pesos = {
            "estado_emocional": 0.5,
            "tono_cognitivo": "reflexivo",
            "norma_delta_aplicada": 0.05,
        }

        res = gestor.inyectar_turno_en_vivo(
            conversor=mock_conv,
            pregunta="¿Cómo estás?",
            respuesta="Lista para operar.",
            info_pesos=info_pesos,
            mostrar_en_terminal=False,
        )

        assert res["turno"] == 1
        assert res["delta_aplicada"] == 0.05
        assert gestor.deriva_sesion == 0.05
        assert "ENRN" in res["stats_familias"]
        assert len(gestor.historial_turnos) == 1

    def test_checkpoint_y_registrar_genera_npz(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifica la creación de archivos comprimidos .npz en el directorio PSNRL."""
        monkeypatch.setattr("LC.celebro.CMFG.pesos_vivos.PSNRL_DIR", tmp_path)

        gestor = GestorPesosVivos()
        mock_conv = MagicMock()
        mock_n = MagicMock()
        mock_n.pesos = np.ones((4, 4), dtype=np.float32)
        mock_conv.neuronas = {"EN1_RN": mock_n}
        delattr(mock_conv, "persistir_pesos_en_psnrl")

        archivos = gestor.checkpoint_y_registrar(
            conversor=mock_conv, sesion=None, etiqueta="test_chk"
        )
        assert len(archivos) >= 1
        npz_generado = archivos[0]
        assert npz_generado.exists()
        assert npz_generado.name.endswith(".npz")

        datos_npz = np.load(npz_generado)
        assert "EN1_RN_pesos" in datos_npz
        assert datos_npz["EN1_RN_pesos"].shape == (4, 4)

    def test_obtener_telemetria_resumen(self) -> None:
        """Comprueba el reporte estadístico de la sesión."""
        gestor = GestorPesosVivos(session_id="SES_TELEMETRIA")
        gestor.turno_contador = 4
        gestor.deriva_sesion = 0.123
        gestor.historial_gsnr = [1.5, 1.8, 2.0]

        resumen = gestor.obtener_telemetria_resumen()
        assert resumen["session_id"] == "SES_TELEMETRIA"
        assert resumen["total_turnos"] == 4
        assert resumen["deriva_total"] == 0.123
        assert 1.7 <= resumen["gsnr_medio"] <= 1.8


# ============================================================================
# 5. INTEGRACIÓN CON MULTI-FAMILIAS Y CONTRATOS
# ============================================================================

class TestIntegracionFamiliasPesosVivos:
    """Pruebas de inyección contemplando las 5 familias completas."""

    def test_inyeccion_cinco_familias_completas(self) -> None:
        """Verifica que el gestor procese correctamente las 5 familias simultáneamente."""
        gestor = GestorPesosVivos()
        mock_conv = MagicMock()
        mock_conv.neuronas = {}

        prefijos = ["EN", "RFSL", "RFEN", "RNP", "SLRN"]
        for pref in prefijos:
            for i in range(1, 3):
                m = MagicMock()
                m.pesos = np.random.randn(4, 4).astype(np.float32)
                mock_conv.neuronas[f"{pref}_{i}"] = m

        info = {
            "estado_emocional": 0.2,
            "tono_cognitivo": "técnico",
            "norma_delta_aplicada": 0.03,
        }

        res = gestor.inyectar_turno_en_vivo(
            conversor=mock_conv,
            pregunta="Consulta técnica profunda",
            respuesta="Respuesta del cluster",
            info_pesos=info,
            mostrar_en_terminal=False,
        )

        assert res["turno"] == 1
        familias = res["stats_familias"]
        for f in ["ENRN", "RF_SL", "RF_EN", "RNP", "SLRN"]:
            assert f in familias

    def test_registro_en_sesion_externa(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifica que si se pasa un objeto sesion, se invoque registrar_archivo_pesos."""
        monkeypatch.setattr("LC.celebro.CMFG.pesos_vivos.PSNRL_DIR", tmp_path)
        gestor = GestorPesosVivos()
        mock_conv = MagicMock()
        mock_conv.neuronas = {}
        delattr(mock_conv, "persistir_pesos_en_psnrl")

        mock_sesion = MagicMock()
        archs = gestor.checkpoint_y_registrar(conversor=mock_conv, sesion=mock_sesion)
        assert mock_sesion.registrar_archivo_pesos.call_count == len(archs)


# ============================================================================
# 6. CONTRATO DE INTERFAZ GLOBAL Y COMPATIBILIDAD HISTÓRICA
# ============================================================================

class TestInterfazGlobalPesosVivos:
    """Verifica las funciones de módulo expuestas hacia celebro."""

    def test_singleton_gestor_pesos_vivos(self) -> None:
        """Comprueba que get_gestor_pesos_vivos retorne una única instancia singleton."""
        g1 = get_gestor_pesos_vivos()
        g2 = get_gestor_pesos_vivos()
        assert g1 is g2

    def test_inyectar_pesos_turno_wrapper(self) -> None:
        """Verifica la invocación de la función compatible inyectar_pesos_turno."""
        mock_conv = MagicMock()
        mock_conv.neuronas = {}
        info = {"norma_delta_aplicada": 0.01}

        turno_num = inyectar_pesos_turno(
            cerebro=None, conversor=mock_conv, pregunta="Pregunta", respuesta="Resp",
            info_pesos=info, mostrar_en_terminal=False
        )
        assert isinstance(turno_num, int)
        assert turno_num >= 1

    def test_checkpoint_y_registrar_wrapper(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Comprueba la invocación de la función compatible checkpoint_y_registrar."""
        monkeypatch.setattr("LC.celebro.CMFG.pesos_vivos.PSNRL_DIR", tmp_path)
        mock_conv = MagicMock()
        mock_conv.neuronas = {}
        delattr(mock_conv, "persistir_pesos_en_psnrl")

        archs = checkpoint_y_registrar(mock_conv, etiqueta="chk_global")
        assert isinstance(archs, list)
        assert len(archs) >= 1
        assert archs[0].exists()


# ============================================================================
# 7. CASOS LÍMITE Y ROBUSTEZ ANTE VALORES ATÍPICOS
# ============================================================================

class TestRobustezPesosVivos:
    """Verifica tolerancia ante arrays vacíos, NaN o dimensiones no convencionales."""

    def test_deriva_muon_arrays_1d_fallback(self) -> None:
        """Comprueba que arrays 1D no fallen y usen el cálculo de norma relativo."""
        w1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        d1d = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        deriva = MetricasDinamicasPesos.deriva_espectral_muon(w1d, d1d)
        assert isinstance(deriva, float)
        assert deriva >= 0.0

    def test_inyeccion_sin_neuronas_en_conversor(self) -> None:
        """Verifica que un conversor sin diccionario de neuronas opere sin errores."""
        gestor = GestorPesosVivos()
        mock_vacio = object()
        info = {"norma_delta_aplicada": 0.02}

        res = gestor.inyectar_turno_en_vivo(
            conversor=mock_vacio,
            pregunta="P",
            respuesta="R",
            info_pesos=info,
            mostrar_en_terminal=False,
        )
        assert res["turno"] >= 1
        assert "stats_familias" in res

    def test_historial_turnos_convergencia(self) -> None:
        """Comprueba que múltiples inyecciones sucesivas preserven la serie temporal de turnos."""
        gestor = GestorPesosVivos()
        mock_conv = MagicMock()
        mock_conv.neuronas = {}

        for i in range(1, 6):
            info = {"norma_delta_aplicada": 0.01 * i, "estado_emocional": 0.1 * i}
            res = gestor.inyectar_turno_en_vivo(
                conversor=mock_conv,
                pregunta=f"Pregunta #{i}",
                respuesta=f"Respuesta #{i}",
                info_pesos=info,
                mostrar_en_terminal=False,
            )
            assert res["turno"] == i

        assert len(gestor.historial_turnos) == 5
        assert gestor.deriva_sesion > 0.0
        assert gestor.historial_turnos[0]["turno"] == 1
        assert gestor.historial_turnos[-1]["turno"] == 5


# Fin de suite exhaustiva test_pesos_vivos.py
