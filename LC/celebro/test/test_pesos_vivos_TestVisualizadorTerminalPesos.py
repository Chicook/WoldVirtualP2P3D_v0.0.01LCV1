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

