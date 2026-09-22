"""Pruebas del diagnóstico factual de arquitectura."""

from __future__ import annotations

from LC.compatibility.architecture import architecture_snapshot


def test_diagnostico_no_confunde_tests_disponibles_con_tests_pasados() -> None:
    state = architecture_snapshot()
    assert state["live_checks"]["architecture_tests_available"] is True
    assert state["live_checks"]["architecture_tests_verified"] is False


def test_refactor_de_modulos_vivos_no_se_declara_completado() -> None:
    state = architecture_snapshot()
    live_refactor = next(item for item in state["capabilities"]
                         if item["id"] == "live-module-refactor")
    assert live_refactor["status"] == "blocked"
