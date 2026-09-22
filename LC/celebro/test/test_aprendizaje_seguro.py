"""Regresiones del aprendizaje sobre los pesos propios de LucIA."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from LC.celebro.CMFG.SBSTM import APRENDIZAJESEGURO as modulo


class NeuronaFalsa:
    def __init__(self, nombre: str) -> None:
        self.nombre = nombre
        self.pesos = np.zeros((2, 2), dtype=np.float32)
        self.sesgo = np.zeros((1, 2), dtype=np.float32)


class ConversorFalso:
    def __init__(self, total: int = 50) -> None:
        nombres = []
        for grupo in ("EN", "RFSL", "RFEN", "RNP", "SLRN"):
            nombres.extend(f"{grupo}{i}" for i in range(1, total // 5 + 1))
        self.neuronas = {n: NeuronaFalsa(n) for n in nombres}
        self.deriva_acumulada = 0.0
        self.pasos_sesion = 0
        self.historial_actualizaciones = []

    def asimilar_respuestas_y_calcular_sintesis(self, pregunta: str, respuesta: str, modelo: str):
        for neurona in self.neuronas.values():
            neurona.pesos += 1.0
            neurona.sesgo += 1.0
        self.pasos_sesion += 1
        info = {"total_neuronas": len(self.neuronas), "norma_delta_aplicada": 1.0,
                "tono_cognitivo": "analítico"}
        self.historial_actualizaciones.append(info)
        return info

    def persistir_pesos_en_psnrl(self, etiqueta=None):
        return Path("pesos.npz"), Path("pesos.json")


def _aislar_rutas(tmp_path, monkeypatch):
    monkeypatch.setattr(modulo, "PSNRL_DIR", tmp_path)
    monkeypatch.setattr(modulo, "SNAPSHOT", tmp_path / "snapshot.npz")
    monkeypatch.setattr(modulo, "SNAPSHOT_META", tmp_path / "snapshot.json")
    monkeypatch.setattr(modulo, "REPLAY", tmp_path / "replay.jsonl")


def test_gate_rechaza_llamada_de_herramienta():
    aprendizaje = modulo.AprendizajeSeguroLucIA()
    resultado = aprendizaje.validar_alineacion("hola", '{"name":"web_search"}')
    assert resultado["aprobado"] is False


def test_reparte_actualizacion_por_las_50_neuronas(tmp_path, monkeypatch):
    _aislar_rutas(tmp_path, monkeypatch)
    conversor = ConversorFalso()
    aprendizaje = modulo.AprendizajeSeguroLucIA()
    resultado = aprendizaje.asimilar(conversor, "explica la memoria neuronal", "La memoria conserva patrones.", "ollama:qwen2.5:3b")
    assert resultado["aceptado"] is True
    assert resultado["reparto_neuronal"]["total_neuronas"] == 50
    assert resultado["reparto_neuronal"]["neuronas_actualizadas"] == 50
    assert all(np.all(n.pesos > 0) for n in conversor.neuronas.values())
    assert json.loads((tmp_path / "replay.jsonl").read_text(encoding="utf-8"))["modelo_origen"].startswith("ollama:")


def test_rollback_si_aparecen_nan(tmp_path, monkeypatch):
    _aislar_rutas(tmp_path, monkeypatch)
    conversor = ConversorFalso()

    def romper(*args, **kwargs):
        conversor.neuronas["EN1"].pesos[:] = np.nan
        conversor.historial_actualizaciones.append({})
        return {"total_neuronas": 50}

    conversor.asimilar_respuestas_y_calcular_sintesis = romper
    aprendizaje = modulo.AprendizajeSeguroLucIA()
    resultado = aprendizaje.asimilar(conversor, "hola", "respuesta valida", "test")
    assert resultado["rollback"] is True
    assert np.all(np.isfinite(conversor.neuronas["EN1"].pesos))
    assert conversor.pasos_sesion == 0

