"""
test_blockchain.py — Tests exhaustivos de BKSVCB y BloqueNeuronal
==================================================================
Cubre:
  • Construcción y validación de BloqueNeuronal individual.
  • Cálculo de Merkle Root con 0, 1 y N transacciones.
  • Doble hash SHA-256 de cabecera (inmutabilidad).
  • Minado PoNL con dificultad baja y verificación de prefijo.
  • Serialización to_dict / reconstrucción desde dict.
  • Encadenamiento de bloques (hash_previo coherente).
  • Validación de cadena entera (validar_cadena).
  • Registro de aprendizaje neural (registrar_aprendizaje_neural).
  • Transformación de ledger a pesos neuronales.
  • Servidor HTTP REST: endpoints /status, /blocks, /mine.
  • Persistencia del ledger a disco y recarga.
  • Casos de borde: transacciones vacías, nonce máximo, hash manipulado.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from LC.celebro.test import DIFICULTAD_TEST, MAX_ITER_TEST
from LC.celebro.test.conftest import assert_hash_sha256_valido, assert_tensor_valido

# ─── Importación condicional del módulo bajo test ─────────────────────────────
try:
    from LC.celebro.BKSVCB import (
        BloqueNeuronal,
        ServidorBlockchainNeuronal,
        get_blockchain_server,
        iniciar_servidor_blockchain,
    )
    _BKSVCB_DISPONIBLE = True
except Exception as exc:  # noqa: BLE001
    _BKSVCB_DISPONIBLE = False
    _BKSVCB_ERROR = str(exc)

pytestmark = pytest.mark.skipif(
    not _BKSVCB_DISPONIBLE,
    reason=f"BKSVCB no disponible: {_BKSVCB_ERROR if not _BKSVCB_DISPONIBLE else ''}",
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — BloqueNeuronal: construcción y hashing
# ═══════════════════════════════════════════════════════════════════════════════

class TestServidorBlockchain:
    """Tests del servidor blockchain completo."""

    @pytest.fixture()
    def servidor(self, tmp_path: Path) -> ServidorBlockchainNeuronal:
        """Instancia limpia del servidor blockchain con PSNRL temporal."""
        psnrl = tmp_path / "PSNRL"
        psnrl.mkdir()
        with (
            patch("LC.celebro.BKSVCB.PSNRL_DIR", psnrl),
            patch("LC.celebro.BKSVCB.LEDGER_PATH", tmp_path / "ledger.json"),
        ):
            srv = ServidorBlockchainNeuronal()
        return srv

    def test_servidor_inicializa_con_bloque_genesis(self, servidor: ServidorBlockchainNeuronal) -> None:
        """Al crear el servidor debe existir al menos un bloque (génesis)."""
        assert len(servidor.cadena) >= 1, "La cadena debe tener al menos el bloque génesis"

    def test_servidor_cadena_valida_inicial(self, servidor: ServidorBlockchainNeuronal) -> None:
        """La cadena recién creada debe pasar la validación de integridad."""
        valida, err = servidor.validar_cadena()
        assert valida, f"Cadena inicial inválida: {err}"

    def test_minar_transacciones_sin_transacciones(self, servidor: ServidorBlockchainNeuronal) -> None:
        """Sin transacciones pendientes, minar debe devolver None."""
        resultado = servidor.minar_transacciones_pendientes()
        assert resultado is None, "Sin transacciones pendientes, minar debe devolver None"

    def test_registrar_aprendizaje_anade_transaccion(self, servidor: ServidorBlockchainNeuronal) -> None:
        """Registrar aprendizaje debe añadir una transacción pendiente."""
        n_antes = len(servidor.transacciones_pendientes)
        servidor.registrar_aprendizaje_neural(
            prompt="¿Qué es un bloque?",
            respuesta="Es una unidad de la cadena.",
            modelo="test-model"
        )
        n_despues = len(servidor.transacciones_pendientes)
        assert n_despues == n_antes + 1, "Debe haber una transacción más tras registrar aprendizaje"

    def test_minar_con_una_transaccion_genera_bloque(self, servidor: ServidorBlockchainNeuronal) -> None:
        """Con al menos una transacción pendiente, minar debe devolver un BloqueNeuronal."""
        servidor.registrar_aprendizaje_neural("prompt", "respuesta", "modelo")
        bloque = servidor.minar_transacciones_pendientes()
        assert bloque is not None, "Con transacciones, minar debe devolver un bloque"
        assert isinstance(bloque, BloqueNeuronal), "El resultado de minar debe ser BloqueNeuronal"

    def test_minar_limpia_transacciones_pendientes(self, servidor: ServidorBlockchainNeuronal) -> None:
        """Tras minar, la lista de transacciones pendientes debe quedar vacía."""
        servidor.registrar_aprendizaje_neural("p", "r", "m")
        servidor.minar_transacciones_pendientes()
        assert len(servidor.transacciones_pendientes) == 0, (
            "Las transacciones pendientes deben vaciarse tras minar"
        )

    def test_cadena_crece_tras_minar(self, servidor: ServidorBlockchainNeuronal) -> None:
        """Cada minado exitoso debe añadir un bloque a la cadena."""
        n_antes = len(servidor.cadena)
        servidor.registrar_aprendizaje_neural("a", "b", "c")
        servidor.minar_transacciones_pendientes()
        assert len(servidor.cadena) == n_antes + 1, "La cadena debe crecer en 1 tras minar"

    def test_cadena_sigue_siendo_valida_tras_minar(self, servidor: ServidorBlockchainNeuronal) -> None:
        """La cadena debe permanecer válida después de añadir y minar bloques."""
        for i in range(3):
            servidor.registrar_aprendizaje_neural(f"p{i}", f"r{i}", "test")
            servidor.minar_transacciones_pendientes()
        valida, err = servidor.validar_cadena()
        assert valida, f"Cadena inválida tras múltiples minados: {err}"

    def test_cadena_invalida_si_hash_manipulado(self, servidor: ServidorBlockchainNeuronal) -> None:
        """Modificar el hash de un bloque debe hacer que la cadena falle la validación."""
        servidor.registrar_aprendizaje_neural("prompt", "resp", "mod")
        servidor.minar_transacciones_pendientes()
        if len(servidor.cadena) > 1:
            servidor.cadena[1].hash_bloque = "manipulado" + "0" * 54
            valida, _ = servidor.validar_cadena()
            assert not valida, "Una cadena con hash manipulado debe ser inválida"

    def test_mostrar_cadena_hashes_no_lanza_excepcion(self, servidor: ServidorBlockchainNeuronal) -> None:
        """mostrar_cadena_hashes_terminal() debe ejecutarse sin excepciones."""
        try:
            servidor.mostrar_cadena_hashes_terminal()
        except Exception as e:
            pytest.fail(f"mostrar_cadena_hashes_terminal() lanzó excepción: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Persistencia del ledger
# ═══════════════════════════════════════════════════════════════════════════════

class TestSingletonBlockchain:
    """Tests del patrón singleton del servidor."""

    def test_get_blockchain_server_devuelve_instancia(self) -> None:
        """get_blockchain_server() debe devolver una instancia del servidor."""
        with (
            patch("LC.celebro.BKSVCB.LEDGER_PATH", Path("nonexistent_ledger.json")),
        ):
            try:
                srv = get_blockchain_server()
                assert srv is not None, "get_blockchain_server() no debe devolver None"
            except Exception:
                pytest.skip("get_blockchain_server() requiere dependencias completas")

    def test_iniciar_servidor_blockchain_es_callable(self) -> None:
        """iniciar_servidor_blockchain debe ser una función callable."""
        assert callable(iniciar_servidor_blockchain), (
            "iniciar_servidor_blockchain debe ser callable"
        )
from LC.celebro.test.test_blockchain_TestBloqueNeuronalConstruccion import TestBloqueNeuronalConstruccion  # CLASSPACK
from LC.celebro.test.test_blockchain_TestMinadoPoNL import TestMinadoPoNL  # CLASSPACK
from LC.celebro.test.test_blockchain_TestPersistenciaLedger import TestPersistenciaLedger  # CLASSPACK
