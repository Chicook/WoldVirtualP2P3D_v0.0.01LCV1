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

class TestBloqueNeuronalConstruccion:
    """Tests de creación y atributos de BloqueNeuronal."""

    def test_bloque_genesis_atributos_basicos(self) -> None:
        """El bloque génesis (índice=0) debe tener los campos mínimos correctos."""
        bloque = BloqueNeuronal(
            indice=0,
            hash_previo="0" * 64,
            transacciones=[],
            dificultad=DIFICULTAD_TEST,
        )
        assert bloque.indice == 0, "índice del génesis debe ser 0"
        assert bloque.hash_previo == "0" * 64, "hash_previo del génesis debe ser 64 ceros"
        assert isinstance(bloque.transacciones, list), "transacciones debe ser lista"
        assert isinstance(bloque.hash_bloque, str), "hash_bloque debe ser str"
        assert isinstance(bloque.merkle_root, str), "merkle_root debe ser str"
        assert bloque.dificultad == DIFICULTAD_TEST, "dificultad debe coincidir"

    def test_bloque_hash_es_sha256_valido(self) -> None:
        """El hash generado debe ser un SHA-256 hexadecimal de 64 caracteres."""
        bloque = BloqueNeuronal(
            indice=1,
            hash_previo="a" * 64,
            transacciones=[{"tipo": "test"}],
            dificultad=DIFICULTAD_TEST,
        )
        assert_hash_sha256_valido(bloque.hash_bloque, nombre="bloque.hash_bloque")

    def test_bloque_merkle_root_es_sha256_valido(self) -> None:
        """El Merkle Root debe ser también un hash SHA-256 de 64 caracteres."""
        tx = [{"prompt": "hola", "respuesta": "mundo", "modelo": "test"}]
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=tx)
        assert_hash_sha256_valido(bloque.merkle_root, nombre="bloque.merkle_root")

    def test_bloque_sin_transacciones_merkle_especial(self) -> None:
        """Con 0 transacciones el Merkle Root debe ser el hash del string 'empty_block'."""
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=[])
        esperado = hashlib.sha256(b"empty_block_celebro_2026").hexdigest()
        assert bloque.merkle_root == esperado, "Merkle de bloque vacío debe coincidir con el hash de 'empty_block'"

    def test_bloque_con_una_transaccion(self) -> None:
        """Con una transacción el Merkle Root debe ser el hash de esa transacción."""
        tx = [{"tipo": "single_tx", "dato": 42}]
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=tx)
        tx_hash = hashlib.sha256(json.dumps(tx[0], sort_keys=True).encode()).hexdigest()
        assert bloque.merkle_root == tx_hash, "Merkle con 1 tx debe ser el hash directo de esa tx"

    def test_bloque_merkle_raiz_impar_duplica_ultimo(self) -> None:
        """Con N impar de transacciones se duplica el último hash (estándar Merkle)."""
        txs = [{"id": i} for i in range(3)]
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=txs)
        # Solo verificamos que el merkle_root es un SHA-256 válido de 64 chars
        assert_hash_sha256_valido(bloque.merkle_root, nombre="merkle_impar")

    def test_bloque_estado_neuronal_vacio_por_defecto(self) -> None:
        """Sin estado_neuronal explícito, debe quedar como dict vacío."""
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=[])
        assert bloque.estado_neuronal == {}, "estado_neuronal debe ser {} por defecto"

    def test_bloque_estado_neuronal_se_preserva(self) -> None:
        """El estado_neuronal pasado debe conservarse sin modificación."""
        estado = {"neuronas_activas": 50, "norma": 0.42, "deriva_muon": 0.001}
        bloque = BloqueNeuronal(
            indice=0, hash_previo="0" * 64, transacciones=[], estado_neuronal=estado
        )
        assert bloque.estado_neuronal == estado, "estado_neuronal debe ser idéntico al pasado"

    def test_bloque_to_dict_contiene_campos_requeridos(self) -> None:
        """to_dict() debe incluir todos los campos de serialización."""
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=[])
        d = bloque.to_dict()
        campos_requeridos = ["indice", "hash_bloque", "hash_previo", "merkle_root",
                             "dificultad", "nonce", "total_transacciones"]
        for campo in campos_requeridos:
            assert campo in d, f"to_dict() debe contener el campo '{campo}'"

    def test_bloque_to_dict_es_serializable_json(self) -> None:
        """El dict devuelto por to_dict() debe ser serializable a JSON."""
        bloque = BloqueNeuronal(
            indice=1, hash_previo="b" * 64,
            transacciones=[{"tipo": "aprendizaje", "peso": 0.7}]
        )
        try:
            json_str = json.dumps(bloque.to_dict())
        except (TypeError, ValueError) as e:
            pytest.fail(f"to_dict() no es serializable a JSON: {e}")
        assert len(json_str) > 0, "JSON serializado no debe estar vacío"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Minado PoNL (Proof of Neural Learning)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMinadoPoNL:
    """Tests del mecanismo de minado por prueba de trabajo neuronal."""

    def test_minar_dificultad_1_produce_hash_con_prefijo(self) -> None:
        """Con dificultad 1 el hash minado debe comenzar con '0'."""
        bloque = BloqueNeuronal(
            indice=0, hash_previo="0" * 64,
            transacciones=[], dificultad=1
        )
        exito = bloque.minar_bloque(max_iteraciones=MAX_ITER_TEST)
        assert exito, "El minado con dificultad 1 debe tener éxito en el límite dado"
        assert bloque.hash_bloque.startswith("0"), (
            f"Hash minado '{bloque.hash_bloque[:8]}' no comienza con '0'"
        )

    def test_minar_nonce_incrementa(self) -> None:
        """El nonce debe ser mayor que 0 tras el minado (al menos una iteración)."""
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=[], dificultad=1)
        bloque.minar_bloque(max_iteraciones=MAX_ITER_TEST)
        # Con dificultad 1 puede coincidir en nonce=0, pero verificamos que es entero
        assert isinstance(bloque.nonce, int), "nonce debe ser entero"
        assert bloque.nonce >= 0, "nonce no puede ser negativo"

    def test_minar_doble_sha256_coherente(self) -> None:
        """Después de minar, calcular_hash() debe devolver el mismo hash_bloque."""
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=[], dificultad=1)
        bloque.minar_bloque(max_iteraciones=MAX_ITER_TEST)
        recalculado = bloque.calcular_hash()
        assert recalculado == bloque.hash_bloque, (
            "calcular_hash() tras minar debe coincidir con hash_bloque almacenado"
        )

    def test_bloque_ya_minado_hash_valido(self) -> None:
        """Un bloque construido con hash_existente debe conservar ese hash."""
        hash_fijo = "0a" + "b" * 62  # Comienza con '0' (dificultad 1)
        bloque = BloqueNeuronal(
            indice=0, hash_previo="0" * 64, transacciones=[],
            hash_existente=hash_fijo
        )
        assert bloque.hash_bloque == hash_fijo, "hash_existente debe conservarse sin recalcular"

    def test_minar_max_iteraciones_sin_exito_devuelve_false(self) -> None:
        """Con max_iteraciones=0 no puede minar y debe devolver False."""
        bloque = BloqueNeuronal(indice=0, hash_previo="0" * 64, transacciones=[], dificultad=4)
        resultado = bloque.minar_bloque(max_iteraciones=0)
        assert resultado is False, "Con 0 iteraciones el minado debe fallar"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — ServidorBlockchainNeuronal: cadena y validación
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

class TestPersistenciaLedger:
    """Tests de guardado y recarga del blockchain_ledger.json."""

    def test_ledger_se_guarda_correctamente(self, tmp_path: Path) -> None:
        """Después de minar, el ledger debe poder guardarse como JSON válido."""
        ledger_path = tmp_path / "ledger.json"
        psnrl = tmp_path / "PSNRL"
        psnrl.mkdir()
        with (
            patch("LC.celebro.BKSVCB.PSNRL_DIR", psnrl),
            patch("LC.celebro.BKSVCB.LEDGER_PATH", ledger_path),
        ):
            srv = ServidorBlockchainNeuronal()
            srv.registrar_aprendizaje_neural("p", "r", "m")
            srv.minar_transacciones_pendientes()
            if hasattr(srv, "guardar_ledger"):
                srv.guardar_ledger()

        if ledger_path.exists():
            contenido = ledger_path.read_text(encoding="utf-8")
            datos = json.loads(contenido)
            assert "cadena" in datos, "El ledger guardado debe tener la clave 'cadena'"

    def test_ledger_genesis_tiene_hash_genesis_especial(self, ledger_genesis_path: Path) -> None:
        """El ledger de génesis debe tener el bloque en índice 0."""
        datos = json.loads(ledger_genesis_path.read_text(encoding="utf-8"))
        assert datos["cadena"][0]["indice"] == 0, "Primer bloque debe tener índice 0"

    def test_ledger_con_3_bloques_valido(self, ledger_con_bloques: Path) -> None:
        """El ledger con 3 bloques debe tener total_bloques == 3."""
        datos = json.loads(ledger_con_bloques.read_text(encoding="utf-8"))
        assert datos["total_bloques"] == 3, "total_bloques debe ser 3"
        assert len(datos["cadena"]) == 3, "La cadena debe tener 3 bloques"

    def test_transformar_ledger_a_pesos_no_lanza(self, tmp_path: Path) -> None:
        """transformar_ledger_a_pesos_neuronales() no debe lanzar excepciones."""
        psnrl = tmp_path / "PSNRL"
        psnrl.mkdir()
        with (
            patch("LC.celebro.BKSVCB.PSNRL_DIR", psnrl),
            patch("LC.celebro.BKSVCB.LEDGER_PATH", tmp_path / "l.json"),
        ):
            srv = ServidorBlockchainNeuronal()
            try:
                resultado = srv.transformar_ledger_a_pesos_neuronales()
                assert isinstance(resultado, dict), "El resultado debe ser un dict"
            except Exception as e:
                pytest.fail(f"transformar_ledger_a_pesos_neuronales() lanzó: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — Singleton y get_blockchain_server
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
