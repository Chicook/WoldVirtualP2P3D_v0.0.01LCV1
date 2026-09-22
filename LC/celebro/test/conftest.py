"""
conftest.py — Fixtures globales de la suite de tests LucIA (v0.0.01LCV1)
=========================================================================
Proporciona:
  • Configuración de sys.path para importaciones desde cualquier CWD.
  • Fixtures de entorno (variables de entorno seguras, tmp_path, ledger).
  • Mocks de red: OpenRouter HTTP, Ollama, daemon IPFS Kubo.
  • Vectores numpy reproducibles para tests de neuronas.
  • Instancias pre-construidas de los subsistemas principales.
  • Marcadores personalizados: slow, net, voice, gpu.
  • Helpers de aserción numpy para comparaciones de tensores.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# ─── RUTAS BASE ───────────────────────────────────────────────────────────────
TEST_DIR: Path    = Path(__file__).resolve().parent
CELEBRO_DIR: Path = TEST_DIR.parent
LC_DIR: Path      = CELEBRO_DIR.parent
ROOT_DIR: Path    = LC_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ─── MARCADORES PERSONALIZADOS ────────────────────────────────────────────────
def pytest_configure(config: pytest.Config) -> None:
    """Registra marcadores custom para evitar warnings de pytest."""
    config.addinivalue_line("markers", "slow: test que tarda más de 2 segundos")
    config.addinivalue_line("markers", "net: requiere conexión de red real")
    config.addinivalue_line("markers", "voice: requiere edge-tts / winmm instalado")
    config.addinivalue_line("markers", "gpu: requiere GPU/CUDA disponible")
    config.addinivalue_line("markers", "ipfs: requiere daemon IPFS Kubo corriendo")


# ─── FIXTURE: entorno de variables de entorno seguras ─────────────────────────
@pytest.fixture(autouse=True)
def entorno_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Inyecta automáticamente variables de entorno seguras en todos los tests.
    Evita que un .env real afecte los tests y previene llamadas de red accidentales.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-00000000000000000000000000000000")
    monkeypatch.setenv("LUCIA_SIMPLE", "1")
    monkeypatch.setenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setenv("LOG_LEVEL", "CRITICAL")
    monkeypatch.setenv("IPFS_API_URL", "http://127.0.0.1:5001")


# ─── FIXTURE: directorio PSNRL temporal ───────────────────────────────────────
@pytest.fixture()
def psnrl_tmp(tmp_path: Path) -> Path:
    """
    Directorio PSNRL aislado por test.
    Se crea en tmp_path para no contaminar el árbol real del proyecto.
    """
    psnrl = tmp_path / "PSNRL"
    psnrl.mkdir()
    return psnrl


# ─── FIXTURE: ledger JSON con bloque génesis ──────────────────────────────────
@pytest.fixture()
def ledger_genesis_path(tmp_path: Path) -> Path:
    """
    Devuelve la ruta a un ledger JSON con el bloque génesis ya inicializado.
    Útil para tests de blockchain sin depender del ledger real del proyecto.
    """
    genesis_tx = {
        "tipo": "genesis",
        "timestamp": 0.0,
        "datos": "WoldVirtualP2P3D Genesis Block — LucIA 2026",
    }
    genesis_block = {
        "indice": 0,
        "hash_bloque": "00" + "0" * 62,
        "hash_previo": "0" * 64,
        "merkle_root": hashlib.sha256(b"genesis").hexdigest(),
        "dificultad": 2,
        "nonce": 0,
        "total_transacciones": 1,
        "transacciones": [genesis_tx],
        "estado_neuronal": {"neuronas_activas": 0, "norma_acumulada": 0.0},
        "timestamp": 0.0,
    }
    ledger = {"cadena": [genesis_block], "total_bloques": 1}
    path = tmp_path / "blockchain_ledger.json"
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@pytest.fixture()
def ledger_con_bloques(tmp_path: Path) -> Path:
    """
    Ledger con 3 bloques encadenados para tests de validación de cadena.
    Los hashes son ficticios pero la estructura está completa.
    """
    bloques = []
    hash_prev = "0" * 64
    for i in range(3):
        raw = f"{i}:{hash_prev}:merkle{i}:2:{i * 7}".encode()
        h1 = hashlib.sha256(raw).digest()
        h2 = hashlib.sha256(h1).hexdigest()
        bloque = {
            "indice": i,
            "hash_bloque": h2,
            "hash_previo": hash_prev,
            "merkle_root": f"merkle{i}",
            "dificultad": 2,
            "nonce": i * 7,
            "total_transacciones": 1,
            "transacciones": [{"tipo": f"tx_{i}", "timestamp": float(i)}],
            "estado_neuronal": {"norma_acumulada": float(i) * 0.01},
            "timestamp": float(i),
        }
        bloques.append(bloque)
        hash_prev = h2
    path = tmp_path / "blockchain_ledger.json"
    path.write_text(json.dumps({"cadena": bloques, "total_bloques": 3}), encoding="utf-8")
    return path


# ─── FIXTURES: vectores numpy para tests neuronales ───────────────────────────
@pytest.fixture()
def vector_512() -> np.ndarray:
    """Vector float32 normalizado de dimensión 512 (semilla fija)."""
    rng = np.random.default_rng(42)
    v = rng.standard_normal(512).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


@pytest.fixture()
def vector_128() -> np.ndarray:
    """Vector float32 de dimensión 128."""
    rng = np.random.default_rng(7)
    return rng.standard_normal(128).astype(np.float32)


@pytest.fixture()
def vector_64() -> np.ndarray:
    """Vector float32 de dimensión 64 (neuronas pequeñas)."""
    rng = np.random.default_rng(13)
    return rng.standard_normal(64).astype(np.float32)


@pytest.fixture()
def matriz_2d_32x32() -> np.ndarray:
    """Matriz float32 de 32×32 para tests de ortogonalización NS-5."""
    rng = np.random.default_rng(55)
    return rng.standard_normal((32, 32)).astype(np.float32)


@pytest.fixture()
def gradiente_512() -> np.ndarray:
    """Gradiente float32 pequeño de dimensión 512 (lr~0.001)."""
    rng = np.random.default_rng(99)
    return rng.standard_normal(512).astype(np.float32) * 0.001


@pytest.fixture()
def batch_vectores(vector_512: np.ndarray) -> np.ndarray:
    """Batch de 8 vectores (8, 512) para tests de procesamiento por lotes."""
    rng = np.random.default_rng(321)
    noise = rng.standard_normal((8, 512)).astype(np.float32) * 0.1
    return np.tile(vector_512, (8, 1)) + noise


# ─── FIXTURE: mock respuesta HTTP de OpenRouter ───────────────────────────────
@pytest.fixture()
def mock_openrouter_ok() -> Dict[str, Any]:
    """Cuerpo JSON de respuesta exitosa de OpenRouter /chat/completions."""
    return {
        "id": "gen-test-abc123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen/qwen3.8-27b:free",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Esta es una respuesta de prueba del mock de OpenRouter.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
    }


@pytest.fixture()
def mock_openrouter_429() -> Dict[str, Any]:
    """Cuerpo JSON de respuesta de límite de tasa (HTTP 429)."""
    return {
        "error": {
            "message": "Rate limit exceeded. Please wait before retrying.",
            "type": "rate_limit_exceeded",
            "code": 429,
        }
    }


# ─── FIXTURE: mock del daemon IPFS Kubo no disponible ─────────────────────────
@pytest.fixture()
def mock_ipfs_inactivo():
    """Simula que el daemon Kubo está caído (ConnectionRefusedError)."""
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused — IPFS mock"),
    ):
        yield


# ─── FIXTURE: mock de Ollama no disponible ────────────────────────────────────
@pytest.fixture()
def mock_ollama_inactivo():
    """Simula que el servidor Ollama no está disponible."""
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("Connection refused — Ollama mock"),
    ):
        yield


# ─── FIXTURE: archivo IAlocal.json temporal ───────────────────────────────────
@pytest.fixture()
def ialocal_json(tmp_path: Path) -> Path:
    """IAlocal.json temporal con configuración de test."""
    config = {
        "provider": "ollama",
        "endpoint": "http://localhost:11434",
        "default_model": "cogito:3b",
        "stream": False,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 256,
        "models_available": ["cogito:3b", "gemma2:2b"],
    }
    path = tmp_path / "IAlocal.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


# ─── FIXTURE: manifest IPFS vacío ─────────────────────────────────────────────
@pytest.fixture()
def manifest_ipfs_vacio(tmp_path: Path) -> Path:
    """Manifiesto IPFS vacío para tests de IPFSManager."""
    manifest = {"version": "2.0", "weights": {}, "total_stored": 0}
    path = tmp_path / "ipfs_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


# ─── FIXTURE: archivos de pesos .npz en PSNRL ─────────────────────────────────
@pytest.fixture()
def psnrl_con_pesos(psnrl_tmp: Path) -> Path:
    """
    PSNRL con 3 archivos .npz de pesos simulados.
    Permite tests de lectura/subida batch a IPFS.
    """
    rng = np.random.default_rng(77)
    for i in range(1, 4):
        data = {
            "pesos": rng.standard_normal((64, 64)).astype(np.float32),
            "sesgo": rng.standard_normal(64).astype(np.float32),
            "iteracion": np.array([i * 100]),
        }
        np.savez_compressed(psnrl_tmp / f"neurona_{i:02d}.npz", **data)
    return psnrl_tmp


# ─── HELPERS DE ASERCIÓN NUMPY ────────────────────────────────────────────────
def assert_tensor_valido(
    arr: np.ndarray,
    esperado_shape: Optional[Tuple[int, ...]] = None,
    dtype_esperado: Optional[np.dtype] = None,
    nombre: str = "tensor",
) -> None:
    """
    Valida que un ndarray no tenga NaN/Inf y coincida con shape y dtype esperados.

    Parameters
    ----------
    arr            : ndarray a validar.
    esperado_shape : tupla de forma esperada o None para omitir.
    dtype_esperado : dtype esperado o None para omitir.
    nombre         : etiqueta para mensajes de error.
    """
    assert isinstance(arr, np.ndarray), f"{nombre} debe ser ndarray, es {type(arr)}"
    assert not np.any(np.isnan(arr)), f"{nombre} contiene NaN"
    assert not np.any(np.isinf(arr)), f"{nombre} contiene Inf"
    if esperado_shape is not None:
        assert arr.shape == esperado_shape, (
            f"{nombre}: shape esperado {esperado_shape}, obtenido {arr.shape}"
        )
    if dtype_esperado is not None:
        assert arr.dtype == dtype_esperado, (
            f"{nombre}: dtype esperado {dtype_esperado}, obtenido {arr.dtype}"
        )


def assert_norma_acotada(arr: np.ndarray, max_norma: float = 1e4, nombre: str = "tensor") -> None:
    """Valida que la norma de Frobenius del tensor esté dentro de un límite razonable."""
    norma = float(np.linalg.norm(arr))
    assert norma < max_norma, (
        f"{nombre}: norma {norma:.4f} excede el máximo permitido {max_norma}"
    )
    assert norma > 0.0, f"{nombre}: norma es cero, tensor vacío o nulo"


def assert_hash_sha256_valido(hash_str: str, nombre: str = "hash") -> None:
    """Valida que una cadena sea un hash SHA-256 hexadecimal de 64 caracteres."""
    assert isinstance(hash_str, str), f"{nombre} debe ser str"
    assert len(hash_str) == 64, f"{nombre}: longitud esperada 64, obtenida {len(hash_str)}"
    try:
        int(hash_str, 16)
    except ValueError:
        pytest.fail(f"{nombre}='{hash_str}' no es hexadecimal válido")


# ─── EXPORTACIONES ────────────────────────────────────────────────────────────
__all__: List[str] = [
    "TEST_DIR",
    "CELEBRO_DIR",
    "ROOT_DIR",
    "assert_tensor_valido",
    "assert_norma_acotada",
    "assert_hash_sha256_valido",
    "psnrl_tmp",
    "ledger_genesis_path",
    "ledger_con_bloques",
    "vector_512",
    "vector_128",
    "vector_64",
    "matriz_2d_32x32",
    "gradiente_512",
    "batch_vectores",
    "mock_openrouter_ok",
    "mock_openrouter_429",
    "mock_ipfs_inactivo",
    "mock_ollama_inactivo",
    "ialocal_json",
    "manifest_ipfs_vacio",
    "psnrl_con_pesos",
    "entorno_test",
]
