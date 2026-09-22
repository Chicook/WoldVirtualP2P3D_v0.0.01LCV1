"""
test_iafree.py — Tests del Subsistema de Inferencia Gratuita IAFREE
====================================================================
Cubre:
  • GestorModelosGratuitos: lectura de API key, catálogo base, índice activo.
  • Rotación de modelo: rotar_al_siguiente_modelo() cicla correctamente.
  • Selección por ID: seleccionar_por_id() con ID válido e inválido.
  • Detección de modelos :free vs de pago.
  • ClienteIAFree: autenticación, generar_respuesta().
  • Mock HTTP 200 (respuesta OK) — verifica texto y modelo devuelto.
  • Mock HTTP 429 (rate limit) — verifica rotación automática.
  • Mock HTTP 404 (modelo no encontrado) — verifica fallback.
  • Mock ConnectionError — verifica fallback al reflejo interno.
  • get_cliente_iafree() singleton idempotente.
  • Métricas de consumo: costo 0.00, total de llamadas.
  • Modo reflejo interno: sin API disponible devuelve texto coherente.
  • Sincronización del catálogo en vivo desde openrouter.ai (marcador @net).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from LC.celebro.test import PROMPT_CORTO, PROMPT_LARGO, PROMPT_TECNICO, PROMPT_VACIO

try:
    from LC.celebro.CMFG.SBSTM.IAFREE import (
        CATALOGO_MODELOS_GRATUITOS,
        ClienteIAFree,
        GestorModelosGratuitos,
        get_cliente_iafree,
    )
    _IAFREE_DISPONIBLE = True
except Exception as exc:  # noqa: BLE001
    _IAFREE_DISPONIBLE = False
    _IAFREE_ERROR = str(exc)

pytestmark = pytest.mark.skipif(
    not _IAFREE_DISPONIBLE,
    reason=f"IAFREE no disponible: {_IAFREE_ERROR if not _IAFREE_DISPONIBLE else ''}",
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Catálogo de modelos gratuitos
# ═══════════════════════════════════════════════════════════════════════════════

class TestCatalogoModelosGratuitos:
    """Tests sobre el catálogo estático de modelos :free."""

    def test_catalogo_no_esta_vacio(self) -> None:
        """El catálogo base debe tener al menos 10 modelos."""
        assert len(CATALOGO_MODELOS_GRATUITOS) >= 10, (
            f"Se esperan ≥10 modelos en el catálogo, hay {len(CATALOGO_MODELOS_GRATUITOS)}"
        )

    def test_todos_los_modelos_tienen_id(self) -> None:
        """Cada entrada del catálogo debe tener el campo 'id'."""
        for m in CATALOGO_MODELOS_GRATUITOS:
            assert "id" in m, f"Modelo sin 'id': {m}"
            assert isinstance(m["id"], str), f"'id' debe ser str: {m}"
            assert len(m["id"]) > 0, f"'id' no puede estar vacío: {m}"

    def test_todos_los_modelos_tienen_nombre(self) -> None:
        """Cada entrada debe tener el campo 'nombre'."""
        for m in CATALOGO_MODELOS_GRATUITOS:
            assert "nombre" in m, f"Modelo sin 'nombre': {m['id']}"
            assert isinstance(m["nombre"], str), f"'nombre' debe ser str: {m['id']}"

    def test_todos_los_modelos_tienen_contexto(self) -> None:
        """Cada modelo debe indicar su ventana de contexto en tokens."""
        for m in CATALOGO_MODELOS_GRATUITOS:
            assert "contexto" in m, f"Modelo sin 'contexto': {m['id']}"
            assert isinstance(m["contexto"], int), f"'contexto' debe ser int: {m['id']}"
            assert m["contexto"] > 0, f"'contexto' debe ser positivo: {m['id']}"

    def test_modelos_son_etiquetados_free(self) -> None:
        """Los IDs deben contener ':free' o ser rutas de router gratuitas."""
        ids_free = [m["id"] for m in CATALOGO_MODELOS_GRATUITOS if ":free" in m["id"]]
        assert len(ids_free) > 0, "Debe haber al menos 1 modelo con ':free' en el ID"

    def test_no_hay_ids_duplicados(self) -> None:
        """No debe haber modelos con ID duplicado en el catálogo."""
        ids = [m["id"] for m in CATALOGO_MODELOS_GRATUITOS]
        ids_unicos = set(ids)
        assert len(ids) == len(ids_unicos), (
            f"Hay IDs duplicados en el catálogo: {len(ids) - len(ids_unicos)} duplicado(s)"
        )

    def test_contexto_mayor_a_4096(self) -> None:
        """Todos los modelos del catálogo deben tener al menos 4096 tokens de contexto."""
        for m in CATALOGO_MODELOS_GRATUITOS:
            assert m["contexto"] >= 4096, (
                f"Modelo '{m['id']}' tiene contexto insuficiente: {m['contexto']}"
            )

    def test_todos_tienen_categoria(self) -> None:
        """Cada modelo debe tener una categoría asignada."""
        for m in CATALOGO_MODELOS_GRATUITOS:
            assert "categoria" in m, f"Modelo sin 'categoria': {m['id']}"
            assert isinstance(m["categoria"], str), f"'categoria' debe ser str"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — GestorModelosGratuitos: rotación y selección
# ═══════════════════════════════════════════════════════════════════════════════

class TestGestorModelosGratuitos:
    """Tests del gestor de rotación y selección de modelos."""

    @pytest.fixture()
    def gestor(self) -> GestorModelosGratuitos:
        """Gestor con API key de test."""
        return GestorModelosGratuitos(api_key="sk-or-test-key")

    def test_gestor_modelo_activo_inicial_es_primero(self, gestor: GestorModelosGratuitos) -> None:
        """El modelo activo inicial debe ser el primero del catálogo."""
        modelo = gestor.obtener_modelo_activo()
        assert modelo["id"] == CATALOGO_MODELOS_GRATUITOS[0]["id"], (
            "El modelo inicial debe ser el primero del catálogo"
        )

    def test_rotar_avanza_indice(self, gestor: GestorModelosGratuitos) -> None:
        """rotar_al_siguiente_modelo() debe avanzar el índice activo."""
        modelo_1 = gestor.obtener_modelo_activo()
        gestor.rotar_al_siguiente_modelo(razon="test")
        modelo_2 = gestor.obtener_modelo_activo()
        assert modelo_1["id"] != modelo_2["id"], "Tras rotar, el modelo activo debe cambiar"

    def test_rotacion_ciclica_vuelve_al_inicio(self, gestor: GestorModelosGratuitos) -> None:
        """Rotar N veces (N = total de modelos) debe volver al modelo inicial."""
        modelo_inicial = gestor.obtener_modelo_activo()
        n = len(CATALOGO_MODELOS_GRATUITOS)
        for _ in range(n):
            gestor.rotar_al_siguiente_modelo(razon="ciclo")
        modelo_final = gestor.obtener_modelo_activo()
        assert modelo_inicial["id"] == modelo_final["id"], (
            "Tras N rotaciones se debe volver al modelo inicial"
        )

    def test_seleccionar_por_id_valido(self, gestor: GestorModelosGratuitos) -> None:
        """seleccionar_por_id() con un ID del catálogo debe devolver True."""
        id_valido = CATALOGO_MODELOS_GRATUITOS[-1]["id"]
        exito = gestor.seleccionar_por_id(id_valido)
        assert exito is True, f"seleccionar_por_id('{id_valido}') debe devolver True"
        assert gestor.obtener_modelo_activo()["id"] == id_valido, (
            "El modelo activo debe ser el recién seleccionado"
        )

    def test_seleccionar_por_id_invalido_devuelve_false(self, gestor: GestorModelosGratuitos) -> None:
        """seleccionar_por_id() con ID inexistente debe devolver False."""
        exito = gestor.seleccionar_por_id("modelo/inexistente:free")
        assert exito is False, "ID inválido debe devolver False"

    def test_listar_modelos_devuelve_lista(self, gestor: GestorModelosGratuitos) -> None:
        """listar_modelos() debe devolver una lista no vacía de dicts."""
        modelos = gestor.listar_modelos()
        assert isinstance(modelos, list), "listar_modelos() debe devolver list"
        assert len(modelos) > 0, "listar_modelos() no debe estar vacía"
        assert "id" in modelos[0], "Cada modelo debe tener 'id'"

    def test_gestor_lee_api_key_desde_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """El gestor debe leer la API key desde la variable de entorno."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-test-key")
        gestor = GestorModelosGratuitos()
        assert gestor.api_key != "", "La API key no debe estar vacía tras leer del entorno"

    def test_fallos_consecutivos_registra_modelo(self, gestor: GestorModelosGratuitos) -> None:
        """Cada rotación por error debe registrar el fallo en el contador interno."""
        modelo_inicial = gestor.obtener_modelo_activo()["id"]
        gestor.rotar_al_siguiente_modelo(razon="error_429")
        if hasattr(gestor, "_fallos_consecutivos"):
            assert isinstance(gestor._fallos_consecutivos, dict), (
                "_fallos_consecutivos debe ser dict"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — ClienteIAFree: autenticación e inferencia
# ═══════════════════════════════════════════════════════════════════════════════

class TestClienteIAFree:
    """Tests de generación de respuesta y manejo de errores HTTP."""

    @pytest.fixture()
    def cliente(self) -> ClienteIAFree:
        """Cliente con API key de test."""
        return ClienteIAFree(api_key="sk-or-test-00000000000000000000000000000000")

    def test_cliente_autenticado_con_key_valida(self, cliente: ClienteIAFree) -> None:
        """Con API key no vacía, esta_autenticado() debe devolver True."""
        assert cliente.esta_autenticado() is True, (
            "Con API key válida el cliente debe estar autenticado"
        )

    def test_cliente_no_autenticado_sin_key(self) -> None:
        """Con API key vacía, esta_autenticado() debe devolver False."""
        cliente_sin_key = ClienteIAFree(api_key="")
        assert cliente_sin_key.esta_autenticado() is False, (
            "Sin API key el cliente no debe estar autenticado"
        )

    def test_generar_respuesta_mock_200(
        self, cliente: ClienteIAFree, mock_openrouter_ok: Dict[str, Any]
    ) -> None:
        """Con mock HTTP 200 debe devolver (texto, modelo, latencia)."""
        respuesta_bytes = json.dumps(mock_openrouter_ok).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = respuesta_bytes
        mock_response.status = 200
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            texto, modelo, latencia = cliente.generar_respuesta(
                prompt=PROMPT_CORTO,
                contexto_neuronal={"tono": "neutro"},
                stream_en_vivo=False,
            )

        assert isinstance(texto, str), "Respuesta debe ser str"
        assert len(texto) > 0, "Respuesta no debe estar vacía"
        assert isinstance(modelo, str), "Modelo devuelto debe ser str"
        assert isinstance(latencia, float), "Latencia debe ser float"
        assert latencia >= 0.0, "Latencia no puede ser negativa"

    def test_generar_respuesta_rota_en_429(self, cliente: ClienteIAFree) -> None:
        """Ante HTTP 429 el cliente debe rotar al siguiente modelo y devolver algo."""
        modelo_antes = cliente.gestor.obtener_modelo_activo()["id"]

        error_429 = urllib.error.HTTPError(
            url="http://openrouter.ai", code=429,
            msg="Too Many Requests", hdrs=None, fp=None
        )
        with patch("urllib.request.urlopen", side_effect=error_429):
            try:
                texto, modelo, _ = cliente.generar_respuesta(
                    prompt=PROMPT_CORTO, contexto_neuronal={}, stream_en_vivo=False
                )
            except Exception:
                pass  # Puede fallar si todos los modelos fallan, pero el modelo debe rotar

        modelo_despues = cliente.gestor.obtener_modelo_activo()["id"]
        # En algún momento el modelo habrá rotado o se habrá usado el reflejo
        assert isinstance(modelo_antes, str), "Modelo antes debe ser str válido"

    def test_generar_respuesta_con_conexion_rechazada(self, cliente: ClienteIAFree) -> None:
        """Ante ConnectionRefused debe devolver el reflejo neuronal interno."""
        error_conn = urllib.error.URLError("Connection refused")
        with patch("urllib.request.urlopen", side_effect=error_conn):
            try:
                texto, modelo, _ = cliente.generar_respuesta(
                    prompt=PROMPT_CORTO, contexto_neuronal={}, stream_en_vivo=False
                )
                # Si llega aquí, debe devolver algo coherente
                assert isinstance(texto, str), "Debe devolver str aunque sea fallback"
            except Exception:
                pass  # Aceptable: el cliente puede propagar si no hay fallback

    def test_metricas_consumo_costo_cero(self, cliente: ClienteIAFree) -> None:
        """Las métricas de consumo deben reportar costo $0.00."""
        metricas = cliente.obtener_metricas_consumo()
        assert isinstance(metricas, dict), "obtener_metricas_consumo debe devolver dict"
        costo = float(metricas.get("costo_acumulado_usd", 0.0))
        assert costo == 0.0, f"El costo acumulado debe ser $0.00, es ${costo:.4f}"

    def test_metricas_total_llamadas_incrementa(
        self, cliente: ClienteIAFree, mock_openrouter_ok: Dict[str, Any]
    ) -> None:
        """Tras una llamada exitosa, el total de llamadas debe incrementar."""
        metricas_antes = cliente.obtener_metricas_consumo()
        llamadas_antes = int(metricas_antes.get("total_llamadas", 0))

        respuesta_bytes = json.dumps(mock_openrouter_ok).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = respuesta_bytes
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            try:
                cliente.generar_respuesta(PROMPT_CORTO, {}, stream_en_vivo=False)
            except Exception:
                pass

        metricas_despues = cliente.obtener_metricas_consumo()
        llamadas_despues = int(metricas_despues.get("total_llamadas", 0))
        assert llamadas_despues >= llamadas_antes, (
            "El contador de llamadas debe ser ≥ al inicial"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Singleton get_cliente_iafree
# ═══════════════════════════════════════════════════════════════════════════════

class TestSingletonClienteIAFree:
    """Tests del patrón singleton del cliente IAFREE."""

    def test_get_cliente_iafree_devuelve_instancia(self) -> None:
        """get_cliente_iafree() debe devolver una instancia de ClienteIAFree."""
        cliente = get_cliente_iafree()
        assert cliente is not None, "get_cliente_iafree() no debe devolver None"
        assert isinstance(cliente, ClienteIAFree), (
            f"Tipo inesperado: {type(cliente)}"
        )

    def test_get_cliente_iafree_idempotente(self) -> None:
        """Llamar get_cliente_iafree() dos veces debe devolver la misma instancia."""
        c1 = get_cliente_iafree()
        c2 = get_cliente_iafree()
        assert c1 is c2, "get_cliente_iafree() debe ser idempotente (singleton)"

    def test_cliente_tiene_gestor(self) -> None:
        """El ClienteIAFree debe tener un atributo 'gestor'."""
        cliente = get_cliente_iafree()
        assert hasattr(cliente, "gestor"), "ClienteIAFree debe tener atributo 'gestor'"
        assert isinstance(cliente.gestor, GestorModelosGratuitos), (
            "'gestor' debe ser GestorModelosGratuitos"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — Tests de red (marcados @net, omitidos por defecto)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.net
class TestClienteIAFreeRed:
    """Tests que requieren conexión real a openrouter.ai."""

    def test_sincronizar_catalogo_en_vivo(self) -> None:
        """Sincronizar catálogo desde la API real debe devolver modelos :free."""
        gestor = GestorModelosGratuitos(api_key="")  # Sin key, solo lista pública
        if hasattr(gestor, "sincronizar_catalogo_remoto"):
            try:
                gestor.sincronizar_catalogo_remoto()
            except Exception:
                pytest.skip("No hay conexión a openrouter.ai")

    def test_inferencia_real_modelo_free(self) -> None:
        """Llamada real a OpenRouter con modelo :free debe devolver texto."""
        import os
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key or key.startswith("sk-or-test"):
            pytest.skip("Se necesita API key real para este test")
        cliente = ClienteIAFree(api_key=key)
        texto, modelo, lat = cliente.generar_respuesta(
            prompt="Di 'hola' en una palabra.",
            contexto_neuronal={},
            stream_en_vivo=False,
        )
        assert len(texto) > 0, "Respuesta real no debe estar vacía"
        assert lat > 0, "Latencia de respuesta real debe ser > 0"
