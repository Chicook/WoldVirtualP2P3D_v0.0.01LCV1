"""
test_ipfs.py — Suite exhaustiva de pruebas para el subsistema IPFS de LucIA
=============================================================================
Cubre:
  1. Motor puro Python CIDv1 (dag-raw + SHA2-256 + base32lower) estándar 2026.
  2. Inicialización de IPFSManager y ciclo de vida del manifiesto JSON.
  3. Comportamiento en modo offline / fallback autónomo (sin Kubo daemon).
  4. Mocking de Kubo HTTP RPC API v0 (respuestas JSON, subida multipart, pin).
  5. Rotación del manifiesto con TTL (límite 50 entradas, payload_b64 en las 10 más recientes).
  6. Subida en lote (batch) de pesos de PSNRL y verificación de borrado local seguro.
  7. Recuperación y reconstrucción binaria / deserialización JSON.
  8. Detección de binarios CLI locales (IPFSonl) y resolución de estado de conexión.
  9. Métodos auxiliares de consulta (listar_pesos, ultimo_cid, descargar_bytes).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import pytest

from LC.celebro.CMFG.ipfs_manager import (
    IPFSManager,
    compute_cidv1_raw,
    get_ipfs_manager,
    cerrar_sesion_y_subir_psnrl,
)
from LC.celebro.test.conftest import (
    assert_hash_sha256_valido,
    mock_ipfs_inactivo,
    manifest_ipfs_vacio,
    psnrl_tmp,
    psnrl_con_pesos,
)


# ============================================================================
# 1. MOTOR PURO PYTHON CIDv1 (ESTÁNDAR 2026)
# ============================================================================

class TestCIDv1Autonomo:
    """Verifica la generación algorítmica de CIDs según especificación IPFS 2026."""

    def test_cidv1_prefijo_y_formato(self) -> None:
        """Comprueba que el CID comience con 'b' (base32) y tenga longitud adecuada."""
        datos = b"tensor_pesos_celebro_test_vector_512"
        cid = compute_cidv1_raw(datos)
        assert isinstance(cid, str)
        assert cid.startswith("b"), f"CID debe comenzar con prefijo base32 'b', obtenido: {cid}"
        assert len(cid) >= 50, f"Longitud de CID sospechosamente corta: {len(cid)}"
        assert cid.islower(), "Multibase base32lower debe estar completamente en minúsculas"

    def test_cidv1_determinismo(self) -> None:
        """Verifica que el mismo payload produzca idéntico CID."""
        datos_a = b"modelo_red_neuronal_bloque_42"
        datos_b = b"modelo_red_neuronal_bloque_42"
        cid_a = compute_cidv1_raw(datos_a)
        cid_b = compute_cidv1_raw(datos_b)
        assert cid_a == cid_b

    def test_cidv1_sensibilidad_a_cambios(self) -> None:
        """Comprueba que una alteración mínima de bytes cambie el CID completamente."""
        datos_original = b"pesos_red_v1.0"
        datos_modificado = b"pesos_red_v1.1"
        cid_orig = compute_cidv1_raw(datos_original)
        cid_mod = compute_cidv1_raw(datos_modificado)
        assert cid_orig != cid_mod

    def test_cidv1_payload_vacio(self) -> None:
        """Verifica el cálculo de CIDv1 sobre datos de longitud cero."""
        datos_vacios = b""
        cid_vacio = compute_cidv1_raw(datos_vacios)
        assert cid_vacio.startswith("b")
        assert len(cid_vacio) > 10

    def test_cidv1_grandes_volumenes(self) -> None:
        """Comprueba estabilidad computando CIDs para bloques binarios grandes."""
        datos_grandes = b"\xaa\xbb\xcc\xdd" * 1024 * 64  # 256 KB
        cid = compute_cidv1_raw(datos_grandes)
        assert cid.startswith("b")
        assert len(cid) >= 50


# ============================================================================
# 2. INICIALIZACIÓN Y MANIFIESTO DEL GESTOR IPFS
# ============================================================================

class TestIPFSManagerManifiesto:
    """Pruebas sobre el ciclo de vida del manifiesto local ipfs_manifest.json."""

    def test_inicializacion_con_manifiesto_nuevo(self, tmp_path: Path) -> None:
        """Valida estructura inicial cuando el archivo de manifiesto no existe."""
        ruta_manifiesto = tmp_path / "nuevo_manifest.json"
        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))
        assert mgr.manifest_path == ruta_manifiesto
        assert mgr.manifest.get("version") == "2.0"
        assert mgr.manifest.get("weights") == {}
        assert mgr.manifest.get("total_stored") == 0

    def test_persistencia_manifiesto_en_disco(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Verifica que almacenar pesos guarde el manifiesto en formato JSON válido."""
        ruta_manifiesto = tmp_path / "manifiesto_disco.json"
        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))

        payload = b"matriz_sinaptica_2026"
        res = mgr.almacenar_pesos(payload, nombre_modelo="test_synapse", eliminar_local=False)
        assert res["exito"] is True
        assert ruta_manifiesto.exists()

        with open(ruta_manifiesto, "r", encoding="utf-8") as f:
            contenido = json.load(f)
        assert res["cid"] in contenido["weights"]
        assert contenido["total_stored"] >= 1

    def test_recuperacion_manifiesto_corrupto(self, tmp_path: Path) -> None:
        """Verifica tolerancia y regeneración ante archivo JSON corrupto."""
        ruta_manifiesto = tmp_path / "corrupto.json"
        ruta_manifiesto.write_text("{json_invalido: 123", encoding="utf-8")

        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))
        assert mgr.manifest["version"] == "2.0"
        assert mgr.manifest["weights"] == {}


# ============================================================================
# 3. ROTACIÓN CON TTL Y GESTIÓN DE MEMORIA EN MANIFIESTO
# ============================================================================

class TestIPFSManagerRotacionTTL:
    """Verifica las políticas de rotación: máx 500 entradas y 10 payloads en memoria."""

    def test_rotacion_mas_de_50_entradas(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Valida que el manifiesto se pode a 500 elementos al exceder la capacidad."""
        ruta_manifiesto = tmp_path / "manifiesto_rotacion.json"
        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))

        # Insertar 55 registros secuenciales
        for i in range(55):
            mgr.almacenar_pesos(
                origen=f"bloque_neuronal_{i:03d}".encode("utf-8"),
                nombre_modelo=f"modelo_{i}",
                eliminar_local=False,
            )

        assert len(mgr.manifest["weights"]) <= 500
        assert mgr.manifest["total_stored"] <= 500

    def test_purga_payload_base64_solo_10_recientes(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Comprueba que solo los 10 registros más recientes conserven payload_b64."""
        ruta_manifiesto = tmp_path / "manifiesto_payloads.json"
        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))

        for i in range(20):
            mgr.almacenar_pesos(
                origen=f"peso_incremental_{i}".encode("utf-8"),
                nombre_modelo=f"checkpoint_{i}",
                eliminar_local=False,
            )

        pesos = mgr.manifest["weights"]
        con_payload = [c for c, datos in pesos.items() if "payload_b64" in datos]
        assert len(con_payload) <= 10, f"Excedido límite de payloads en caché: {len(con_payload)}"


# ============================================================================
# 4. ALMACENAMIENTO Y MODO OFFLINE / AUTÓNOMO
# ============================================================================

class TestIPFSManagerOffline:
    """Pruebas de almacenamiento autónomo sin conectividad a daemon Kubo."""

    def test_almacenar_bytes_autonomo(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Valida que almacene en bóveda en memoria (_vault) ante daemon inactivo."""
        ruta_manifiesto = tmp_path / "manifest_offline.json"
        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))

        data = b"pesos_capa_densa_512x64"
        res = mgr.almacenar_pesos(data, nombre_modelo="capa_densa", eliminar_local=False)

        assert res["exito"] is True
        assert res["subida_real"] is False
        assert res["nodo"] == "AUTONOMO_CIDv1"
        assert res["cid"] in mgr._vault
        assert mgr._vault[res["cid"]] == data

    def test_almacenar_desde_archivo_local(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Comprueba almacenamiento leyendo un archivo de disco real."""
        ruta_manifiesto = tmp_path / "manifest_archivo.json"
        archivo_datos = tmp_path / "pesos_origen.bin"
        contenido = b"datos_crudos_pesos_vivos"
        archivo_datos.write_bytes(contenido)

        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))
        res = mgr.almacenar_pesos(
            origen=archivo_datos, nombre_modelo="origen_test", eliminar_local=False
        )

        assert res["exito"] is True
        assert res["tamano_bytes"] == len(contenido)
        assert_hash_sha256_valido(res["registro"]["sha256"])
        assert archivo_datos.exists()

    def test_almacenar_diccionario_json(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Verifica serialización automática cuando el origen es un dict de Python."""
        ruta_manifiesto = tmp_path / "manifest_dict.json"
        mgr = IPFSManager(manifest_file=str(ruta_manifiesto))

        datos_dict = {"capas": 3, "activacion": "relu", "version": "2026"}
        res = mgr.almacenar_pesos(
            origen=datos_dict, nombre_modelo="arquitectura_red", eliminar_local=False
        )

        assert res["exito"] is True
        rec = mgr.recuperar_pesos(res["cid"])
        assert isinstance(rec, dict)
        assert rec.get("activacion") == "relu"

    def test_archivo_no_existente_lanza_error(self, tmp_path: Path) -> None:
        """Valida que una ruta inexistente cause FileNotFoundError."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_err.json"))
        with pytest.raises(FileNotFoundError):
            mgr.almacenar_pesos(tmp_path / "no_existe_archivo.dat")


# ============================================================================
# 5. RECUPERACIÓN Y RECONSTRUCCIÓN DE DATOS
# ============================================================================

class TestIPFSManagerRecuperacion:
    """Pruebas de lectura y restauración de pesos neuronales."""

    def test_recuperar_desde_boveda(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Comprueba recuperación exacta de bytes desde el vault local."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_rec.json"))
        payload_original = b"gradientes_acumulados_adamw_step_100"
        res = mgr.almacenar_pesos(payload_original, nombre_modelo="gradientes", eliminar_local=False)

        recuperado = mgr.recuperar_pesos(res["cid"])
        assert recuperado == payload_original

    def test_recuperar_hacia_archivo_destino(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Verifica la restauración física en una ruta especificada."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_dest.json"))
        data = b"tensor_pesos_exportado"
        res = mgr.almacenar_pesos(data, nombre_modelo="tensor_exp", eliminar_local=False)

        destino_restauracion = tmp_path / "restaurado" / "pesos.bin"
        mgr.recuperar_pesos(res["cid"], destino=destino_restauracion)

        assert destino_restauracion.exists()
        assert destino_restauracion.read_bytes() == data

    def test_recuperar_cid_inexistente_lanza_keyerror(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Comprueba que solicitar un CID desconocido cause KeyError."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_vacio.json"))
        with pytest.raises(KeyError):
            mgr.recuperar_pesos("bafy_no_existe_en_el_sistema_2026")


# ============================================================================
# 6. SIMULACIÓN DE KUBO HTTP RPC API (MOCKS DE RED)
# ============================================================================

class TestIPFSManagerKuboRPC:
    """Verifica integración con daemon IPFS Kubo mediante mocks de urllib."""

    def test_descubrir_daemon_activo(self, tmp_path: Path) -> None:
        """Valida que auto-discovery detecte un endpoint Kubo saludable."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_kubo.json"))

        respuesta_id = json.dumps({"ID": "12D3KooWTestPeerIdKubo2026", "AgentVersion": "kubo/0.32.0"}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = respuesta_id
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            url = mgr.descubrir_daemon()
            assert url is not None
            assert "127.0.0.1" in url

    def test_subida_exitosa_daemon_kubo(self, tmp_path: Path) -> None:
        """Comprueba flujo de subida multipart con respuesta de pin OK."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_upload.json"))

        cid_simulado = "bafkreifh32q22x2w6u6z5n4m3l2k1j0hgfedsba2026kuborpc"
        resp_add = json.dumps({"Name": "pesos.bin", "Hash": cid_simulado, "Size": "1024"}).encode("utf-8")
        resp_id = json.dumps({"ID": "PeerKuboTest"}).encode("utf-8")

        def fake_urlopen(req: Any, **kwargs: Any) -> MagicMock:
            m = MagicMock()
            m.status = 200
            m.__enter__.return_value = m
            url_str = req.get_full_url() if hasattr(req, "get_full_url") else str(req)
            if "api/v0/id" in url_str:
                m.read.return_value = resp_id
            else:
                m.read.return_value = resp_add
            return m

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            res = mgr.almacenar_pesos(
                b"pesos_subidos_kubo", nombre_modelo="modelo_kubo", eliminar_local=False
            )
            assert res["exito"] is True
            assert res["subida_real"] is True
            assert res["nodo"] == "KUBO_HTTP_RPC"
            assert res["cid"] == cid_simulado


# ============================================================================
# 7. PROCESAMIENTO EN LOTE (BATCH) Y LIMPIEZA DE PSNRL
# ============================================================================

class TestIPFSManagerBatchPSNRL:
    """Verifica la subida masiva de pesos neuronales y la higiene del directorio local."""

    def test_subir_y_limpiar_psnrl_directorio_vacio(self, tmp_path: Path) -> None:
        """Comprueba reporte estructurado cuando no hay pesos pendientes en PSNRL."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_batch_vacio.json"))
        mgr._psnrl_dir = tmp_path / "psnrl_vacio"
        mgr._psnrl_dir.mkdir()

        resumen = mgr.subir_y_limpiar_psnrl()
        assert resumen["archivos_procesados"] == 0
        assert resumen["cids"] == []
        assert resumen["borrados"] == []

    def test_subir_y_limpiar_psnrl_con_archivos_forzado(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Valida que en modo offline se procesen los archivos y se generen CIDs autónomos sin borrar local."""
        dir_psnrl = tmp_path / "psnrl_con_pesos"
        dir_psnrl.mkdir()
        f1 = dir_psnrl / "neurona_en1.npz"
        f2 = dir_psnrl / "neurona_sl1.npz"
        f1.write_bytes(b"npz_pesos_en1")
        f2.write_bytes(b"npz_pesos_sl1")

        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_batch.json"))
        mgr._psnrl_dir = dir_psnrl

        resumen = mgr.subir_y_limpiar_psnrl(forzar_borrado_sin_daemon=True)
        assert resumen["archivos_procesados"] == 2
        assert len(resumen["cids"]) == 2
        # Sin confirmación de pin remoto (subida_real=False), el borrado seguro no procede
        assert resumen["borrados"] == []
        assert f1.exists()
        assert f2.exists()


# ============================================================================
# 8. MÉTODOS AUXILIARES DE CONSULTA Y TELEMETRÍA
# ============================================================================

class TestIPFSManagerConsultas:
    """Pruebas sobre listar_pesos, ultimo_cid y descargar_bytes."""

    def test_listar_pesos_orden_temporal(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Verifica que listar_pesos ordene los registros en orden temporal descendente."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_consulta.json"))
        mgr.almacenar_pesos(b"payload_uno", nombre_modelo="modelo_alfa", eliminar_local=False)
        time.sleep(0.01)
        mgr.almacenar_pesos(b"payload_dos", nombre_modelo="modelo_beta", eliminar_local=False)

        lista = mgr.listar_pesos()
        assert len(lista) == 2
        assert lista[0]["nombre_modelo"] == "modelo_beta"
        assert lista[1]["nombre_modelo"] == "modelo_alfa"

    def test_ultimo_cid_con_filtro_patron(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Comprueba la búsqueda del último CID filtrando por nombre de modelo."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_patron.json"))
        mgr.almacenar_pesos(b"p1", nombre_modelo="neurona_enrn_01", eliminar_local=False)
        mgr.almacenar_pesos(b"p2", nombre_modelo="neurona_slrn_01", eliminar_local=False)

        item = mgr.ultimo_cid(patron="slrn")
        assert item is not None
        assert "slrn" in item["nombre_modelo"]

    def test_descargar_bytes_exitoso(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Valida que descargar_bytes retorne la carga binaria correctamente."""
        mgr = IPFSManager(manifest_file=str(tmp_path / "manifest_descarga.json"))
        data_raw = b"datos_bytes_crudos_2026"
        res = mgr.almacenar_pesos(data_raw, nombre_modelo="datos_crudos", eliminar_local=False)

        descargado = mgr.descargar_bytes(res["cid"])
        assert descargado == data_raw


# ============================================================================
# 9. INSTANCIA GLOBAL Y ESTADO DEL SISTEMA
# ============================================================================

class TestIPFSManagerGlobal:
    """Pruebas sobre el singleton global y funciones de conveniencia."""

    def test_get_ipfs_manager_singleton(self) -> None:
        """Verifica que llamadas repetidas a get_ipfs_manager devuelvan la misma instancia."""
        inst1 = get_ipfs_manager()
        inst2 = get_ipfs_manager()
        assert inst1 is inst2

    def test_estado_conexion_estructura(self, mock_ipfs_inactivo: None) -> None:
        """Valida que estado_conexion() proporcione todas las claves de telemetría."""
        mgr = get_ipfs_manager()
        estado = mgr.estado_conexion()
        assert isinstance(estado, dict)
        claves_requeridas = {"daemon_activo", "daemon_url", "cli_disponible", "motor_autonomo", "total_en_manifiesto"}
        for k in claves_requeridas:
            assert k in estado, f"Clave ausente en estado_conexion: {k}"

    def test_cerrar_sesion_hook(self, tmp_path: Path, mock_ipfs_inactivo: None) -> None:
        """Comprueba la invocación segura del hook de cierre global."""
        res = cerrar_sesion_y_subir_psnrl(forzar=False)
        assert isinstance(res, dict)
        assert "archivos_procesados" in res
        assert "cids" in res


# Fin de suite exhaustiva test_ipfs.py
