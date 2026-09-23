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

