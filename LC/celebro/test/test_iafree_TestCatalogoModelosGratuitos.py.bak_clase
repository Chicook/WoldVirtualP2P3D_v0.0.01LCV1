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

