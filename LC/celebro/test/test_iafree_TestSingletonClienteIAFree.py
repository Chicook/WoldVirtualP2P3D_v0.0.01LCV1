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

