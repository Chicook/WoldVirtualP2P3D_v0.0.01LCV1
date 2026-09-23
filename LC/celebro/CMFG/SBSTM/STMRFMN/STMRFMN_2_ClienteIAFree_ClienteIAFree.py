class ClienteIAFree:
    def __init__(self):
        self.gestor = self._GestorModelos()
        self._modelo_activo_id: str = "lucia-default-v1"

    class _GestorModelos:
        def __init__(self):
            self._modelos = [
                {"id": "lucia-default-v1", "nombre": "LucIA Default", "parametros": "7B"},
                {"id": "lucia-flash-v2", "nombre": "LucIA Flash", "parametros": "3B"},
            ]

        def obtener_modelo_activo(self) -> Dict[str, str]:
            return next((m for m in self._modelos if m["id"] == "lucia-default-v1"), self._modelos[0])

        def listar_modelos(self) -> List[Dict[str, str]]:
            return list(self._modelos)

    def consultar(self, prompt: str, max_tokens: int = 256) -> str:
        logger.info(f"[IA Free] Consulta: {prompt[:60]}...")
        return f"[Respuesta simulada a: '{prompt[:40]}...']"

    def cambiar_modelo(self, modelo_id: str) -> bool:
        modelos = self.gestor.listar_modelos()
        if any(m["id"] == modelo_id for m in modelos):
            self._modelo_activo_id = modelo_id
            return True
        return False


# ═══════════════════════════════════════════════════════════════════
# CLASS 6: IPFS - cliente ligero para almacenamiento descentralizado
# ═══════════════════════════════════════════════════════════════════
