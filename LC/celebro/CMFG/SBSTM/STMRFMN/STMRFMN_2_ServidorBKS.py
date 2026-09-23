class ServidorBKS:
    def __init__(self):
        self.cadena: List[Dict[str, Any]] = []
        self._ultimo_hash: str = "0" * 64

    def _hash_bloque(self, bloque: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(bloque, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def agregar(self, datos: Dict[str, Any]) -> str:
        bloque = {"index": len(self.cadena), "timestamp": time.time(), "datos": datos, "prev_hash": self._ultimo_hash}
        bloque["hash"] = self._hash_bloque(bloque)
        self._ultimo_hash = bloque["hash"]
        self.cadena.append(bloque)
        return bloque["hash"]

    def obtener_bloque(self, index: int) -> Optional[Dict[str, Any]]:
        if 0 <= index < len(self.cadena):
            return self.cadena[index]
        return None

    def profundidad(self) -> int:
        return len(self.cadena)

    def validar_cadena(self) -> Tuple[bool, str]:
        for i in range(1, len(self.cadena)):
            a, b = self.cadena[i - 1], self.cadena[i]
            if b["prev_hash"] != a["hash"] or b["hash"] != self._hash_bloque(b):
                return False, f"Incoherencia en bloque {i}"
        return True, "OK"

    def limpiar(self) -> None:
        self.cadena.clear()
        self._ultimo_hash = "0" * 64


# ═══════════════════════════════════════════════════════════════════
# CLASS 3: Conversor PSN - neuronas / embedding vectorial basico
# ═══════════════════════════════════════════════════════════════════
