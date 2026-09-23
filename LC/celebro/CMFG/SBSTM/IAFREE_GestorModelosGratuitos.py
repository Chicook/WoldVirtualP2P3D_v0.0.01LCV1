class GestorModelosGratuitos:
    """Administra la lista de modelos gratuitos, pruebas de liveness y conmutacion por error."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        # None significa "leer del entorno"; "" significa "sin clave explícita"
        if api_key is None:
            self.api_key = self._leer_api_key()
        else:
            self.api_key = api_key
        self._modelos: List[Dict[str, Any]] = list(CATALOGO_MODELOS_GRATUITOS)
        self._indice_activo = 0
        self._fallos_consecutivos: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._ultimo_refresco = 0.0

    def _leer_api_key(self) -> str:
        """Extrae la clave API desde .env o el entorno de ejecucion."""
        if ENV_FILE.exists():
            try:
                txt = ENV_FILE.read_text(encoding="utf-8-sig", errors="replace")
                for line in txt.splitlines():
                    if line.strip().startswith("OPENROUTER_API_KEY="):
                        return line.strip().split("=", 1)[1].strip().strip("'\"")
            except Exception:
                pass
        return os.getenv("OPENROUTER_API_KEY", "").strip()

    def obtener_modelo_activo(self) -> Dict[str, Any]:
        """Retorna el modelo que actualmente encabeza la cola de inferencia."""
        with self._lock:
            return self._modelos[self._indice_activo]

    def rotar_al_siguiente_modelo(self, razon: str = "error") -> Dict[str, Any]:
        """Avanza al siguiente modelo gratuito de la lista ante errores o limites de tasa."""
        with self._lock:
            actual = self._modelos[self._indice_activo]["id"]
            self._fallos_consecutivos[actual] = self._fallos_consecutivos.get(actual, 0) + 1
            self._indice_activo = (self._indice_activo + 1) % len(self._modelos)
            nuevo = self._modelos[self._indice_activo]
            logger.warning(
                "Rotacion de modelo gratuito: %s -> %s (Razon: %s)",
                actual, nuevo["id"], razon
            )
            return nuevo

    def seleccionar_por_id(self, modelo_id: str) -> bool:
        """Fija manualmente un modelo especifico por su identificador."""
        with self._lock:
            for idx, m in enumerate(self._modelos):
                if m["id"] == modelo_id or m["id"].split(":")[0] == modelo_id:
                    self._indice_activo = idx
                    return True
            return False

    def listar_modelos(self, categoria: Optional[str] = None) -> List[Dict[str, Any]]:
        """Devuelve los modelos gratuitos registrados, opcionalmente filtrados por categoria."""
        with self._lock:
            if not categoria:
                return list(self._modelos)
            return [m for m in self._modelos if m.get("categoria") == categoria]

    def actualizar_catalogo_online(self) -> int:
        """Consulta la API de OpenRouter y agrega nuevos modelos gratuitos (máx 60, con caché)."""
        ahora = time.time()
        if (ahora - self._ultimo_refresco) < 300.0:
            return len(self._modelos)
        cached = self._cargar_cache()
        if cached:
            self._fusionar_items(cached)
            self._ultimo_refresco = ahora
            return len(self._modelos)

        url = "https://openrouter.ai/api/v1/models"
        req = urllib.request.Request(url, headers={"User-Agent": "LucIA-IAFREE-2026"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                nuevos_registrados = 0
                existentes = {m["id"] for m in self._modelos}

                for item in data.get("data", []):
                    m_id = item.get("id", "")
                    pricing = item.get("pricing", {})
                    p_cost = float(pricing.get("prompt", 0) or 0)
                    c_cost = float(pricing.get("completion", 0) or 0)

                    if (":free" in m_id or (p_cost == 0.0 and c_cost == 0.0)) and m_id not in existentes:
                        nuevo_m = {
                            "id": m_id,
                            "nombre": item.get("name", m_id),
                            "contexto": item.get("context_length", 262144),
                            "categoria": "descubierto",
                            "descripcion": item.get("description", "Modelo gratuito descubierto en linea.")[:120],
                        }
                        with self._lock:
                            if len(self._modelos) < 60:
                                self._modelos.append(nuevo_m)
                                existentes.add(m_id)
                        nuevos_registrados += 1

                self._ultimo_refresco = ahora
                self._guardar_cache(data.get("data", []))
                logger.info("Catalogo gratuito actualizado: %d modelos anadidos.", nuevos_registrados)
                return len(self._modelos)
        except Exception as exc:
            logger.debug("No se pudo actualizar el catalogo online: %s", exc)
            return len(self._modelos)

    def _cargar_cache(self) -> List[Dict[str, Any]]:
        try:
            if CACHE_FILE.exists():
                raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and "data" in raw:
                    return raw["data"]
                if isinstance(raw, list):
                    return raw
        except Exception as exc:
            logger.debug("Cache IAFREE ilegible: %s", exc)
        return []

    def _guardar_cache(self, items: List[Dict[str, Any]]) -> None:
        try:
            CACHE_FILE.write_text(json.dumps({"ts": time.time(), "data": items[:200]}, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.debug("Cache IAFREE no guardada: %s", exc)

    def _fusionar_items(self, items: List[Dict[str, Any]]) -> int:
        nuevos = 0
        existentes = {m["id"] for m in self._modelos}
        for item in items:
            m_id = item.get("id", "") if isinstance(item, dict) else ""
            if ":free" in m_id and m_id not in existentes and len(self._modelos) < 60:
                self._modelos.append({"id": m_id, "nombre": item.get("name", m_id),
                                      "contexto": item.get("context_length", 262144),
                                      "categoria": "descubierto",
                                      "descripcion": "Desde caché local."})
                existentes.add(m_id)
                nuevos += 1
        return nuevos


# ─── CLIENTE DE INFERENCIA GRATUITA RESILIENTE (IAFREE CLIENT) ──────────────
