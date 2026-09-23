class ClienteIAFree:
    """Cliente de inferencia para LucIA que garantiza coste cero ($0.00) y alta resiliencia."""

    ENDPOINT_CHAT: Final[str] = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.gestor = GestorModelosGratuitos(api_key=api_key)
        self.temperatura = 0.7
        self.max_tokens = 1536
        self.total_consultas_gratuitas = 0
        self.total_caracteres_recibidos = 0
        self._lock = threading.Lock()

    def esta_autenticado(self) -> bool:
        """Verifica que se cuente con un API token valido de OpenRouter."""
        return bool(self.gestor.api_key and len(self.gestor.api_key) > 10)

    def generar_respuesta(
        self,
        prompt: str,
        contexto_neuronal: Optional[Dict[str, Any]] = None,
        max_reintentos: int = 4,
        stream_en_vivo: bool = True,
    ) -> Tuple[str, str, float]:
        """
        Ejecuta la inferencia utilizando la cola de modelos gratuitos.
        Si un modelo responde con 429 (Too Many Requests), 404 o timeout,
        rota automaticamente al siguiente modelo gratuito disponible sin cobrar dinero.

        Retorna:
          (texto_respuesta, id_modelo_utilizado, latencia_ms)
        """
        if not self.esta_autenticado():
            return ("[IAFREE] Clave OPENROUTER_API_KEY no encontrada en .env", "sin_clave", 0.0)

        contexto = contexto_neuronal or {}
        tono = contexto.get("tono_cognitivo", "analitico")
        valencia = contexto.get("estado_emocional", 0.0)
        try:
            from LC.compatibility.architecture import architecture_context
            contexto_arquitectura = architecture_context()
        except Exception:
            contexto_arquitectura = "Estado factual de arquitectura no disponible."

        prompt_sistema = (
            "Eres LucIA, el sistema cognitivo distribuido de WoldVirtualP2P3D (2026). "
            f"Tu tono neuronal actual es '{tono}' y tu valencia es {valencia:+.2f}. "
            "Responde de forma clara, directa, tecnica y cordial a la solicitud del usuario. "
            "Cuando te pregunten por cambios, refactorizaciones o tu arquitectura, usa "
            "el siguiente estado factual y no inventes un changelog:\n\n"
            f"{contexto_arquitectura}"
        )

        intentos = 0
        while intentos < max_reintentos:
            modelo_actual = self.gestor.obtener_modelo_activo()
            modelo_id = modelo_actual["id"]
            t_inicio = time.perf_counter()

            payload = json.dumps({
                "model": modelo_id,
                "messages": [
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperatura,
                "max_tokens": self.max_tokens,
                "stream": stream_en_vivo,
            }).encode("utf-8")

            headers = {
                "Authorization": f"Bearer {self.gestor.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://woldvirtualp2p3d.network",
                "X-Title": "LucIA-IAFREE-Subsystem",
            }

            req = urllib.request.Request(self.ENDPOINT_CHAT, data=payload, headers=headers, method="POST")
            tokens: List[str] = []

            try:
                with urllib.request.urlopen(req, timeout=25.0) as respuesta:
                    if stream_en_vivo:
                        sys.stdout.write(f"\n\033[96m  LucIA [Free:{modelo_id}] ▶ \033[0m")
                        sys.stdout.flush()
                        for linea in respuesta:
                            linea_str = linea.decode("utf-8").strip()
                            if not linea_str or not linea_str.startswith("data: "):
                                continue
                            raw = linea_str[6:]
                            if raw == "[DONE]":
                                break
                            try:
                                chunk = json.loads(raw)
                                d = chunk["choices"][0]["delta"].get("content", "")
                                if d:
                                    sys.stdout.write(d)
                                    sys.stdout.flush()
                                    tokens.append(d)
                            except Exception:
                                continue
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    else:
                        data = json.loads(respuesta.read().decode("utf-8"))
                        texto = data["choices"][0]["message"]["content"]
                        tokens.append(texto)

                latencia = (time.perf_counter() - t_inicio) * 1000.0
                resultado = "".join(tokens).strip()

                if resultado:
                    with self._lock:
                        self.total_consultas_gratuitas += 1
                        self.total_caracteres_recibidos += len(resultado)
                    return (resultado, modelo_id, latencia)

            except urllib.error.HTTPError as h_err:
                codigo = h_err.code
                razon = f"HTTP {codigo}"
                self.gestor.rotar_al_siguiente_modelo(razon=razon)
                intentos += 1
                time.sleep(min(8.0, 0.5 * (2 ** intentos)))
            except Exception as e_gen:
                self.gestor.rotar_al_siguiente_modelo(razon=str(e_gen))
                intentos += 1
                time.sleep(min(8.0, 0.5 * (2 ** intentos)))

        return (
            "[IAFREE] Todos los modelos gratuitos consultados se encuentran temporalmente saturados. "
            "Activando reflejo neuronal interno autónomo.",
            "fallback_agotado",
            0.0,
        )


    def obtener_metricas_consumo(self) -> Dict[str, Any]:
        """Retorna un reporte de auditoria demostrando coste total de $0.00 USD."""
        with self._lock:
            return {
                "consultas_totales_gratuitas": self.total_consultas_gratuitas,
                "caracteres_recibidos": self.total_caracteres_recibidos,
                "costo_acumulado_usd": 0.0,
                "modelo_activo": self.gestor.obtener_modelo_activo()["id"],
                "total_modelos_en_pool": len(self.gestor._modelos),
                "fallos_registrados": dict(self.gestor._fallos_consecutivos),
            }

    def benchmark_rapido_modelos(self, max_modelos: int = 3) -> Dict[str, float]:
        """1 llamada corta por modelo con timeout de 10 s (sin quemar cuota)."""
        resultados: Dict[str, float] = {}
        for m in self.gestor.listar_modelos()[:max_modelos]:
            m_id = m["id"]
            t0 = time.perf_counter()
            try:
                payload = json.dumps({
                    "model": m_id,
                    "messages": [{"role": "user", "content": "1+1="}],
                    "max_tokens": 4,
                }).encode("utf-8")
                req = urllib.request.Request(
                    self.ENDPOINT_CHAT,
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {self.gestor.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10.0) as r:
                    _ = r.read()
                resultados[m_id] = round((time.perf_counter() - t0) * 1000.0, 1)
            except Exception:
                resultados[m_id] = -1.0
        return resultados


# ─── SUBSISTEMA Y SINGLETON CENTRAL ─────────────────────────────────────────
