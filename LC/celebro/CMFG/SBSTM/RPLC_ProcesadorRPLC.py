class ProcesadorRPLC:
    """
    Orquestador maestro de las 3 capas de transformacion RPLC:
      Capa 1: AnalizadorTextualNeuronal  - vectorizacion del texto bruto
      Capa 2: TransformadorCognitivoRPLC - modulacion sinaptica Hebbiana
      Capa 3: ReformuladorLinguisticoLucia - reconstruccion en voz de LucIA

    Uso desde mainLCSTM.py:
      procesador = get_procesador_rplc()
      texto_lucia, metricas = procesador.procesar(texto_openrouter, contexto_neuronal)
    """

    def __init__(self) -> None:
        self.analizador = AnalizadorTextualNeuronal()
        self.transformador = TransformadorCognitivoRPLC()
        self.reformulador = ReformuladorLinguisticoLucia()
        self._stats: Dict[str, Any] = {"total": 0, "tiempo_medio_ms": 0.0, "ajenos": 0, "deriva": 0.0}
        self._lock = threading.Lock()

    def procesar(self, texto: str, ctx: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """Pipeline completo: vectoriza, transforma y reformula en voz de LucIA."""
        t0 = time.perf_counter()
        if not texto or not texto.strip():
            return ("Mi canal cognitivo no recibio contenido para procesar.", {})

        vec_in = self.analizador.vectorizar(texto)
        tono = self.analizador.analizar_tono(texto)
        densidad = self.analizador.calcular_densidad_semantica(texto)
        ajena = self.analizador.detectar_voz_ajena(texto)

        vec_out = self.transformador.transformar(vec_in, ctx)
        val = self.transformador.valencia(vec_out)
        n_act = len(self.transformador.activos(vec_out))
        deriva = self.transformador.deriva_media()

        texto_lucia = self.reformulador.reformular(texto, tono, val, vec_out, n_act)

        dt = (time.perf_counter() - t0) * 1000.0
        metricas: Dict[str, Any] = {
            "tono": tono,
            "valencia_sinaptica": val,
            "densidad_semantica": densidad,
            "neuronas_activas": n_act,
            "deriva_media": deriva,
            "voz_ajena_detectada": ajena,
            "tiempo_rplc_ms": round(dt, 2),
        }
        with self._lock:
            n = self._stats["total"]
            self._stats["total"] = n + 1
            self._stats["tiempo_medio_ms"] = round((self._stats["tiempo_medio_ms"] * n + dt) / (n + 1), 2)
            self._stats["deriva"] = deriva
            if ajena:
                self._stats["ajenos"] += 1
        return (texto_lucia, metricas)

    def procesar_simple(self, texto: str) -> str:
        t, _ = self.procesar(texto)
        return t

    def metricas_globales(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)

    def diagnostico(self, texto: str) -> Dict[str, Any]:
        vec = self.analizador.vectorizar(texto)
        tono = self.analizador.analizar_tono(texto)
        tok = self.analizador.tokenizar(texto)
        vt = self.transformador.transformar(vec)
        val = self.transformador.valencia(vt)
        act = self.transformador.activos(vt)
        return {
            "tokens_extraidos": len(tok),
            "tokens_unicos": len(set(tok)),
            "tono": tono,
            "densidad_semantica": self.analizador.calcular_densidad_semantica(texto),
            "neuronas_activas": len(act),
            "indices_activos": act[:10],
            "valencia_vectorial": val,
            "deriva_actual": self.transformador.deriva_media(),
        }


