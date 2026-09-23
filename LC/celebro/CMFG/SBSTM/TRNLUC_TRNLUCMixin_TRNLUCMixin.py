class TRNLUCMixin:
    """Mezcla de turno: inferencia, cierre y telemetria."""

    def procesar_turno_dialogo(self, prompt: str) -> None:
        """
        Ejecuta el pipeline cognitivo completo por turno a coste cero con renderizado STYLOS:
        1. Pre-procesamiento por las 50 neuronas para tono y estado emocional.
        2. Inferencia gratuita via IAFREE.
        3. Formateo y presentacion de respuesta estilizada (tarjetas redondeadas y markdown).
        4. Asimilacion de gradientes Hebbianos en tensores de PSNRL.
        5. Emision de transaccion sinaptica a Blockchain BKSVCB y minado automatico.
        """
        self.turno_actual += 1
        t_inicio = time.perf_counter()

        if self.conversor_psn is None or self.servidor_bks is None:
            print("  [AVISO] Subsistemas no inicializados; turno omitido.")
            return
        # Fase 1: Pre-activacion neuronal
        estado_previo = self.conversor_psn.procesar_consulta_a_pesos(prompt)

        # Las preguntas sobre el estado del sistema se responden desde datos
        # locales verificables, no desde una conjetura del modelo remoto.
        low_diag = prompt.lower()
        if any(k in low_diag for k in (
                "que cambios", "qué cambios", "que hiciste", "qué hiciste",
                "changelog", "actualizaciones", "refactorizacion", "refactorización",
                "diagnostico de lucia", "diagnóstico de lucia", "estado de la arquitectura")):
            try:
                from LC.compatibility.architecture import architecture_answer
                respuesta_diag = architecture_answer()
                return self._cerrar_turno(respuesta_diag, "LucIA-Diagnostico-Local", 0.0,
                                          prompt, estado_previo, t_inicio)
            except Exception:
                pass

        # Fase 1b: MDSTM — orden de descarga/consulta local se EJECUTA aqui,
        # sin preguntar al modelo remoto (asi LucIA nunca dice "no puedo").
        if self.gestor_mdstm is not None:
            try:
                if self.gestor_mdstm.es_orden_descarga(prompt):
                    respuesta_md = self.gestor_mdstm.ejecutar_orden(prompt)
                    if respuesta_md:
                        respuesta, modelo_usado = respuesta_md, "LucIA-MDSTM-local"
                        latencia_llm = 0.0
                        return self._cerrar_turno(respuesta, modelo_usado, latencia_llm,
                                                  prompt, estado_previo, t_inicio)
                low = prompt.lower()
                if any(k in low for k in ("que modelos tienes", "modelos descargados",
                                          "cuanta ram", "cuánta ram", "cuanta vram",
                                          "que hardware", "qué hardware", "qué puedes descargar",
                                          "que puedes descargar")):
                    respuesta = self.gestor_mdstm.informe_para_lucia()
                    return self._cerrar_turno(respuesta, "LucIA-MDSTM-local", 0.0,
                                              prompt, estado_previo, t_inicio)
            except Exception:
                pass

        # Fase 1c: HRCTRC — orden de construccion se EJECUTA aqui con permisos
        # via codigo (crear carpetas, overlay, unificar), sin modelo remoto.
        if self.gestor_hrctrc is not None:
            try:
                if self.gestor_hrctrc.es_orden_constructor(prompt):
                    respuesta_h = self.gestor_hrctrc.ejecutar_orden(prompt)
                    if respuesta_h:
                        return self._cerrar_turno(respuesta_h, "LucIA-HRCTRC-local", 0.0,
                                                  prompt, estado_previo, t_inicio)
                low_h = prompt.lower()
                if any(k in low_h for k in ("estado del constructor", "estado constructor",
                                            "que hay en constructor", "qué hay en constructor")):
                    respuesta = self.gestor_hrctrc.informe_para_lucia()
                    return self._cerrar_turno(respuesta, "LucIA-HRCTRC-local", 0.0,
                                              prompt, estado_previo, t_inicio)
            except Exception:
                pass

        # Fase 1d: HRCTRC_RFCT — refactor/version se EJECUTA con modelo local,
        # sin pasar por el remoto (nunca dice "no puedo leer esa ruta").
        if self.refactorizador is not None:
            try:
                if self.refactorizador.es_orden_refactor(prompt):
                    respuesta_r = self.refactorizador.ejecutar_orden(prompt)
                    if respuesta_r:
                        return self._cerrar_turno(respuesta_r, "LucIA-RFCT-local", 0.0,
                                                  prompt, estado_previo, t_inicio)
            except Exception:
                pass

        # Fase 1e: INTEGRACIONRF — integrar/bitacora/IPFS/rama se EJECUTA aqui.
        if self.integrador is not None:
            try:
                if self.integrador.es_orden_integracion(prompt):
                    respuesta_i = self.integrador.ejecutar_orden(prompt)
                    if respuesta_i:
                        return self._cerrar_turno(respuesta_i, "LucIA-INTEGRACIONRF-local", 0.0,
                                                  prompt, estado_previo, t_inicio)
            except Exception:
                pass

        # Fase 1f: HRCNTR — orden de monitor/refactor neural se EJECUTA aqui.
        if self.gestor_hrctrc is not None:
            try:
                if self.gestor_hrctrc.es_orden_constructor(prompt):
                    from LC.celebro.CMFG.SBSTM.HRCNTR import ejecutar_hrctnr
                    respuesta_h = ejecutar_hrctnr(mostrar_barra=False).get("mensaje", "")
                    if respuesta_h:
                        return self._cerrar_turno("Refactor Neural: " + respuesta_h,
                                                  "LucIA-HRCNTR-local", 0.0,
                                                  prompt, estado_previo, t_inicio)
            except Exception:
                pass

        # Fase 2: Rotacion Ollama -> LM Studio -> OpenRouter gratuito.
        # Los backends locales se prueban antes de consumir el pool gratuito.
        respuesta = ""
        modelo_usado = "Reflejo-Interno"
        latencia_llm = 0.0

        if self.rotador_ia is not None:
            resultado_ia = self.rotador_ia.consultar(
                prompt=prompt,
                contexto=estado_previo,
                cliente_openrouter=self.cliente_iafree,
            )
            respuesta = str(resultado_ia.get("texto", ""))
            modelo_usado = str(resultado_ia.get("modelo", modelo_usado))
            latencia_llm = float(resultado_ia.get("latencia_ms", 0.0))
        elif self.ia_local_lista:
            txt_local, mid_local, lat_local = self._responder_ia_local(prompt, estado_previo)
            respuesta, modelo_usado, latencia_llm = txt_local, mid_local, lat_local

        if not respuesta:
            respuesta = self._generar_reflejo_interno(prompt, estado_previo)
        elif self.conversor_psn is not None:
            # Fase 3: respuesta externa -> pesos -> propagación por las 50 neuronas.
            try:
                if self.aprendizaje_seguro is not None:
                    sintesis = self.aprendizaje_seguro.asimilar(
                        conversor=self.conversor_psn,
                        pregunta=prompt,
                        respuesta_fuente=respuesta,
                        modelo_origen=modelo_usado,
                    )
                    if not sintesis.get("aceptado", False):
                        raise ValueError(str(sintesis.get("motivo", "gate neuronal rechazado")))
                    if self.turno_actual % 5 == 0:
                        sintesis["replay"] = self.aprendizaje_seguro.replayar(
                            self.conversor_psn, limite=2)
                else:
                    sintesis = self.conversor_psn.asimilar_respuestas_y_calcular_sintesis(
                        prompt=prompt,
                        respuesta_modelo=respuesta,
                        modelo_nombre=modelo_usado,
                    )
                contexto_sintesis = {**estado_previo, **sintesis}
                # Fase 4: segunda pasada lingüística. LucIA vuelve a generar
                # la respuesta usando el estado neuronal ya distribuido.
                respuesta_sintesis = ""
                if self.sintetizador_lucia is not None:
                    resultado_sintesis = self.sintetizador_lucia.sintetizar(
                        pregunta=prompt,
                        respuesta_fuente=respuesta,
                        estado_neuronal=contexto_sintesis,
                        modelo_origen=modelo_usado,
                        cliente_openrouter=self.cliente_iafree,
                    )
                    respuesta_sintesis = str(resultado_sintesis.get("texto", ""))
                    if respuesta_sintesis:
                        respuesta = respuesta_sintesis
                        modelo_usado = str(resultado_sintesis.get("modelo", "LucIA-sintesis"))
                # Fase 5: respaldo estilístico si ningún backend puede
                # realizar la segunda pasada.
                if not respuesta_sintesis and _RPLC_DISPONIBLE and reprocesar_con_metricas is not None:
                    respuesta, _ = reprocesar_con_metricas(respuesta, contexto_sintesis)
                    modelo_usado = f"LucIA-reglas[{modelo_usado}]"
            except Exception:
                if _RPLC_DISPONIBLE and reprocesar_con_metricas is not None:
                    try:
                        respuesta, _ = reprocesar_con_metricas(respuesta, estado_previo)
                    except Exception:
                        pass

        self._cerrar_turno(respuesta, modelo_usado, latencia_llm, prompt, estado_previo, t_inicio)

    def _cerrar_turno(self, respuesta: str, modelo_usado: str, latencia_llm: float,
                      prompt: str, estado_previo: Dict[str, Any], t_inicio: float) -> None:
        """Fases 3-5 del turno: render STYLOS + voz, registro BKSVCB, minado y telemetria."""
        if EstiloTerminalLucIA:
            formatear_respuesta_lucia(respuesta, modelo=modelo_usado)
        else:
            print(f"\nLucIA [{modelo_usado}] ▶ {respuesta}\n")

        if _VOZ_DISPONIBLE and _hablar_voz is not None:
            _hablar_voz(respuesta, esperar=False)

        duracion_ms = (time.perf_counter() - t_inicio) * 1000.0

        # Fase 4: Asimilacion en 50 neuronas y registro en Blockchain
        cid_ipfs = "--"
        try:
            res_chain = self.servidor_bks.registrar_aprendizaje_neural(
                prompt=prompt, respuesta=respuesta, modelo=f"LucIA-{modelo_usado}"
            )
            cid_ipfs = res_chain.get("ipfs_cid", "--")
        except Exception as ex_bks:
            print(f"\033[38;5;203m  [Blockchain Error] {ex_bks}\033[0m")

        # Fase 5: Minado automatico por turnos
        if self.turno_actual % 3 == 0:
            bloque_nuevo = self.servidor_bks.minar_transacciones_pendientes()
            if bloque_nuevo:
                if EstiloTerminalLucIA:
                    EstiloTerminalLucIA.renderizar_notificacion_bloque(bloque_nuevo.indice, bloque_nuevo.hash_bloque)
                else:
                    print(f"  [Bloque #{bloque_nuevo.indice} Minado] {bloque_nuevo.hash_bloque[:24]}...")

        # Telemetria estilizada en consola
        self._imprimir_telemetria_turno(duracion_ms, cid_ipfs, modelo_usado)

        # Bitacora viva INTEGRACIONRF: cada turno queda para el .md de cierre
        try:
            if self.integrador is not None:
                self.integrador.registrar(
                    "turno", f"#{self.turno_actual} mod={modelo_usado} "
                             f"{duracion_ms:.0f}ms cid={str(cid_ipfs)[:12]}")
        except Exception:
            pass

    def _generar_reflejo_interno(self, prompt: str, info: Dict[str, Any]) -> str:
        tono = info.get("tono_cognitivo", "reflexivo")
        val = info.get("estado_emocional", 0.0)
        norma = info.get("norma_delta_aplicada", 0.0)
        return (
            f"Estimulo cognitivo asimilado en red distribuida. Tono: **{tono}** | "
            f"Valencia afectiva: `{val:+.3f}` | Delta sinaptico: `{norma:.5f}`.\n"
            "Pesos neuronales sincronizados y activos en persistencia `PSNRL`."
        )

    def _imprimir_telemetria_turno(self, duracion_ms: float, cid: str, modelo_id: str) -> None:
        deriva = self.conversor_psn.deriva_acumulada
        neuronas = len(self.conversor_psn.neuronas)
        txs_espera = len(self.servidor_bks.transacciones_pendientes)

        if EstiloTerminalLucIA:
            badge_turno(
                turno=self.turno_actual,
                dt=duracion_ms,
                mod=modelo_id,
                der=deriva,
                neu=neuronas,
                cid=cid,
                txs=txs_espera,
            )
        else:
            print(f"  Turno #{self.turno_actual} | {duracion_ms:.0f}ms | Mod: {modelo_id} | Deriva: {deriva:.4f} | CID: {cid[:16]}")
