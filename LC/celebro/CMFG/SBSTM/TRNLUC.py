"""
TRNLUC.py - Turno de dialogo de LucIA (2026)
Pipeline por turno: fases locales, IAFREE, cierre, reflejo y telemetria.
Mixin de OrquestadorSistemaLucIA; importa desde BASELUC.
"""
from __future__ import annotations

from LC.celebro.CMFG.SBSTM.BASELUC import (
    Any, Dict, Optional, time, EstiloTerminalLucIA, _RPLC_DISPONIBLE,
    _VOZ_DISPONIBLE, _hablar_voz, badge_turno, formatear_respuesta_lucia,
    reprocesar_con_metricas,
)


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

        # Fase 2: Inferencia gratuita con rotacion automatica via IAFREE
        respuesta = ""
        modelo_usado = "Reflejo-Interno"
        latencia_llm = 0.0

        if self.cliente_iafree and self.cliente_iafree.esta_autenticado():
            resp_txt, mod_id, lat = self.cliente_iafree.generar_respuesta(
                prompt=prompt,
                contexto_neuronal=estado_previo,
                stream_en_vivo=False,
            )
            if resp_txt and not resp_txt.startswith("[IAFREE]"):
                respuesta = resp_txt
                modelo_usado = mod_id
                latencia_llm = lat

        # Fase 2b: Fallback autonomo a IA local (DSIALCLGRG) si OpenRouter fallo
        if not respuesta and self.ia_local_lista:
            txt_local, mid_local, lat_local = self._responder_ia_local(prompt, estado_previo)
            if txt_local:
                respuesta = txt_local
                modelo_usado = mid_local
                latencia_llm = lat_local

        if not respuesta:
            respuesta = self._generar_reflejo_interno(prompt, estado_previo)
        elif _RPLC_DISPONIBLE and reprocesar_con_metricas is not None:
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
