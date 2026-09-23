class GestorPesosVivos:
    """Orquesta la inyeccion dinamica de turnos, telemetria viva y checkpoints."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.turno_contador: int = 0
        self.deriva_sesion: float = 0.0
        self.historial_gsnr: List[float] = []
        self.historial_turnos: List[Dict[str, Any]] = []
        self.visualizador = VisualizadorTerminalPesos()
        logger.info("GestorPesosVivos 2026 instanciado | Sesion: %s", self.session_id)

    def inyectar_turno_en_vivo(self, conversor: Any, pregunta: str, respuesta: str,
                              info_pesos: Dict[str, Any], mostrar_en_terminal: bool = True) -> Dict[str, Any]:
        """
        Ejecuta la inyeccion sinaptica viva de turno en las 50 neuronas de Celebro,
        calcula metricas 2026 y despliega el monitor en la terminal.
        """
        self.turno_contador += 1
        t0 = time.perf_counter()

        # 1. Extraccion de metricas base
        emocion = float(info_pesos.get("estado_emocional", 0.0))
        tono = str(info_pesos.get("tono_cognitivo", "analítico y reflexivo"))
        delta_aplicada = float(info_pesos.get("norma_delta_aplicada",
                                             info_pesos.get("norma_delta_pesos", 0.01)))
        self.deriva_sesion += delta_aplicada

        # 2. Asimilacion en memoria de salida a traves del conversor
        if hasattr(conversor, "actualizar_memoria_salida"):
            conversor.actualizar_memoria_salida(pregunta, respuesta, modelo_origen="turno_vivo_2026")

        # 3. Calculo de metricas en tiempo real por cada uno de los 5 subsistemas
        familias = ["ENRN", "RF_SL", "RF_EN", "RNP", "SLRN"]
        stats_familias: Dict[str, Dict[str, float]] = {}
        gsnr_acumulado = []

        neuronas_dict = getattr(conversor, "neuronas", {})
        for fam in familias:
            pref = "EN" if fam == "ENRN" else ("RFSL" if fam == "RF_SL" else ("RFEN" if fam == "RF_EN" else fam))
            mats = []
            for k, n in neuronas_dict.items():
                if k.startswith(pref):
                    for attr in ["pesos", "q_online", "pesos_actor", "q_table", "pesos_actor_global", "pesos_q_principal"]:
                        w = getattr(n, attr, None)
                        if isinstance(w, np.ndarray) and w.size > 0:
                            mats.append(w)
                            break
            if mats:
                norma_media = float(np.mean([np.linalg.norm(m) for m in mats]))
                entropia = float(np.mean([MetricasDinamicasPesos.entropia_shannon(m) for m in mats]))
                actividad = float(delta_aplicada * (0.8 + 0.4 * entropia))
            else:
                norma_media, entropia, actividad = 1.0, 0.5, delta_aplicada

            stats_familias[fam] = {
                "count": len(mats) if mats else 10,
                "norma_media": norma_media,
                "delta_actividad": actividad,
                "entropia": entropia,
            }
            gsnr_acumulado.append(1.0 / (entropia + 1e-4))

        gsnr_global = float(np.mean(gsnr_acumulado)) if gsnr_acumulado else 1.0
        self.historial_gsnr.append(gsnr_global)

        # 4. Renderizado en terminal
        panel_texto = self.visualizador.render_panel_turno(
            turno_id=self.turno_contador,
            pregunta=pregunta,
            stats_familias=stats_familias,
            deriva_total=self.deriva_sesion,
            gsnr_global=gsnr_global,
            emocion=emocion,
            tono=tono,
        )

        if mostrar_en_terminal:
            print(panel_texto)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        resultado = {
            "turno": self.turno_contador,
            "delta_aplicada": delta_aplicada,
            "deriva_sesion": self.deriva_sesion,
            "gsnr_global": gsnr_global,
            "stats_familias": stats_familias,
            "tiempo_ms": round(dt_ms, 2),
            "panel_texto": panel_texto,
        }
        self.historial_turnos.append(resultado)
        return resultado

    def checkpoint_y_registrar(self, conversor: Any, sesion: Optional[Any] = None,
                              etiqueta: str = "pesos_vivos") -> List[Path]:
        """
        Genera checkpoint binario y metadata JSON en PSNRL, actualizando pesos_activos.
        Registra los archivos para sincronizacion IPFS si la sesion esta provista.
        """
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = f"{etiqueta}_{ts}"
        archivos_generados: List[Path] = []

        if hasattr(conversor, "persistir_pesos_en_psnrl"):
            npz_path, json_path = conversor.persistir_pesos_en_psnrl(etiqueta=tag)
            archivos_generados.extend([npz_path, json_path])
        else:
            npz_path = PSNRL_DIR / f"{tag}_pesos.npz"
            dict_pesos = {}
            for k, n in getattr(conversor, "neuronas", {}).items():
                for attr in ["pesos", "q_online", "pesos_actor", "q_table"]:
                    w = getattr(n, attr, None)
                    if isinstance(w, np.ndarray):
                        dict_pesos[f"{k}_{attr}"] = w.astype(np.float32)
            np.savez_compressed(npz_path, **dict_pesos)
            archivos_generados.append(npz_path)

        # Registro para IPFS
        if sesion is not None and hasattr(sesion, "registrar_archivo_pesos"):
            for arch in archivos_generados:
                try:
                    sesion.registrar_archivo_pesos(arch)
                    logger.info("pesos_vivos: %s registrado para IPFS", arch.name)
                except Exception as e:
                    logger.warning("Error registrando archivo en sesion: %s", e)

        # Render de confirmacion en consola
        resumen = self.visualizador.render_resumen_cierre(
            total_turnos=self.turno_contador,
            deriva_acumulada=self.deriva_sesion,
            archivos_guardados=archivos_generados
        )
        print(resumen)
        return archivos_generados

    def obtener_telemetria_resumen(self) -> Dict[str, Any]:
        """Retorna el resumen consolidado de telemetria de pesos vivos."""
        return {
            "session_id": self.session_id,
            "total_turnos": self.turno_contador,
            "deriva_total": self.deriva_sesion,
            "gsnr_medio": float(np.mean(self.historial_gsnr)) if self.historial_gsnr else 1.0,
            "historial_turnos_len": len(self.historial_turnos),
        }


# ===========================================================================
# INTERFAZ COMPATIBLE HACIA ATRAS (CONTRATO HISTORICO DE CELEBRO)
# ===========================================================================
