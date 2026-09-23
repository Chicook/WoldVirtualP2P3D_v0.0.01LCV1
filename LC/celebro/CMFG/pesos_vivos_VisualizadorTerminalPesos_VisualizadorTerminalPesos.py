class VisualizadorTerminalPesos:
    """Renderiza paneles de telemetria en tiempo real de pesos vivos en terminal."""

    @staticmethod
    def render_panel_turno(turno_id: int, pregunta: str, stats_familias: Dict[str, Dict[str, float]],
                           deriva_total: float, gsnr_global: float, emocion: float, tono: str) -> str:
        """Genera el frame en consola con metricas de pesos vivos 2026."""
        lineas = []
        w = 80
        sep = f"{ANSI.CYAN}{'═' * w}{ANSI.RESET}"
        sub_sep = f"{ANSI.DIM}{'─' * w}{ANSI.RESET}"

        # 1. Cabecera principal
        lineas.append(sep)
        lineas.append(
            f"{ANSI.BOLD}{ANSI.WHITE}🧠 MONITOR DE PESOS VIVOS CELEBRO 2026 {ANSI.RESET}| "
            f"{ANSI.GREEN}TURNO #{turno_id:04d}{ANSI.RESET} | "
            f"GSNR: {ANSI.YELLOW}{gsnr_global:.2f}{ANSI.RESET} | "
            f"Δ Deriva: {ANSI.MAGENTA}{deriva_total:.5f}{ANSI.RESET}"
        )
        ctx = pregunta[:60] + "..." if len(pregunta) > 60 else pregunta
        lineas.append(f"{ANSI.DIM}Contexto Entrada: {ANSI.RESET}{ctx}")
        lineas.append(
            f"Valencia Emocional: {ANSI.CYAN}{emocion:+.3f}{ANSI.RESET} | Tono: {ANSI.WHITE}{tono}{ANSI.RESET}"
        )
        lineas.append(sub_sep)

        # 2. Encabezado de la tabla de subsistemas
        lineas.append(
            f"{ANSI.BOLD}{'SUBSISTEMA':<12} {'NEURONAS':<10} {'NORMA MEDIA':<13} {'ACTIVIDAD Δ':<14} {'TELEMETRIA SINÁPTICA':<25}{ANSI.RESET}"
        )
        lineas.append(sub_sep)

        paleta = {
            "ENRN": ANSI.CYAN,
            "RF_SL": ANSI.BLUE,
            "RF_EN": ANSI.GREEN,
            "RNP": ANSI.YELLOW,
            "SLRN": ANSI.MAGENTA,
        }

        # 3. Filas de subsistemas
        for fam, d in stats_familias.items():
            col = paleta.get(fam, ANSI.WHITE)
            n_count = int(d.get("count", 10))
            norma = d.get("norma_media", 0.0)
            delta = d.get("delta_actividad", 0.0)
            barra = MetricasDinamicasPesos.barra_progreso(delta, max_val=2.5, ancho=14, color=col)
            lineas.append(
                f"{col}{ANSI.BOLD}{fam:<12}{ANSI.RESET} "
                f"{n_count:<10} "
                f"{norma:<13.4f} "
                f"{delta:<14.5f} "
                f"{barra}"
            )

        # 4. Pie de estado de estabilidad
        lineas.append(sub_sep)
        media_entropia = float(np.mean([d.get("entropia", 0.5) for d in stats_familias.values()]))
        estabilidad, diag = MetricasDinamicasPesos.calcular_indice_estabilidad(gsnr_global, media_entropia, deriva_total)
        color_diag = ANSI.GREEN if estabilidad > 0.7 else (ANSI.YELLOW if estabilidad > 0.4 else ANSI.RED)
        lineas.append(
            f"Salud Sináptica: {color_diag}{diag}{ANSI.RESET} | "
            f"Entropía Global: {ANSI.CYAN}{media_entropia:.3f}{ANSI.RESET} | "
            f"Índice: {ANSI.WHITE}{estabilidad:.2%}{ANSI.RESET}"
        )
        lineas.append(sep)
        return "\n".join(lineas)

    @staticmethod
    def render_resumen_cierre(total_turnos: int, deriva_acumulada: float,
                              archivos_guardados: List[Path]) -> str:
        """Despliega panel de consolidacion al finalizar la sesion."""
        lineas = []
        w = 80
        lineas.append(f"{ANSI.GREEN}{'═' * w}{ANSI.RESET}")
        lineas.append(f"{ANSI.BOLD}{ANSI.GREEN}💾 CONSOLIDACION DE PESOS VIVOS COMPLETADA (PSNRL / IPFS){ANSI.RESET}")
        lineas.append(f"{'─' * w}")
        lineas.append(f"Turnos procesados en sesion: {ANSI.WHITE}{total_turnos}{ANSI.RESET}")
        lineas.append(f"Deriva sinaptica total acumulada: {ANSI.MAGENTA}{deriva_acumulada:.6f}{ANSI.RESET}")
        lineas.append(f"Archivos registrados para persistencia:")
        for arch in archivos_guardados:
            lineas.append(f"  • {ANSI.CYAN}{arch.name}{ANSI.RESET} ({arch.stat().st_size if arch.exists() else 0} bytes)")
        lineas.append(f"{ANSI.GREEN}{'═' * w}{ANSI.RESET}")
        return "\n".join(lineas)


# ===========================================================================
# GESTOR DE PESOS VIVOS (ORQUESTADOR DE TURNO Y PERSISTENCIA)
# ===========================================================================
