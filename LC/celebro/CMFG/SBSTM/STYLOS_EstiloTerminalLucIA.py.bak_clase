import sys
from typing import Any, Dict
from LC.celebro.CMFG.SBSTM.STYLOS import Glifos, obtener_ancho_consola, longitud_visual
from LC.celebro.CMFG.SBSTM.STYLOS_ColoresLucIA import ColoresLucIA
from LC.celebro.CMFG.SBSTM.STYLOS_RenderizadorMarkdown import RenderizadorMarkdown


class EstiloTerminalLucIA:
    """Motor de diseno visual para respuestas conversacionales y paneles de control."""

    @staticmethod
    def renderizar_encabezado_sesion(sesion_id: str, modelo_activo: str, total_neuronas: int = 50) -> None:
        """Despliega el banner principal con estetica de consola futurista estilo OpenCode."""
        ancho = obtener_ancho_consola()
        borde_arr = Glifos.ESQ_ARR_IZQ + (Glifos.LINEA_H * (ancho - 2)) + Glifos.ESQ_ARR_DER
        borde_aba = Glifos.ESQ_ABA_IZQ + (Glifos.LINEA_H * (ancho - 2)) + Glifos.ESQ_ABA_DER
        linea_div = Glifos.CRUZ_IZQ + (Glifos.LINEA_H * (ancho - 2)) + Glifos.CRUZ_DER

        print()
        print(f"{ColoresLucIA.CIAN_ELECTRICO}{borde_arr}{ColoresLucIA.RESET}")
        titulo = f" WOLDVIRTUALP2P3D  ✦  LUCIA COGNITIVE CONSOLE 2026 "
        pad_tit = ancho - 2 - len(titulo)
        print(f"{ColoresLucIA.CIAN_ELECTRICO}{Glifos.LINEA_V}{ColoresLucIA.RESET}{ColoresLucIA.BOLD}{ColoresLucIA.BLANCO_LUMINOSO}{titulo}{' ' * max(0, pad_tit)}{ColoresLucIA.CIAN_ELECTRICO}{Glifos.LINEA_V}{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.GRIS_METALLIC}{linea_div}{ColoresLucIA.RESET}")

        linea_info = f"  ID: {ColoresLucIA.VIOLETA_NEON}{sesion_id}{ColoresLucIA.RESET} | Mod: {ColoresLucIA.AMBAR_SOLAR}{modelo_activo}{ColoresLucIA.RESET} | Sinapsis: {ColoresLucIA.VERDE_SINAPTICO}{total_neuronas} Activas{ColoresLucIA.RESET} | Coste: {ColoresLucIA.ESMERALDA_VIVO}$0.00{ColoresLucIA.RESET}"
        pad_info = ancho - 2 - longitud_visual(linea_info)
        print(f"{ColoresLucIA.CIAN_ELECTRICO}{Glifos.LINEA_V}{ColoresLucIA.RESET}{linea_info}{' ' * max(0, pad_info)}{ColoresLucIA.CIAN_ELECTRICO}{Glifos.LINEA_V}{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.CIAN_ELECTRICO}{borde_aba}{ColoresLucIA.RESET}\n")

    @staticmethod
    def renderizar_prompt_usuario(prompt: str) -> None:
        """Muestra el estimulo ingresado por el usuario en una tarjeta compacta."""
        ancho = obtener_ancho_consola()
        etiqueta = f" Tu > "
        print(f"\n{ColoresLucIA.BOLD}{ColoresLucIA.AMBAR_SOLAR}╭──{etiqueta}{Glifos.LINEA_H * (ancho - len(etiqueta) - 4)}╮{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.AMBAR_SOLAR}│{ColoresLucIA.RESET}  {ColoresLucIA.BLANCO_LUMINOSO}{prompt}{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.AMBAR_SOLAR}╰{'─' * (ancho - 2)}╯{ColoresLucIA.RESET}\n")

    @staticmethod
    def iniciar_tarjeta_respuesta(modelo_id: str = "openrouter/free") -> None:
        """Imprime el encabezado redondeado de la tarjeta antes del streaming de LucIA."""
        ancho = obtener_ancho_consola()
        insignia = f" LucIA  [Free: {modelo_id}] "
        resto = ancho - len(insignia) - 3
        print(f"{ColoresLucIA.VIOLETA_NEON}╭─{ColoresLucIA.BOLD}{ColoresLucIA.CIAN_ELECTRICO}{insignia}{ColoresLucIA.VIOLETA_NEON}{Glifos.LINEA_H * max(2, resto)}╮{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.VIOLETA_NEON}│{ColoresLucIA.RESET}")

    @staticmethod
    def imprimir_token_stream(token: str) -> None:
        """Escribe los tokens en tiempo real manteniendo el margen visual de la tarjeta."""
        # Se imprime directamente conservando fluidez visual
        sys.stdout.write(f"{ColoresLucIA.BLANCO_LUMINOSO}{token}{ColoresLucIA.RESET}")
        sys.stdout.flush()

    @staticmethod
    def cerrar_tarjeta_respuesta() -> None:
        """Cierra la tarjeta de respuesta de LucIA tras finalizar la emision."""
        ancho = obtener_ancho_consola()
        print(f"\n{ColoresLucIA.VIOLETA_NEON}│{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.VIOLETA_NEON}╰{Glifos.LINEA_H * (ancho - 2)}╯{ColoresLucIA.RESET}")

    @staticmethod
    def renderizar_respuesta_completa(texto: str, modelo_id: str = "openrouter/free") -> None:
        """Formatea e imprime una respuesta completa con deteccion de Markdown y tablas."""
        ancho = obtener_ancho_consola()
        insignia = f" LucIA  [Free: {modelo_id}] "
        resto = ancho - len(insignia) - 3

        print(f"{ColoresLucIA.VIOLETA_NEON}╭─{ColoresLucIA.BOLD}{ColoresLucIA.CIAN_ELECTRICO}{insignia}{ColoresLucIA.VIOLETA_NEON}{Glifos.LINEA_H * max(2, resto)}╮{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.VIOLETA_NEON}│{ColoresLucIA.RESET}")

        lineas = texto.splitlines()
        en_tabla = False
        buffer_tabla = []

        for l in lineas:
            # Deteccion de tablas Markdown
            if l.strip().startswith("|") and l.strip().endswith("|"):
                en_tabla = True
                buffer_tabla.append(l)
                continue
            elif en_tabla:
                en_tabla = False
                print(RenderizadorMarkdown.renderizar_tabla(buffer_tabla, ancho))
                buffer_tabla = []

            # Linea normal con formateo de sintaxis
            linea_fmt = RenderizadorMarkdown.colorear_sintaxis(l)
            print(f"  {linea_fmt}")

        if en_tabla and buffer_tabla:
            print(RenderizadorMarkdown.renderizar_tabla(buffer_tabla, ancho))

        print(f"{ColoresLucIA.VIOLETA_NEON}│{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.VIOLETA_NEON}╰{Glifos.LINEA_H * (ancho - 2)}╯{ColoresLucIA.RESET}")

    @staticmethod
    def renderizar_barra_telemetria(
        turno: int,
        duracion_ms: float,
        modelo_id: str,
        deriva: float,
        neuronas: int,
        cid_ipfs: str,
        txs_espera: int,
    ) -> None:
        """Imprime un badge de telemetria compacto con colores cybernéticos."""
        cid_corte = cid_ipfs[:16] + "..." if len(cid_ipfs) > 16 and cid_ipfs != "--" else cid_ipfs

        badge_turno = f"{ColoresLucIA.VIOLETA_NEON}Turno #{turno}{ColoresLucIA.RESET}"
        badge_ms = f"{ColoresLucIA.CIAN_ELECTRICO}{duracion_ms:.0f}ms{ColoresLucIA.RESET}"
        badge_mod = f"{ColoresLucIA.AMBAR_SOLAR}{modelo_id}{ColoresLucIA.RESET}"
        badge_deriva = f"{ColoresLucIA.MAGENTA_CUANTICO}Deriva: {deriva:.4f}{ColoresLucIA.RESET}"
        badge_sinapsis = f"{ColoresLucIA.VERDE_SINAPTICO}Sinapsis: {neuronas}{ColoresLucIA.RESET}"
        badge_cid = f"{ColoresLucIA.GRIS_METALLIC}IPFS: {cid_corte}{ColoresLucIA.RESET}"
        badge_costo = f"{ColoresLucIA.ESMERALDA_VIVO}$0.00 USD{ColoresLucIA.RESET}"

        print(
            f"  {Glifos.FLECHA} {badge_turno} │ {badge_ms} │ {badge_mod} │ "
            f"{badge_deriva} │ {badge_sinapsis} │ {badge_cid} │ {badge_costo}\n"
        )

    @staticmethod
    def renderizar_notificacion_bloque(indice: int, hash_bloque: str) -> None:
        """Muestra una notificacion destacada al minar un nuevo bloque en la blockchain."""
        h_vis = hash_bloque[:32] + "..."
        print(
            f"  {ColoresLucIA.BOLD}{ColoresLucIA.VERDE_SINAPTICO}◈ BLOQUE #{indice} MINADO PoNL{ColoresLucIA.RESET} "
            f"{ColoresLucIA.DORADO_BLOCK}[Hash: {h_vis}]{ColoresLucIA.RESET}"
        )

    @staticmethod
    def renderizar_panel_ayuda() -> None:
        """Despliega una guia visual de comandos con colores distintivos."""
        ancho = obtener_ancho_consola()
        print(f"\n{ColoresLucIA.CIAN_ELECTRICO}╭─{ColoresLucIA.BOLD} COMANDOS DISPONIBLES EN CONSOLA {ColoresLucIA.CIAN_ELECTRICO}{Glifos.LINEA_H * (ancho - 37)}╮{ColoresLucIA.RESET}")
        comandos = [
            ("estado / status", "Muestra el estado de la blockchain, 50 neuronas y auditoria de costo."),
            ("modelos / free", "Lista todos los modelos gratuitos de OpenRouter activos."),
            ("modelo <id>", "Cambia en caliente al modelo :free especificado sin reiniciar."),
            ("minar / mine", "Consolida las transacciones pendientes en un bloque PoNL."),
            ("salir / exit", "Guarda el checkpoint sinaptico en PSNRL y sincroniza con IPFS."),
        ]
        for cmd, desc in comandos:
            c_str = f"  {ColoresLucIA.AMBAR_SOLAR}{cmd:<18}{ColoresLucIA.RESET} {ColoresLucIA.GRIS_METALLIC}→{ColoresLucIA.RESET} {ColoresLucIA.BLANCO_LUMINOSO}{desc}{ColoresLucIA.RESET}"
            pad = ancho - 2 - longitud_visual(c_str)
            print(f"{ColoresLucIA.CIAN_ELECTRICO}│{ColoresLucIA.RESET}{c_str}{' ' * max(0, pad)}{ColoresLucIA.CIAN_ELECTRICO}│{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.CIAN_ELECTRICO}╰{Glifos.LINEA_H * (ancho - 2)}╯{ColoresLucIA.RESET}\n")

    @staticmethod
    def renderizar_tarjeta_sistema(titulo: str, datos: Dict[str, Any]) -> None:
        """Despliega una tarjeta estructurada de clave-valor para diagnosticos."""
        ancho = obtener_ancho_consola()
        encabezado = f" {titulo} "
        pad_h = ancho - len(encabezado) - 3
        print(f"{ColoresLucIA.DORADO_BLOCK}╭─{ColoresLucIA.BOLD}{encabezado}{ColoresLucIA.DORADO_BLOCK}{Glifos.LINEA_H * max(2, pad_h)}╮{ColoresLucIA.RESET}")
        for clave, valor in datos.items():
            linea = f"  {ColoresLucIA.CIAN_ELECTRICO}{clave:<24}{ColoresLucIA.RESET} : {ColoresLucIA.BLANCO_LUMINOSO}{valor}{ColoresLucIA.RESET}"
            pad = ancho - 2 - longitud_visual(linea)
            print(f"{ColoresLucIA.DORADO_BLOCK}│{ColoresLucIA.RESET}{linea}{' ' * max(0, pad)}{ColoresLucIA.DORADO_BLOCK}│{ColoresLucIA.RESET}")
        print(f"{ColoresLucIA.DORADO_BLOCK}╰{Glifos.LINEA_H * (ancho - 2)}╯{ColoresLucIA.RESET}")


# ─── ACCESOS DIRECTOS Y EXPORTACIONES ────────────────────────────────────────
