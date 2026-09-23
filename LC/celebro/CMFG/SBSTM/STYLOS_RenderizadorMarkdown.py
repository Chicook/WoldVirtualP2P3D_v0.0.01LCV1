from typing import Dict, List

class RenderizadorMarkdown:
    """Convierte texto plano y Markdown basico en salida coloreada y estructurada."""

    @classmethod
    def colorear_sintaxis(cls, linea: str) -> str:
        """Aplica estilos visuales a negritas, codigos, cursivas y listas."""
        # 1. Bloques de codigo inline `codigo`
        linea = re.sub(
            r"`([^`]+)`",
            rf"{ColoresLucIA.AMBAR_SOLAR}\1{ColoresLucIA.RESET}",
            linea,
        )
        # 2. Negrita **texto**
        linea = re.sub(
            r"\*\*([^*]+)\*\*",
            rf"{ColoresLucIA.BOLD}{ColoresLucIA.BLANCO_LUMINOSO}\1{ColoresLucIA.RESET}",
            linea,
        )
        # 3. Cursiva *texto*
        linea = re.sub(
            r"\*([^*]+)\*",
            rf"{ColoresLucIA.ITALIC}{ColoresLucIA.CIAN_ELECTRICO}\1{ColoresLucIA.RESET}",
            linea,
        )
        # 4. Encabezados ###
        if linea.strip().startswith("###"):
            titulo = linea.strip().lstrip("#").strip()
            return f"\n  {ColoresLucIA.BOLD}{ColoresLucIA.VIOLETA_NEON}◈ {titulo}{ColoresLucIA.RESET}"
        if linea.strip().startswith("##"):
            titulo = linea.strip().lstrip("#").strip()
            return f"\n  {ColoresLucIA.BOLD}{ColoresLucIA.CIAN_ELECTRICO}✦ {titulo}{ColoresLucIA.RESET}"
        if linea.strip().startswith("#"):
            titulo = linea.strip().lstrip("#").strip()
            return f"\n  {ColoresLucIA.BOLD}{ColoresLucIA.DORADO_BLOCK}━ {titulo} ━{ColoresLucIA.RESET}"

        # 5. Listas de vinetas
        if re.match(r"^\s*[-*•]\s+", linea):
            contenido = re.sub(r"^\s*[-*•]\s+", "", linea)
            return f"    {ColoresLucIA.VIOLETA_NEON}•{ColoresLucIA.RESET} {contenido}"

        # 6. Listas numeradas
        m_num = re.match(r"^(\s*)(\d+)\.\s+(.*)$", linea)
        if m_num:
            esp, num, cont = m_num.groups()
            return f"{esp}  {ColoresLucIA.CIAN_ELECTRICO}{num}.{ColoresLucIA.RESET} {cont}"

        return linea

    @classmethod
    def renderizar_tabla(cls, lineas_tabla: List[str], ancho_max: int = 80) -> str:
        """Parsea y genera una tabla Unicode alineada con encabezados destacados."""
        filas = []
        for l in lineas_tabla:
            partes = [c.strip() for c in l.split("|")[1:-1]]
            if partes and not all(set(c) <= {"-", ":", " "} for c in partes):
                filas.append(partes)

        if not filas:
            return ""

        num_cols = max(len(f) for f in filas)
        anchos = [0] * num_cols
        for f in filas:
            for i in range(min(num_cols, len(f))):
                anchos[i] = max(anchos[i], longitud_visual(f[i]))

        lineas_salida = []
        sep_horiz = "  " + Glifos.ESQ_ARR_IZQ + (Glifos.LINEA_H * (sum(anchos) + (num_cols * 3) - 1)) + Glifos.ESQ_ARR_DER
        lineas_salida.append(f"{ColoresLucIA.GRIS_METALLIC}{sep_horiz}{ColoresLucIA.RESET}")

        for idx_f, f in enumerate(filas):
            celdas_fmt = []
            for i in range(num_cols):
                val = f[i] if i < len(f) else ""
                pad = anchos[i] - longitud_visual(val)
                val_col = cls.colorear_sintaxis(val)
                celdas_fmt.append(f"{val_col}{' ' * pad}")

            color_borde = ColoresLucIA.VIOLETA_NEON if idx_f == 0 else ColoresLucIA.GRIS_METALLIC
            linea_str = f"  {color_borde}{Glifos.LINEA_V}{ColoresLucIA.RESET} " + f" {color_borde}{Glifos.LINEA_V}{ColoresLucIA.RESET} ".join(celdas_fmt) + f" {color_borde}{Glifos.LINEA_V}{ColoresLucIA.RESET}"
            lineas_salida.append(linea_str)

            if idx_f == 0:
                sep_div = "  " + Glifos.CRUZ_IZQ + (Glifos.LINEA_H * (sum(anchos) + (num_cols * 3) - 1)) + Glifos.CRUZ_DER
                lineas_salida.append(f"{ColoresLucIA.VIOLETA_NEON}{sep_div}{ColoresLucIA.RESET}")

        sep_fin = "  " + Glifos.ESQ_ABA_IZQ + (Glifos.LINEA_H * (sum(anchos) + (num_cols * 3) - 1)) + Glifos.ESQ_ABA_DER
        lineas_salida.append(f"{ColoresLucIA.GRIS_METALLIC}{sep_fin}{ColoresLucIA.RESET}")
        return "\n".join(lineas_salida)


# ─── MOTOR PRINCIPAL DE TARJETAS Y PRESENTACION ──────────────────────────────
