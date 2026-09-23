class AnalizadorEstructuraParrafos:
    """
    Clasifica la topologia del texto fuente (listas, codigo o narrativa)
    para modular la disposicion sintactica en la reformulacion.
    """

    @staticmethod
    def tipificar(texto: str) -> str:
        """Identifica si el texto es lista, codigo, frase o prosa corrida."""
        lineas = [ln.strip() for ln in texto.split("\n") if ln.strip()]
        if not lineas:
            return "vacio"
        n_listas = sum(1 for ln in lineas if re.match(r"^(?:[-*•]|\d+[.)])\s+", ln))
        if n_listas >= max(2, len(lineas) // 2):
            return "lista_estructurada"
        if any(ln.startswith("```") or "def " in ln or "class " in ln for ln in lineas):
            return "codigo_tecnico"
        if len(lineas) == 1 and len(texto) < 130:
            return "asercion_corta"
        return "prosa_cognitiva"

    @staticmethod
    def modular_simbolos(texto: str, tipo_estructura: str) -> str:
        """Adapta marcadores de lista o cabeceras al formato neural de LucIA."""
        if tipo_estructura == "lista_estructurada":
            texto = re.sub(r"^[\t ]*[-*•]\s*", "  ◆ ", texto, flags=re.MULTILINE)
            texto = re.sub(r"^[\t ]*(\d+)[.)]\s*", r"  [\1] ", texto, flags=re.MULTILINE)
        return texto


