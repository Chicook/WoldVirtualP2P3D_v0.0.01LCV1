"""
lucIA.CORE.text_encoder - Codificador semántico determinista y ligero para LucIA
================================================================================

Extrae un vector de características semánticas (1, 4) con significado lingüístico real:
- v[0] = Densidad semántica (riqueza de vocabulario único vs total palabras)
- v[1] = Carga emocional (intensidad detectada, signos !, ?, mayúsculas, palabras clave emotivas)
- v[2] = Complejidad léxica (longitud promedio de palabras)
- v[3] = Densidad de información (ratio palabras de contenido vs stopwords)
"""

import re
import numpy as np
from typing import Optional, Tuple

# Palabras funcionales en español (stopwords básicas)
STOPWORDS_ES = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "en", "con", "por", "para", "sin", "sobre",
    "y", "o", "u", "e", "ni", "pero", "sino", "aunque", "porque",
    "que", "qué", "si", "no", "se", "me", "te", "le", "lo", "les",
    "a", "ante", "bajo", "cabe", "contra", "desde", "hacia", "hasta",
    "son", "es", "era", "fue", "ser", "estar", "ha", "he", "han",
    "su", "sus", "mi", "mis", "tu", "tus", "ya", "muy", "más", "menos",
    "también", "sí", "esto", "esta", "estos", "estas", "ese", "esa",
    "yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas"
})

# Palabras con alta carga emocional
EMOCIONALES_ES = frozenset({
    "genial", "increíble", "terrible", "horrible", "fantástico", "fatal",
    "amor", "odio", "miedo", "feliz", "triste", "alegre", "enojado",
    "gracias", "por favor", "perdón", "urgente", "ayuda", "problema",
    "perfecto", "dañado", "error", "lucia", "cerebro", "inteligencia",
    "futuro", "vida", "maravilla", "alegría", "furia", "éxito"
})


class TextEncoder:
    """
    Encoder semántico ligero para alimentar las neuronas de entrada (ENRN) y memoria (RF_SL)
    en Celebro. Produce vectores (1, 4) acotados en [0.0, 1.0].
    """

    def __init__(self):
        pass

    def encode(self, texto: str) -> np.ndarray:
        """Convierte una cadena de texto en un vector semántico (1, 4)."""
        if not texto or not texto.strip():
            return np.zeros((1, 4), dtype=np.float32)

        tokens_brutos = re.findall(r'\b\w+\b', texto.lower())
        total_tokens = max(1, len(tokens_brutos))

        # 1. Densidad semántica: vocabulario único / total palabras
        vocab_unico = len(set(tokens_brutos))
        densidad_semantica = min(1.0, vocab_unico / total_tokens)

        # 2. Carga emocional
        cuenta_emocionales = sum(1 for t in tokens_brutos if t in EMOCIONALES_ES)
        exclamaciones = len(re.findall(r'[!¡]', texto))
        interrogaciones = len(re.findall(r'[?¿]', texto))
        mayusculas = len(re.findall(r'\b[A-ZÁÉÍÓÚÑ]{2,}\b', texto))
        score_emocional = (cuenta_emocionales * 2.0 + exclamaciones * 1.5 + interrogaciones * 1.0 + mayusculas * 1.5)
        carga_emocional = min(1.0, score_emocional / max(5.0, total_tokens * 0.3))

        # 3. Complejidad léxica (longitud media de palabra normalizada)
        longitud_media = sum(len(t) for t in tokens_brutos) / total_tokens
        # normalizado entre 3 letras (0.0) y 10 letras (1.0)
        complejidad_lexica = min(1.0, max(0.0, (longitud_media - 3.0) / 7.0))

        # 4. Densidad de información (palabras de contenido no-stopwords)
        palabras_contenido = sum(1 for t in tokens_brutos if t not in STOPWORDS_ES)
        densidad_informacion = min(1.0, palabras_contenido / total_tokens)

        vector = np.array([[
            densidad_semantica,
            carga_emocional,
            complejidad_lexica,
            densidad_informacion
        ]], dtype=np.float32)

        return vector

    def encode_pair(self, prompt: str, respuesta: str) -> np.ndarray:
        """
        Combina la interacción (pregunta del usuario y respuesta de LucIA)
        en un vector consolidado (1, 4) para retroalimentar Celebro.
        """
        v_prompt = self.encode(prompt)
        v_resp = self.encode(respuesta)
        # Promedio ponderado (40% prompt, 60% respuesta del modelo)
        v_comb = (v_prompt * 0.4) + (v_resp * 0.6)
        return v_comb.astype(np.float32)
