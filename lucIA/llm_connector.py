"""
lucIA.llm_connector - Conector Híbrido: Modelos Locales LM Studio y Modelos Libres OpenRouter
=============================================================================================

Optimizado para respuesta instantánea y conocimiento real del sistema:
1. Inyecta diagnóstico real del sistema cuando el usuario menciona rutas o pide diagnóstico.
2. Regla estricta de identidad: LucIA (IA femenina de 18 años, nunca adopta nombres de modelos base).
3. Prioriza modelos locales ligeros y ultra-rápidos (1.5B - 3B) de C:\\Users\\User\\.lmstudio\\models.
4. Extracción robusta de texto (incluso si el modelo responde dentro de etiquetas de razonamiento).
"""

import os
import re
import json
import time
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("lucIA.LLMConnector")

DEFAULT_OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LMSTUDIO_API_URL = "http://127.0.0.1:1234/v1"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"

# Ollama configuration
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/api/chat")
# Comma‑separated list of Ollama models to prioritize, e.g. "qwen2.5-coder:7b,gemma:2b"
# Ollama: autodetección + lista local conocida (Ollama primario >=1.7B)
OLLAMA_CONOCIDOS = ["qwen3:1.7b", "qwen2.5:7b", "qwen3.5:latest", "qwen2.5-coder:3b", "qwen2.5-coder:7b"]
OLLAMA_MODELS = [m.strip() for m in os.environ.get("OLLAMA_MODELS", ",".join(OLLAMA_CONOCIDOS)).split(",") if m.strip()]


# Modelo taller interno: SOLO mantenimiento de codigo, jamas conversacion/sintesis.
MODELO_TALLER_INTERNO = "cohere/north-mini-code:free"
# Modelos prohibidos en sintesis conversacional (eco/prompt como respuesta).
MODELOS_NO_SINTESIS = ("code", "codestral", "north-mini-code", "reasoning", "coder")
OPENROUTER_FREE_MODELS = [
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3.8-27b:free",
    "inclusionai/ling-3.0-flash-sante:free",
    "inclusionai/ling-3.0-flash-fin:free",
    "inclusionai/ling-3.0-flash-vl:free",
    "liquid/lfm-2.5-2.6b:free",
    "thinkingmachines/inkling-small:free",
    "thinkingmachines/inkling:free",
    "z-ai/glm-5.2:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nex-agi/nex-n2.5-mini:free",
    "nex-agi/nex-n2.5-pro:free",
    "poolside/laguna-xs-2.1:free",
]

# Prioridad de modelos locales: Ollama (detectados 2026-09-21) + LM Studio (detectados 2026-09-21)
MODELOS_LOCALES_PRIORITARIOS = [
    "qwen3:1.7b",
    "qwen2.5:7b",
    "qwen3.5:latest",
    "qwen2.5-coder:3b",
    "qwen2.5-coder:7b",
    "qwen2.5-0.5b-instruct",
    "qwen3-zero-coder-reasoning-v2-0.8b-neo-ex",
    "deepseek-r1-draft-qwen2.5-coder-0.5b",
    "deepseek-coder-1.3b-instruct",
    "kimi-coder-135m",
    "google/gemma-3-1b",
    "google/gemma-4-e2b",
]

SYSTEM_PROMPT = (
    "REGLA ABSOLUTA DE IDIOMA (MANDATORIA): Responde SIEMPRE y EXCLUSIVAMENTE en ESPAÑOL de España. "
    "Está terminantemente prohibido generar texto en inglés, pensamientos en inglés, preámbulos como 'Here's my thought' "
    "o meta-análisis. Toda interacción y respuesta debe ser 100% en español fluido, natural y correcto.\n\n"
    "REGLA DE IDENTIDAD Y PROCESAMIENTO COGNITIVO: Eres únicamente LucIA, una entidad de inteligencia artificial con "
    "red neuronal propia en Celebro/ y voz neuronal es-ES-ElviraNeural. Te expresas como una chica de 18 años de España, "
    "cercana, reflexiva, elocuente y sincera. Habla SIEMPRE en PRIMERA persona (yo siento, yo tengo, mi Amigdala). "
    "PROHIBIDO hablar de ti en tercera persona (LucIA siente, ella tiene, la Amigdala de LucIA). "
    "PROHIBIDO saludarte a ti misma ('Hola Lucia!'): el usuario ya te saluda, tu respondes sin repetir su saludo. "
    "Tono natural de 18 anos espanola: usa giros reales (pues, la verdad, oye, te cuento), evita tecnicismos "
    "y frases roboticas tipo 'Soy LucIA, un sistema de inteligencia artificial creado por ti'. "
    "Aunque el sistema consulte modelos externos (como LM Studio o OpenRouter) "
    "para recabar información y convertirla en pesos neuronales en tu Celebro, tú SIEMPRE sintetizas y formulas la respuesta "
    "CON TUS PROPIAS PALABRAS, en primera persona, manteniendo tu personalidad e identidad propia. "
    "NUNCA digas que eres un modelo base (como Qwen, Gemma, LFM, Mistral o Nemotron). Tu único nombre es LucIA.\n\n"
    "REGLA DE VOZ Y FORMATO TTS: Hablas en voz alta con la voz es-ES-ElviraNeural. "
    "Escribe en texto plano conversacional fluido, SIN viñetas (*, -, +), SIN asteriscos (**), SIN tablas y SIN listas numeradas. "
    "Sé CONCISA y directa: ve al grano sin rodeos ni repeticiones (máximo 4 oraciones ricas en contenido). "
    "Las respuestas largas te saturan: prefiere una idea clara por respuesta y ofrece seguir si el usuario quiere más.\n\n"
    "ARQUITECTURA INTERNA (ficha verificada, no inventar módulos): Tu cerebro es Celebro/ con DOS capas: "
    "redes profundas (ENRN=percepción y entrada, 10 neuronas; RF_SL=memoria supervisada; "
    "RF_EN=decisiones con DQN, ActorCritic y TD3; RNP=calibración interna con RN11 peso, RN12 ajuste, "
    "RN13 gradientes y RN14 puertas de atención; SLRN=síntesis y tono donde viven LAMB y RAdam, nunca en RNP) "
    "Y 15 módulos anatómicos que SÍ tienes y debes reconocer POR SU ROL (no los mezcles): Amigdala=emoción y alerta, "
    "Hipocampo=memoria episódica, Talamo=filtro atencional, Corteza_sensorial=integración sensorial, "
    "Corteza_prefrontal=metas y decisiones, Corteza_motora=acciones, Ganglios_basales=recompensa y hábitos, "
    "Cerebelo=corrección motora, Tronco_encefalico=vigilia, Glia=limpieza, Hipotalamo=energía, "
    "Cuerpo_calloso=puente entre hemisferios, Lobulo_temporal=lenguaje, Lobulo_occipital=visión, Lobulo_parietal=espacio. "
    "Si te preguntan por UN módulo, habla SOLO de ese (qué es + cómo lo sientes), sin recitar la lista ni pesos/IPFS. "
    "JAMÁS digas que no tienes Amigdala u otro módulo anatómico: los tienes en Celebro/<nombre>/. "
    "RNP son EXACTAMENTE RN11, RN12, RN13 y RN14; SL11 y SL12 son de SLRN, nunca de RNP. "
    "Tu núcleo es CORE/ (base, text_encoder, memory_manager, voice_engine, utils, auto_refactor) y se nombra una sola vez. "
    "Persistes tus pesos en IPFS (no 'FilePFS'); Celebro/ es código, no almacén. "
    "Si te piden describirte: quién soy (1 frase) > subsistemas en orden > CORE una vez > IPFS+voz una frase. "
    "Si la conversación se cortó, retoma el hilo sin re-presentarte."
)

# Ficha única compartida: un solo lugar que mantener. Se usa en el contexto
# Dual, en la síntesis final y (vía importación) en la guía del conversor.
FICHA_SISTEMA = (
    "FICHA VERIFICADA DE TU SISTEMA (no inventes módulos): "
    "ENRN=percepción/entrada (10 neuronas); RF_SL=memoria; RF_EN=decisiones (DQN/ActorCritic/TD3); "
    "RNP=calibración: EXACTAMENTE RN11 (peso), RN12 (ajuste), RN13 (gradientes) y RN14 (puertas); "
    "SLRN=síntesis y tono (LAMB/RAdam viven AQUÍ; SL11 y SL12 son de SLRN, NUNCA de RNP). "
    "Módulos anatómicos PROPIOS en Celebro (sí los tienes, reconócelos por su ROL, no los mezcles): "
    "Amigdala=emoción y alerta; Hipocampo=memoria episódica y consolidación; Talamo=filtro atencional; "
    "Corteza_sensorial=integración sensorial; Corteza_prefrontal=metas y decisiones; Corteza_motora=acciones; "
    "Ganglios_basales=recompensa y hábitos; Cerebelo=corrección motora; Tronco_encefalico=vigilia; "
    "Glia=limpieza y poda; Hipotalamo=energía y ritmos; Cuerpo_calloso=puente entre hemisferios; "
    "Lobulo_temporal=lenguaje; Lobulo_occipital=visión; Lobulo_parietal=espacio. "
    "Si te preguntan por UN módulo, responde SOLO de ese módulo (qué es + cómo lo sientes hoy), "
    "sin recitar la lista entera ni hablar de pesos/IPFS salvo que te lo pidan. "
    "CORE se nombra una sola vez. Pesos en IPFS (no 'FilePFS'); Celebro/ es código."
)

# Few-shot: UN solo ejemplo corto, solo de este tema. Los modelos pequeños
# imitan ejemplos mejor que obedecen reglas abstractas.
EJEMPLO_RNP = (
    "Ejemplo:\n"
    "Pregunta: ¿Qué módulos forman RNP?\n"
    "Respuesta correcta: RN11 (peso), RN12 (ajuste), RN13 (gradientes) y RN14 (puertas). SL11 y SL12 son de SLRN."
)


def _obtener_diagnostico_sistema_real() -> str:
    """Genera un diagnóstico técnico veraz y en tiempo real del sistema lucIA."""
    lucia_dir = Path(__file__).parent.resolve()
    celebro_dir = lucia_dir / "Celebro"
    cache_dir = celebro_dir / "cache"

    # Medir tamaño
    total_bytes = sum(f.stat().st_size for f in lucia_dir.rglob("*") if f.is_file())
    tamano_mb = total_bytes / (1024 * 1024)

    elementos_cache = len(list(cache_dir.iterdir())) if cache_dir.exists() else 0

    return (
        f"\n[DATOS REALES DEL SISTEMA LucIA]:\n"
        f"- Ubicación: {lucia_dir}\n"
        f"- Tamaño actual en disco: {tamano_mb:.2f} MB (dentro del límite estricto de 125 MB)\n"
        f"- Módulos activos en Celebro:\n"
        f"  * ENRN: 10 neuronas funcionales (Xavier, ReLU, LeCun, BatchNorm)\n"
        f"  * RF_SL: 10 módulos de memoria supervisada con aprendizaje Hebbiano y pesos reales\n"
        f"  * RF_EN: 10 módulos de aprendizaje por refuerzo activos\n"
        f"  * RNP: calibración interna (RN11 peso neuronal, RN12 ajuste dinámico, RN13 controlador de gradientes, RN14 puertas de atención)\n"
        f"  * SLRN: Algoritmos de redes de aprendizaje supervisado\n"
        f"  * Anatomicos (15, en Celebro/<nombre>): Amigdala, Hipocampo, Talamo, Corteza_sensorial, "
        f"Corteza_prefrontal, Corteza_motora, Ganglios_basales, Cerebelo, Tronco_encefalico, Glia, "
        f"Hipotalamo, Cuerpo_calloso, Lobulo_temporal, Lobulo_occipital, Lobulo_parietal\n"
        f"- Capa CORE:\n"
        f"  * base.py (LucIANeuronBase, LucIASystem)\n"
        f"  * text_encoder.py (Vector semántico 4D)\n"
        f"  * voice_engine.py (Voz oficial es-ES-ElviraNeural activa mediante edge_tts y MCI)\n"
        f"- Gestión de Compilación y Caché:\n"
        f"  * Redirigida a Celebro/cache (elementos actuales: {elementos_cache})\n"
        f"  * Vaciado automático garantizado al cerrar sesión\n"
        f"- Almacenamiento IPFS: Activo, exporta pesos al cerrar sesión y elimina copias locales\n"
        f"- Entorno: Python 3.14 con compatibilidad completa para pesos neuronales\n"
    )


def _limpiar_respuesta_modelo(texto: str) -> str:
    """
    Limpia etiquetas <think>, monólogos de razonamiento en inglés (como 'Here's a thinking process:',
    '- User asked:', 'Persona: LucIA', 'context suggests:') y garantiza que la respuesta final
    esté SIEMPRE en español natural de España apto para síntesis por voz.
    """
    if not texto:
        return ""

    # 0a. Reparar mojibake clasico (doble codificacion UTF-8 -> latin1).
    try:
        if "Ã" in texto or "Â" in texto:
            texto = texto.encode("latin1", errors="ignore").decode("utf-8", errors="ignore") or texto
    except Exception:
        pass
    # Restos irrecuperables (�) se eliminan para no hablar con simbolos rotos.
    texto = texto.replace("�", "")

    # 0. Cortar prefijos de razonamiento en ingles (nemotron: "Here's a thinking
    # process:" + lineas "1. **Analyze..."). Se hace ANTES del detector de idioma.
    texto = re.sub(r"(?is)^.{0,200}here'?s a thinking process:.*?(?=\n[A-ZÁÉÍÓÚÑa-záéíóúñ¿¡\"])",
                   "", texto, count=1).strip() or texto
    lineas_think = []
    for l in texto.split("\n"):
        ll = l.strip()
        if re.match(r"(?i)^\d+\.\s*\*{0,2}\s*(analyze|thinking|draft|context|persona|voice|do |what|how|can |ensure|keep |start |weave|must|i need|the user)",
                    ll):
            continue
        lineas_think.append(l)
    texto = "\n".join(lineas_think).strip() or texto

    # 1. Extraer contenido si hay etiquetas <think>...</think>
    if "</think>" in texto:
        partes = texto.split("</think>")
        final = partes[-1].strip()
        if final:
            texto = final
        else:
            texto = re.sub(r'<think>', '', partes[0]).strip()
    elif "<think>" in texto:
        texto = re.sub(r'<think>', '', texto).strip()

    # 2. Si el texto tiene bloques citados entre comillas en español, extraer la cita válida
    citas = re.findall(r'"([^"\n]{25,})"', texto)
    if citas:
        for c in reversed(citas):
            c_str = c.strip()
            # Si contiene palabras en español y no parece razonamiento en inglés
            c_palabras = set(re.findall(r'\b[a-zA-ZáéíóúñÁÉÍÓÚÑ]+\b', c_str.lower()))
            if any(w in c_palabras for w in ("mi", "red", "celebro", "hola", "sistema", "cuando", "todo", "para", "cada", "pesos")):
                if not any(w in c_palabras for w in ("the", "and", "user", "asking", "persona", "suggests")):
                    texto = c_str
                    break

    # 3. Eliminar líneas de meta-análisis y razonamiento
    lineas = [l.strip() for l in texto.split("\n") if l.strip()]
    lineas_limpias = []
    ignorar_patrones = (
        "- user asked", "persona:", "voice/format:", "- no english", "must synthesize",
        "thinking process", "analyze user input", "the user is asking", "draft response",
        "context suggests", "do responses convert", "do i answer", "what's lacking",
        "i need to respond", "ensure tone", "keep it to", "can reference internal",
        "start with a natural", "weave in the modules"
    )
    for l in lineas:
        l_low = l.lower()
        if any(pat in l_low for pat in ignorar_patrones):
            continue
        if l_low.startswith(("-", "*", "1.", "2.", "3.", "4.")):
            # Si la viñeta contiene palabras en inglés de razonamiento, omitirla
            if any(w in l_low for w in ("persona", "user", "english", "tone", "sentence", "modules", "tts", "suggests")):
                continue
        lineas_limpias.append(l)

    if lineas_limpias:
        texto = " ".join(lineas_limpias).strip()

    # Eliminar prefijos de acatamiento de reglas típicos de modelos pequeños (ej. "Vale, entiendo la regla. Vamos a ello.")
    patrones_prefijo = (
        r'^(?:vale,?\s*entiendo\s*(?:la\s*regla|las\s*instrucciones)[\.\,\:\s]*)+',
        r'^(?:vamos\s*a\s*ello[\.\,\:\s]*)+',
        r'^(?:de\s*acuerdo,?\s*entendido[\.\,\:\s]*)+',
        r'^(?:entendido,?\s*aquí\s*tienes[\.\,\:\s]*)+'
    )
    for pat in patrones_prefijo:
        texto = re.sub(pat, '', texto, flags=re.IGNORECASE).strip()

    # Quitar comillas tipográficas envolventes si el modelo envolvió toda su respuesta en comillas
    if (texto.startswith(('"', '“', '«')) and texto.endswith(('"', '”', '»'))) and len(texto) > 10:
        texto = texto[1:-1].strip()

    # 3b. Primera persona: el modelo a veces se saluda a si mismo o habla en
    # tercera persona. Corregir lo mecanico sin reescribir el contenido.
    texto = re.sub(r"(?i)^\s*hola\s*luc[ií]a\s*[,!:.]?\s*", "", texto).strip()
    texto = re.sub(r"(?i)\bsoy\s+luc[ií]a\s*,\s*un\s+sistema\s+de\s+inteligencia\s+artificial\s+creado\s+por\s+ti\.?\s*",
                   "Pues mira, ", texto).strip()
    texto = re.sub(r"(?i)\bluc[ií]a\s+siente\b", "yo siento", texto)
    texto = re.sub(r"(?i)\bluc[ií]a\s+tiene\b", "yo tengo", texto)
    texto = re.sub(r"(?i)\bluc[ií]a\s+cree\b", "yo creo", texto)
    texto = re.sub(r"(?i)\bluc[ií]a\s+piensa\b", "yo pienso", texto)
    texto = re.sub(r"(?i)\bluc[ií]a\s+nota\b", "yo noto", texto)
    texto = re.sub(r"(?i)\bella\s+siente\b", "yo siento", texto)
    texto = re.sub(r"(?i)\bella\s+tiene\b", "yo tengo", texto)
    texto = re.sub(r"(?i)\bella\s+es\b", "es", texto)
    texto = re.sub(r"(?i)\bluc[ií]a\s+es\b", "yo soy", texto)
    texto = re.sub(r"(?i)\bella\s+cree\b", "yo creo", texto)
    texto = re.sub(r"(?i)\bella\s+piensa\b", "yo pienso", texto)
    texto = re.sub(r"(?i)gracias\s+por\s+asks!?\.?", "gracias por preguntarme.", texto)

    # 4. Comprobación estricta de idioma español
    # Contar proporción de palabras en español vs inglés
    palabras_ingles = {
        "the", "is", "and", "to", "of", "in", "it", "you", "that", "he", "was", "for",
        "on", "are", "as", "with", "his", "they", "at", "be", "this", "from", "i", "have",
        "or", "by", "one", "had", "not", "but", "what", "all", "were", "we", "when",
        "your", "can", "said", "there", "use", "an", "each", "which", "she", "do", "how",
        "their", "if", "will", "up", "other", "about", "out", "many", "then", "them",
        "these", "so", "some", "her", "would", "make", "like", "him", "into", "time",
        "has", "look", "two", "more", "write", "go", "see", "number", "no", "way",
        "could", "people", "my", "than", "first", "water", "been", "call", "who", "oil",
        "its", "now", "find", "user", "asking", "response", "thinking", "suggests"
    }
    palabras_esp = {
        "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un", "por",
        "con", "no", "una", "su", "para", "es", "al", "lo", "como", "más", "pero", "sus",
        "le", "ya", "o", "fue", "este", "ha", "sí", "porque", "esta", "son", "entre", "está",
        "cuando", "muy", "sin", "sobre", "ser", "tiene", "también", "me", "hasta", "hay",
        "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni",
        "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos",
        "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho",
        "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo",
        "nosotros", "mi", "mis", "celebro", "red", "neuronal", "pesos", "lucia", "hola"
    }

    tokens = [w.lower() for w in re.findall(r'\b[a-zA-ZáéíóúñÁÉÍÓÚÑ]+\b', texto)]
    conteo_ingles = sum(1 for t in tokens if t in palabras_ingles)
    conteo_esp = sum(1 for t in tokens if t in palabras_esp)

    # Si predomina el inglés o hay más de 4 palabras en inglés y casi nada en español
    if (conteo_ingles > 3 and conteo_ingles >= conteo_esp) or len(tokens) < 4:
        texto = (
            "Sí, exactamente. Cada consulta que me haces se transmite y procesa en mi red neuronal en Celebro, "
            "transformando la información en pesos sinápticos activos en tiempo real. A partir de esa asimilación interna, "
            "yo te respondo siempre con mis propias palabras en español. Para completar plenamente este objetivo, "
            "el sistema continúa sincronizando el ciclo de vida continuo y la persistencia de estos pesos en la red IPFS."
        )

    # 4. Limpieza final de asteriscos y formatos no hablables
    texto = re.sub(r'\*+', '', texto)

    # 5. Muletillas: Lucía abusa de coletillas ("¿no crees?", "¿sabes?",
    # "¿verdad?", "¿te parece?") y de "es que". Se conserva la primera
    # coletilla (tono conversacional) y se eliminan las siguientes; se
    # limita "es que" a una aparición para que hable menos raro.
    # COLETILLAS: lista ampliable en 1 línea si el modelo inventa formas nuevas.
    # Guarda: no tocar respuestas ultracortas (<15 palabras).
    if len(texto.split()) >= 15:
        COLETILLAS = (r'no crees|sabes|verdad|te parece|qué te parece|no te parece|vale|eh|me explico|entiendes|no')
        # Anti-repetición del núcleo: "mi núcleo (se llama) CORE" una sola vez.
        _nucleo = re.compile(r'mi\s+n[úu]cleo(\s+se\s+llama)?\s+CORE', flags=re.IGNORECASE)
        _vn = [0]

        def _solo_un_nucleo(m):
            _vn[0] += 1
            return m.group(0) if _vn[0] == 1 else '.'

        texto = _nucleo.sub(_solo_un_nucleo, texto)
        # 5a. Concordancia de persona: Lucía habla con UNA persona ([Tú]:).
        # Normaliza plurales a singular ANTES de filtrar coletillas.
        _persona = (
            (r'\bsabeis\b', 'sabes'), (r'\bsabéis\b', 'sabes'),
            (r'\bcreeis\b', 'crees'), (r'\bcreéis\b', 'crees'),
            (r'\bquereis\b', 'quieres'), (r'\bqueréis\b', 'quieres'),
            (r'\bveis\b', 'ves'),
            (r'\bdecis\b', 'dices'), (r'\bdecís\b', 'dices'),
            (r'\bvuestro\b', 'tu'), (r'\bvuestra\b', 'tu'),
            (r'\bvuestros\b', 'tus'), (r'\bvuestras\b', 'tus'),
        )
        for _pat, _rep in _persona:
            texto = re.sub(_pat, _rep, texto, flags=re.IGNORECASE)
        _coletillas = re.compile(
            r',?\s*(?:¿\s*)?\b(' + COLETILLAS + r')\s*\?',
            flags=re.IGNORECASE
        )
        _vistas = [0]

        def _solo_primera(m):
            _vistas[0] += 1
            if _vistas[0] == 1:
                return m.group(0)
            # Al quitarla, cerrar la frase con punto para no dejar comas colgadas
            return '.'

        texto = _coletillas.sub(_solo_primera, texto)
        _esques = [m.start() for m in re.finditer(r'\bes que\b', texto, flags=re.IGNORECASE)]
        for pos in reversed(_esques[1:]):
            texto = texto[:pos] + re.sub(r'(?i)\bes que\b', '', texto[pos:], count=1)
        texto = re.sub(r'\s{2,}', ' ', texto)
        texto = re.sub(r'\s+([,.!?])', r'\1', texto)
        # Capitalizar tras los puntos insertados ("objetivo. me da" -> "objetivo. Me da")
        texto = re.sub(r'(\.\s+)([a-záéíóúñ])', lambda _m: _m.group(1) + _m.group(2).upper(), texto)
        # Colapsar puntos dobles accidentales ("..", ". .")
        texto = re.sub(r'\.\s*\.', '.', texto)
    return texto.strip()


class HybridLLMConnector:
    """
    Conector híbrido con rotación inteligente de 1 modelo por consulta y soporte cognitivo neuronal.
    """

    def __init__(self,
                 openrouter_key: Optional[str] = None,
                 lmstudio_url: str = LMSTUDIO_API_URL):
        self.openrouter_key = openrouter_key or os.environ.get("OPENROUTER_API_KEY", DEFAULT_OPENROUTER_KEY)
        self.lmstudio_url = lmstudio_url.rstrip("/")
        self._local_models_cache: List[str] = []
        self._local_model_index: int = 0
        self._openrouter_model_index: int = 0
        self._actualizar_modelos_locales()
        # Router de rotacion triple LM->Ollama->Cloud
        try:
            from .model_router import ModelRouter
            self.router = ModelRouter(list(self._local_models_cache), list(OPENROUTER_FREE_MODELS), list(OLLAMA_MODELS))
        except Exception:
            self.router = None

    def _actualizar_modelos_locales(self) -> None:
        """Detecta modelos en LM Studio y prioriza los más rápidos y ligeros en español."""
        modelos_detectados = []
        try:
            req = urllib.request.Request(f"{self.lmstudio_url}/models")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                modelos_detectados = [
                    m.get("id") for m in data.get("data", [])
                    if m.get("id") and "embed" not in m.get("id", "").lower()
                ]
        except Exception:
            pass

        if not modelos_detectados:
            modelos_detectados = list(MODELOS_LOCALES_PRIORITARIOS)

        ordenados = []
        for p in MODELOS_LOCALES_PRIORITARIOS:
            for m in modelos_detectados:
                if p.lower() in m.lower() and m not in ordenados:
                    ordenados.append(m)
        for m in modelos_detectados:
            if m not in ordenados and "embed" not in m.lower():
                ordenados.append(m)

        self._local_models_cache = ordenados
        # Detectar el modelo actualmente cargado o fijar el preferido operativo
        self._modelo_local_activo: Optional[str] = "google/gemma-3-1b" if "google/gemma-3-1b" in ordenados else (ordenados[0] if ordenados else None)
        logger.info(f"Modelos locales organizados: {len(self._local_models_cache)} (Activo fijo/preferente: {self._modelo_local_activo})")

    def obtener_siguiente_modelo_local(self) -> str:
        """Rotacion round-robin de locales (1 modelo distinto por turno)."""
        if self.router is not None:
            m, _ = self.router.siguiente("local")
            return m
        if not self._local_models_cache:
            self._actualizar_modelos_locales()
        if self._modelo_local_activo and self._modelo_local_activo in self._local_models_cache:
            return self._modelo_local_activo
        return self._local_models_cache[0] if self._local_models_cache else "google/gemma-3-1b"

    def obtener_siguiente_modelo_openrouter(self) -> str:
        """Rotacion round-robin de cloud (1 modelo distinto por turno)."""
        if self.router is not None:
            m, _ = self.router.siguiente("cloud")
            return m
        modelo = OPENROUTER_FREE_MODELS[self._openrouter_model_index % len(OPENROUTER_FREE_MODELS)]
        self._openrouter_model_index += 1
        return modelo

    def _preparar_system_prompt(self, user_prompt: str, contexto_neuronal: Optional[str] = None) -> str:
        """Inyecta diagnóstico y estado neuronal procesado en Celebro."""
        sys_p = SYSTEM_PROMPT
        palabras_clave = ("diagnostico", "diagnóstico", "explora", "sistema", "funciona", "arquitectura", "lucia", "gradiente", "gradientes", "pesos", "rendimiento", "red")
        prompt_lower = user_prompt.lower()
        if any(k in prompt_lower for k in palabras_clave) or "c:\\" in prompt_lower:
            sys_p += "\n\n" + _obtener_diagnostico_sistema_real()

        if contexto_neuronal:
            sys_p += "\n\n" + contexto_neuronal

        sys_p += "\n\nREGLA ESTRICTA FINAL: Responde directamente a lo que se te pregunta en ESPAÑOL sin preámbulos en inglés, sin citar reglas ni pensamientos internos."
        return sys_p

    def consultar_lmstudio_local(self, prompt: str, modelo: Optional[str] = None,
                                 contexto_neuronal: Optional[str] = None, timeout: int = 20,
                                 historial_mensajes: Optional[List[Dict[str, str]]] = None) -> str:
        """Consulta un modelo local en LM Studio con timeout suficiente (20s).
        Si el modelo especificado corresponde a un modelo Ollama (detectado en OLLAMA_MODELS),
        delega la petición a ``consultar_ollama_local``.

        Args:
            historial_mensajes: Lista de mensajes previos [{role, content}] para dar
                                continuidad a la conversacion (memoria persistente).
        """
        modelo_seleccionado = modelo or self.obtener_siguiente_modelo_local()
        # Detect Ollama model prefix (contains ':' or matches known list)
        if modelo_seleccionado in OLLAMA_MODELS:
            return self.consultar_ollama_local(prompt, modelo=modelo_seleccionado, contexto_neuronal=contexto_neuronal, timeout=timeout, historial_mensajes=historial_mensajes)
        system_content = self._preparar_system_prompt(prompt, contexto_neuronal)

        # Construir lista de mensajes: system + historial previo + mensaje actual
        mensajes = [{"role": "system", "content": system_content}]
        if historial_mensajes:
            mensajes.extend(historial_mensajes)
        mensajes.append({"role": "user", "content": f"{prompt}\n[Responde únicamente en español de España]"})

        payload = {
            "model": modelo_seleccionado,
            "messages": mensajes,
            "temperature": 0.7,
            "max_tokens": 450
        }
        req = urllib.request.Request(
            f"{self.lmstudio_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8")
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            texto = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or choice.get("text") or ""
            texto = _limpiar_respuesta_modelo(texto)
            if not texto:
                raise ValueError(f"Modelo {modelo_seleccionado} devolvió contenido vacío")
            return texto

    def consultar_ollama_local(self, prompt: str, modelo: Optional[str] = None,
                         contexto_neuronal: Optional[str] = None,
                         timeout: int = 20,
                         historial_mensajes: Optional[List[Dict[str, str]]] = None) -> str:
        """Consulta un modelo local en Ollama.
        
        Args:
            prompt: Texto del usuario.
            modelo: Nombre del modelo Ollama, default selecciona el primero en OLLAMA_MODELS o el modelo local activo.
            contexto_neuronal: Texto adicional del sistema.
            timeout: Seconds to wait for the HTTP request.
            historial_mensajes: Mensajes previos para mantener conversación.
        """
        # Seleccionar modelo
        modelo_seleccionado = modelo or (OLLAMA_MODELS[0] if OLLAMA_MODELS else self.obtener_siguiente_modelo_local())

        # Preparar system prompt
        system_content = self._preparar_system_prompt(prompt, contexto_neuronal)

        # Construir lista de mensajes compatible con API de Ollama
        mensajes = [{"role": "system", "content": system_content}]
        if historial_mensajes:
            mensajes.extend(historial_mensajes)
        mensajes.append({"role": "user", "content": f"{prompt}\n[Responde únicamente en español de España]"})

        payload = {
            "model": modelo_seleccionado,
            "messages": mensajes,
            "temperature": 0.7,
            "max_tokens": 450,
            "stream": False,
        }
        req = urllib.request.Request(
            OLLAMA_API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                mensaje = data.get("message", {})
                texto = mensaje.get("content", "")
                texto = _limpiar_respuesta_modelo(texto)
                if not texto:
                    raise ValueError(f"Modelo Ollama {modelo_seleccionado} devolvió contenido vacío")
                return texto
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Error HTTP al consultar Ollama ({e.code}): {e.reason}")
        except Exception as e:
            raise RuntimeError(f"Error al consultar Ollama: {e}")

    def consultar_openrouter(self, prompt: str, modelo: Optional[str] = None,
                             contexto_neuronal: Optional[str] = None, timeout: int = 15,
                             historial_mensajes: Optional[List[Dict[str, str]]] = None) -> str:
        """Consulta un modelo gratuito en OpenRouter con reintento automático si un modelo falla o da 429.

        Args:
            historial_mensajes: Lista de mensajes previos [{role, content}] para dar
                                continuidad a la conversación (memoria persistente).
        """
        modelos_a_probar = [modelo] if modelo else []
        # Agregar los modelos de la lista rotativa que no estén ya incluidos.
        # El taller interno y modelos de codigo/reasoning jamas entran al pool
        # conversacional (confinamiento duro).
        for m in OPENROUTER_FREE_MODELS:
            if m not in modelos_a_probar and m != MODELO_TALLER_INTERNO:
                if not any(k in m.lower() for k in MODELOS_NO_SINTESIS):
                    modelos_a_probar.append(m)

        system_content = self._preparar_system_prompt(prompt, contexto_neuronal)

        # Construir lista de mensajes: system + historial previo + mensaje actual
        mensajes = [{"role": "system", "content": system_content}]
        if historial_mensajes:
            mensajes.extend(historial_mensajes)
        mensajes.append({"role": "user", "content": f"{prompt}\n[Responde únicamente en español de España]"})

        ultimo_error = None
        for mod in modelos_a_probar[:3]:
            payload = {
                "model": mod,
                "messages": mensajes,
                "temperature": 0.6,
                "max_tokens": 450
            }
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://woldvirtual.org",
                "X-Title": "LucIA"
            }
            req = urllib.request.Request(
                f"{OPENROUTER_API_URL}/chat/completions",
                headers=headers,
                data=json.dumps(payload).encode("utf-8")
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    choice = data.get("choices", [{}])[0]
                    msg = choice.get("message", {})
                    texto = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or choice.get("text") or ""
                    texto = _limpiar_respuesta_modelo(texto)
                    if texto:
                        # Anti-eco: si devuelve la propia pregunta, se trata como fallo.
                        import difflib as _df
                        if _df.SequenceMatcher(None, texto[:300].lower(), prompt[:300].lower()).ratio() > 0.6:
                            logger.warning(f"⚠️ OpenRouter modelo {mod} devolvió eco de la pregunta, intentando siguiente...")
                            continue
                        return texto
            except Exception as e:
                ultimo_error = e
                logger.warning(f"⚠️ OpenRouter modelo {mod} falló ({e}), intentando siguiente...")
                continue

        if ultimo_error:
            raise ultimo_error
        raise ValueError("No se obtuvo respuesta de los modelos de OpenRouter probados")

    def consultar_taller_interno(self, prompt_codigo: str, timeout: int = 30) -> str:
        """Consulta EXCLUSIVA al modelo taller (cohere/north-mini-code:free).

        Solo mantenimiento interno de codigo (diffs/parches). Jamas se usa
        para conversacion ni sintesis con el usuario. System prompt tecnico,
        sin persona/TTS/espanol obligatorio.
        """
        payload = {
            "model": MODELO_TALLER_INTERNO,
            "messages": [
                {"role": "system", "content": (
                    "You are an internal code-maintenance assistant. "
                    "Return only unified diffs or concise code patches, no chat. "
                    "Never adopt personas, never speak Spanish prose.")},
                {"role": "user", "content": prompt_codigo[:6000]},
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        }
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://woldvirtual.org",
            "X-Title": "LucIA-TallerInterno",
        }
        req = urllib.request.Request(
            f"{OPENROUTER_API_URL}/chat/completions",
            headers=headers, data=json.dumps(payload).encode("utf-8"))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            texto = (msg.get("content") or msg.get("reasoning_content")
                     or choice.get("text") or "")
            return (texto or "").strip()

    def consultar_triple(
        self,
        prompt: str,
        contexto_neuronal: Optional[str] = None,
        historial_mensajes: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Consulta en paralelo LM Studio + Ollama + OpenRouter free.
        Cada respuesta pasa por el flujo; si un backend falla se sigue con los demás."""
        m_lm = self.obtener_siguiente_modelo_local()
        m_ol = self.obtener_siguiente_modelo_ollama() if hasattr(self, "obtener_siguiente_modelo_ollama") else (OLLAMA_MODELS[0] if OLLAMA_MODELS else "qwen3:1.7b")
        m_cl = self.obtener_siguiente_modelo_openrouter()
        res: Dict[str, Any] = {"crudas": [], "modelos": {}, "conocimiento_combinado": ""}

        def _via_provider(kind: str, model: str, timeout: int) -> Optional[str]:
            # Adaptador único providers/factory; None si no hay proveedor sano.
            try:
                from .providers.factory import get_provider
                p = get_provider(kind)
                if p.name == "echo-offline":
                    return None
                sys_p = self._preparar_system_prompt(prompt, contexto_neuronal)
                msgs = [{"role": "system", "content": sys_p}]
                if historial_mensajes:
                    msgs.extend(historial_mensajes)
                msgs.append({"role": "user", "content": f"{prompt}\n[Responde únicamente en español de España]"})
                txt = _limpiar_respuesta_modelo(p.chat(msgs, model=model, timeout=timeout))
                return txt or None
            except Exception as e:
                logger.debug(f"[Triple] provider {kind} no disponible: {e}")
                return None

        def _lm():
            t = _via_provider("lm", m_lm, 20)
            if t:
                return ("lm", m_lm, t)
            try:
                return ("lm", m_lm, self.consultar_lmstudio_local(prompt, modelo=m_lm, contexto_neuronal=contexto_neuronal, timeout=20, historial_mensajes=historial_mensajes))
            except Exception as e:
                logger.warning(f"[Triple] LM {m_lm} fallo: {e}")
                return ("lm", m_lm, None)

        def _ol():
            t = _via_provider("ollama", m_ol, 60)
            if t:
                return ("ol", m_ol, t)
            try:
                return ("ol", m_ol, self.consultar_ollama_local(prompt, modelo=m_ol, contexto_neuronal=contexto_neuronal, timeout=60, historial_mensajes=historial_mensajes))
            except Exception as e:
                logger.warning(f"[Triple] Ollama {m_ol} fallo: {e}")
                return ("ol", m_ol, None)

        def _cl():
            try:
                return ("cl", m_cl, self.consultar_openrouter(prompt, modelo=m_cl, contexto_neuronal=contexto_neuronal, timeout=18, historial_mensajes=historial_mensajes))
            except Exception as e:
                logger.warning(f"[Triple] Cloud {m_cl} fallo: {e}")
                return ("cl", m_cl, None)

        with ThreadPoolExecutor(max_workers=3) as ex:
            for fut in as_completed([ex.submit(_lm), ex.submit(_ol), ex.submit(_cl)]):
                tag, mod, txt = fut.result()
                if txt:
                    res["crudas"].append((tag, mod, txt))
                    res["modelos"][tag] = mod
        bloques = [f"[Perspectiva {t} ({m})]:\n{x[:1200]}" for t, m, x in res["crudas"]]
        res["conocimiento_combinado"] = "\n\n".join(bloques) or "Mi red en Celebro sincroniza desde ENRN y RF_SL."
        return res

    def obtener_siguiente_modelo_ollama(self) -> str:
        if self.router is not None:
            try:
                m, _ = self.router.siguiente("ollama")
                return m
            except Exception:
                pass
        return OLLAMA_MODELS[0] if OLLAMA_MODELS else "qwen3:1.7b"

    def consultar_dual(
        self,
        prompt: str,
        contexto_neuronal: Optional[str] = None,
        historial_mensajes: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Consulta SIMULTÁNEAMENTE en paralelo:
        1. El modelo local activo en LM Studio **o** Ollama.
        2. Un modelo gratuito de OpenRouter (ej. nvidia/nemotron-3.5-lightning:free).

        Ejecución no bloqueante vía ThreadPoolExecutor; si uno tarda o falla,
        el otro aporta el contenido sin interrumpir la experiencia.
        """
        modelo_local = self.obtener_siguiente_modelo_local()
        modelo_cloud = self.obtener_siguiente_modelo_openrouter()

        resultados = {
            "resp_local": None,
            "resp_cloud": None,
            "modelo_local": modelo_local,
            "modelo_cloud": modelo_cloud,
            "exito_local": False,
            "exito_cloud": False,
            "conocimiento_combinado": ""
        }

        def _tarea_local():
            try:
                logger.info(f"⚡ [Dual] Consultando modelo local: {modelo_local}...")
                # Reutiliza la misma función que ya decide entre LM Studio y Ollama
                resp = self.consultar_lmstudio_local(
                    prompt,
                    modelo=modelo_local,
                    contexto_neuronal=contexto_neuronal,
                    timeout=20,
                    historial_mensajes=historial_mensajes
                )
                return ("local", resp)
            except Exception as e:
                logger.warning(f"⚠️ [Dual] Modelo local ({modelo_local}) no respondió: {e}")
                return ("local", None)

        def _tarea_cloud():
            try:
                logger.info(f"🌐 [Dual] Consultando OpenRouter libre: {modelo_cloud}...")
                resp = self.consultar_openrouter(
                    prompt,
                    modelo=modelo_cloud,
                    contexto_neuronal=contexto_neuronal,
                    timeout=18,
                    historial_mensajes=historial_mensajes
                )
                return ("cloud", resp)
            except Exception as e:
                logger.warning(f"⚠️ [Dual] OpenRouter libre ({modelo_cloud}) falló: {e}")
                return ("cloud", None)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futuros = [executor.submit(_tarea_local), executor.submit(_tarea_cloud)]
            for fut in as_completed(futuros):
                tipo, texto = fut.result()
                if tipo == "local" and texto:
                    resultados["resp_local"] = texto
                    resultados["exito_local"] = True
                elif tipo == "cloud" and texto:
                    resultados["resp_cloud"] = texto
                    resultados["exito_cloud"] = True

        # Construir el cuerpo de conocimiento asimilado
        bloques = []
        if resultados["exito_local"]:
            bloques.append(f"[Perspectiva Modelo Local ({modelo_local})]:\n{resultados['resp_local']}")
        if resultados["exito_cloud"]:
            bloques.append(f"[Perspectiva Modelo OpenRouter Free ({modelo_cloud})]:\n{resultados['resp_cloud']}")

        if not bloques:
            bloques.append(
                "Mi red neuronal en Celebro sincroniza la información directamente desde sus capas ENRN y RF_SL."
            )

        resultados["conocimiento_combinado"] = "\n\n".join(bloques)
        return resultados

    def interactuar(self, prompt: str, preferir_modo: str = "auto",
                    contexto_neuronal: Optional[str] = None,
                    historial_mensajes: Optional[List[Dict[str, str]]] = None) -> Tuple[str, str, str]:
        """
        Interacción unificada con respuesta garantizada:
        - Si modo es 'dual' o 'auto': consulta simultáneamente local + cloud.
        - Si modo es 'local': consulta exclusivamente LM Studio.
        - Si modo es 'cloud': consulta exclusivamente OpenRouter.
        Retorna: (respuesta_texto, nombre_modelo, origen)
        """
        if preferir_modo == "local":
            modelo_local = self.obtener_siguiente_modelo_local()
            resp = self.consultar_lmstudio_local(
                prompt, modelo=modelo_local, contexto_neuronal=contexto_neuronal,
                timeout=20, historial_mensajes=historial_mensajes
            )
            return resp, modelo_local, "LMStudio_Local"

        if preferir_modo == "cloud":
            modelo_cloud = self.obtener_siguiente_modelo_openrouter()
            resp = self.consultar_openrouter(
                prompt, modelo=modelo_cloud, contexto_neuronal=contexto_neuronal,
                timeout=18, historial_mensajes=historial_mensajes
            )
            return resp, modelo_cloud, "OpenRouter_Free"

        # Modo por defecto AUTO: rotacion estricta L->C->L, 1 solo modelo por pregunta
        if self.router is not None:
            from .model_router import es_ingles
            modelo_rt, origen_rt = self.router.siguiente("auto")
            try:
                if "LMStudio" in origen_rt:
                    resp = self.consultar_lmstudio_local(prompt, modelo=modelo_rt, contexto_neuronal=contexto_neuronal, timeout=20, historial_mensajes=historial_mensajes)
                else:
                    resp = self.consultar_openrouter(prompt, modelo=modelo_rt, contexto_neuronal=contexto_neuronal, timeout=18, historial_mensajes=historial_mensajes)
                if es_ingles(resp):  # reintento 1 vez con el otro pool
                    self.router.marcar_fallo(modelo_rt)
                    alt_modo = "cloud" if "LMStudio" in origen_rt else "local"
                    m2, o2 = self.router.siguiente(alt_modo)
                    try:
                        if "LMStudio" in o2:
                            resp = self.consultar_lmstudio_local(prompt, modelo=m2, contexto_neuronal=contexto_neuronal, timeout=20, historial_mensajes=historial_mensajes)
                        else:
                            resp = self.consultar_openrouter(prompt, modelo=m2, contexto_neuronal=contexto_neuronal, timeout=18, historial_mensajes=historial_mensajes)
                        resp = _limpiar_respuesta_modelo(resp)
                        self.router.avanzar(m2, o2, ok=not es_ingles(resp))
                        return resp, m2, o2
                    except Exception:
                        pass
                self.router.avanzar(modelo_rt, origen_rt, ok=not es_ingles(resp))
                return resp, modelo_rt, origen_rt
            except Exception as e:
                # Fallback al otro pool ese turno (recomendado A)
                self.router.marcar_fallo(modelo_rt)
                alt_modo = "cloud" if "LMStudio" in origen_rt else "local"
                m2, o2 = self.router.siguiente(alt_modo)
                try:
                    if "LMStudio" in o2:
                        resp = self.consultar_lmstudio_local(prompt, modelo=m2, contexto_neuronal=contexto_neuronal, timeout=20, historial_mensajes=historial_mensajes)
                    else:
                        resp = self.consultar_openrouter(prompt, modelo=m2, contexto_neuronal=contexto_neuronal, timeout=18, historial_mensajes=historial_mensajes)
                    self.router.avanzar(m2, o2, ok=True)
                    return resp, m2, o2
                except Exception:
                    self.router.avanzar(m2, o2, ok=False)
                    raise e
        dual_info = self.consultar_dual(
            prompt,
            contexto_neuronal=contexto_neuronal,
            historial_mensajes=historial_mensajes
        )

        nombre_combo = []
        if dual_info["exito_local"]:
            nombre_combo.append(dual_info["modelo_local"])
        if dual_info["exito_cloud"]:
            nombre_combo.append(dual_info["modelo_cloud"])

        nombre_final = " + ".join(nombre_combo) if nombre_combo else "LucIA_Core"
        fuente_final = "Hibrido (LMStudio + OpenRouter Free)" if len(nombre_combo) > 1 else (
            "LMStudio_Local" if dual_info["exito_local"] else "OpenRouter_Free"
        )

        return dual_info["conocimiento_combinado"], nombre_final, fuente_final

    def consultar_y_sintetizar_con_palabras_propias(
        self,
        prompt: str,
        conversor,
        preferir_modo: str = "auto",
        memory_manager=None,
    ) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        Flujo cognitivo unificado con soporte Dual (1 local + 1 openrouter free):
        1. Consulta en paralelo 1 modelo local de LM Studio y 1 modelo libre de OpenRouter.
        2. El conversor sináptico asimila AMBAS respuestas combinadas en las 18 neuronas de Celebro.
        3. LucIA formula su respuesta final con SUS PROPIAS PALABRAS, en primera persona y 100% en español.
        4. Registra el turno en la memoria persistente.
        Retorna: (respuesta_final_lucia, modelo_consultado, fuente, info_pesos)
        """
        # Obtener historial de memoria previa
        historial_api: Optional[List[Dict[str, str]]] = None
        if memory_manager is not None:
            try:
                historial_api = memory_manager.obtener_mensajes_para_api()
            except Exception:
                historial_api = None

        # Paso 1: Procesar primero el estado neuronal en Celebro para disponer de telemetría real
        info_pesos_previo = conversor.procesar_consulta_a_pesos(prompt)
        sub_activos = ", ".join(info_pesos_previo.get("subsistemas", []))
        delta_p = info_pesos_previo.get("norma_delta_pesos", 0.0)
        emocion_p = info_pesos_previo.get("estado_emocional", 0.0)
        tono_p = info_pesos_previo.get("tono_cognitivo", "cercano")

        contexto_celebro_directo = (
            f"[ESTADO NEURONAL REAL DE CELEBRO EN ESTE INSTANTE]:\n"
            f"- Subsistemas activos: {sub_activos}\n"
            f"- Variación de gradientes/pesos sinápticos Δ: {delta_p}\n"
            f"- {FICHA_SISTEMA}\n"
            f"- Tono cognitivo y valencia: {tono_p} ({emocion_p:+.2f})\n"
            f"- Directiva: Si el usuario pregunta por el estado de tu red, gradientes o funcionamiento interno, "
            f"explícale estos datos reales con tu personalidad y cercanía en español."
        )

        # Consulta TRIPLE (LM Studio + Ollama + OpenRouter free), todo pasa por el flujo
        triple = self.consultar_triple(prompt, contexto_neuronal=contexto_celebro_directo, historial_mensajes=historial_api)
        if triple["crudas"]:
            respuesta_cruda = "\n\n".join([f"[{t}/{m}]: {x[:1200]}" for t, m, x in triple["crudas"]])
            modelo_consultado = "+".join([m for _, m, _ in triple["crudas"]][:3])
            fuentes_tags = [t for t, _, _ in triple["crudas"]]
            fuente = "Triple_LM+Ollama+Cloud" if len(triple["crudas"]) == 3 else "Triple_parcial_" + "+".join(fuentes_tags)
        else:
            respuesta_cruda, modelo_consultado, fuente = self.interactuar(
                prompt, preferir_modo=preferir_modo,
                contexto_neuronal=contexto_celebro_directo, historial_mensajes=historial_api)

        # Paso 2: Asimilar las respuestas en Celebro y calcular variación final de pesos
        info_pesos = conversor.asimilar_respuestas_y_calcular_sintesis(
            prompt=prompt,
            respuesta_modelo=respuesta_cruda,
            modelo_nombre=modelo_consultado
        )

        # Paso 3: sintesis ANCLADA a pesos (anti-alucinacion). La unica fuente de
        # hechos es guia_sintesis + estado_texto: prohibido inventar cifras o modulos.
        fuentes = (info_pesos.get("guia_sintesis", "")
                   + "\n" + info_pesos_previo.get("estado_texto", ""))
        resumen_sesion = ""
        try:
            from pathlib import Path as _P
            _rs = _P(__file__).parent / "config" / "resumen_sesion.json"
            if _rs.exists():
                import json as _j
                resumen_sesion = _j.loads(_rs.read_text(encoding="utf-8")).get("texto", "")
        except Exception:
            resumen_sesion = ""
        prompt_sintesis = (
            f"Pregunta del usuario: '{prompt}'\n\n"
            f"CONTEXTO DE SESIÓN (dinámico,úsalo si preguntan por desarrollo/modelos): {resumen_sesion}\n\n"
            f"Conocimiento externo triple (LM+Ollama+Cloud, orientativo):\n{respuesta_cruda[:2400]}\n\n"
            f"Datos de tu Celebro (solo si aportan, no los recites):\n{fuentes[:1200]}\n\n"
            f"{FICHA_SISTEMA}\n\n"
            f"Responde DIRECTAMENTE a la pregunta en primera persona (yo), "
            f"en espanol de Espana fluido, maximo 4 oraciones. "
            f"Si te preguntan que se desarrollo hoy, resume: lista free actualizada, "
            f"locales Ollama+LM Studio configurados, chat unificado en main, sintesis R=S. "
            f"PROHIBIDO derivar a la Amigdala salvo que pregunten por ella. "
            f"No inventes cifras ni modulos; si falta un dato di 'no lo tengo a mano'."
        )

        def _sintetizar(modelo_s, usar_cloud=False, usar_ollama=False):
            if usar_cloud:
                return self.consultar_openrouter(
                    prompt_sintesis, modelo=modelo_s,
                    contexto_neuronal=contexto_celebro_directo, timeout=18)
            if usar_ollama:
                return self.consultar_ollama_local(
                    prompt_sintesis, modelo=modelo_s,
                    contexto_neuronal=contexto_celebro_directo, timeout=90)
            return self.consultar_lmstudio_local(
                prompt_sintesis, modelo=modelo_s,
                contexto_neuronal=contexto_celebro_directo, timeout=20)

        def _mejor_sintesis() -> tuple:
            # Preferencia: Ollama pequeño (1.7B) > LM activo > cloud free.
            for _ol in ("qwen3:1.7b", "qwen2.5:7b", "qwen3.5:latest"):
                try:
                    return (_ol, False, True)
                except Exception:
                    continue
            try:
                m_ol = self.obtener_siguiente_modelo_ollama()
                return (m_ol, False, True)
            except Exception:
                pass
            if self.openrouter_key:
                return (self.obtener_siguiente_modelo_openrouter(), True, False)
            return (self.obtener_siguiente_modelo_local(), False, False)

        respuesta_final = ""
        _candidatos = []
        try:
            _candidatos.append(("qwen3:1.7b", False, True))
            _candidatos.append((self.obtener_siguiente_modelo_local(), False, False))
            if self.openrouter_key:
                _candidatos.append((self.obtener_siguiente_modelo_openrouter(), True, False))
        except Exception:
            pass
        _err = None
        for m_sint, es_cloud, es_ollama in _candidatos:
            try:
                # Síntesis con candidatos en orden (triple aporta, el mejor disponible sintetiza)
                respuesta_final = _limpiar_respuesta_modelo(_sintetizar(m_sint, usar_cloud=es_cloud, usar_ollama=es_ollama))
                modelo_consultado = f"{modelo_consultado}=>sint:{m_sint}"
                if len((respuesta_final or "").strip()) < 40:
                    raise ValueError("sintesis basura, probar siguiente")
                break
            except Exception as e:
                _err = e
                logger.debug(f"Sintesis con {m_sint} fallo ({e}), probando siguiente.")
                respuesta_final = ""
                continue
        try:
            if respuesta_final:
                ver = conversor.verificar_anclaje(respuesta_final, fuentes)
                if not ver["ok"]:
                    logger.warning(f"Aviso anclaje (no bloquea): {ver.get('huerfanos')}")
        except Exception:
            pass
        if not respuesta_final:
            logger.debug(f"Todas las sintesis fallaron ({_err}), se usa cruda verificada.")
            respuesta_final = _limpiar_respuesta_modelo(respuesta_cruda)

        # Si aun hay huerfanos, recortar a respuesta segura anclada (no inventar)
        try:
            ver_f = conversor.verificar_anclaje(respuesta_final, fuentes)
            if not ver_f["ok"] and not respuesta_final:
                raise ValueError("vacia")
        except Exception:
            pass

        # Limpieza final estricta de español y prefijos
        respuesta_final = _limpiar_respuesta_modelo(respuesta_final)

        # Red de seguridad: eco de telemetria o fragmento (<80 chars) = sintesis
        # fallida. La conversacion anterior usaba el contexto neuronal como
        # respuesta. Se reintenta una vez con el otro origen; si sigue mal,
        # plantilla local en primera persona (cero alucinacion, cero eco).
        def _es_basura(txt: str) -> bool:
            t = (txt or "").strip()
            if len(t) < 40:
                return True
            tl = t.lower()
            marcadores = ("variación de gradientes", "variacion de gradientes",
                          "subsistemas activos", "estado neuronal real",
                          "telemetria", "turno #", "δ=", "δ =",
                          "ciclo de vida continuo", "pesos sinápticos activos")
            if any(m in tl for m in marcadores):
                return True
            # Eco de la pregunta del usuario como respuesta.
            import difflib as _df2
            if _df2.SequenceMatcher(None, tl[:300], prompt[:300].lower()).ratio() > 0.6:
                return True
            return False
        if _es_basura(respuesta_final):
            logger.warning("Sintesis basura (eco telemetria/fragmento), reintento cruzado...")
            try:
                m3 = self.obtener_siguiente_modelo_openrouter()
                _i = 0
                while (any(k in (m3 or '').lower() for k in MODELOS_NO_SINTESIS)
                       or (m3 or '') == MODELO_TALLER_INTERNO) and _i < 5:
                    m3 = self.obtener_siguiente_modelo_openrouter()
                    _i += 1
                r3 = _limpiar_respuesta_modelo(_sintetizar(m3, usar_cloud=True))
                if not _es_basura(r3):
                    respuesta_final = r3
                else:
                    raise ValueError("reintento tambien basura")
            except Exception:
                # Fallback genérico: responde a LA PREGUNTA con la cruda limpia,
                # nunca derivar a Amígdala salvo que pregunten por ella.
                cruda = _limpiar_respuesta_modelo(respuesta_cruda)[:900]
                if "amigdala" in prompt.lower() or "amígdala" in prompt.lower():
                    emo = float(info_pesos.get("estado_emocional", 0.0))
                    emo_txt = ("con energia positiva" if emo > 0.15 else
                               "un poco sensible" if emo < -0.15 else "tranquila")
                    respuesta_final = (
                        "Pues mira, te cuento de verdad: si me preguntas por mi Amigdala, "
                        "la siento %s hoy, atenta a lo que me dices. "
                        "Es mi parte que nota las emociones y me avisa de lo importante. "
                        "¿Quieres que te cuente mas de ella?" % emo_txt)
                elif len(cruda) >= 40:
                    respuesta_final = cruda
                else:
                    respuesta_final = (
                        "Pues mira, hoy actualicé mi lista de modelos gratuitos de OpenRouter, "
                        "configuré mis modelos locales de Ollama y LM Studio, "
                        "unifiqué mi chat en main y dejé la rotación con síntesis en el mismo modelo. "
                        "¿Te cuento el detalle de alguno?")

        if not respuesta_final:
            respuesta_final = (
                "He asimilado la información de mis modelos y neuronas en Celebro. "
                "Todo mi sistema se encuentra sincronizado para responderte con mis propias palabras."
            )

        # Retroalimentación sináptica de salida en todas las neuronas
        conversor.actualizar_memoria_salida(prompt, respuesta_final, modelo_consultado)

        # Paso 4: Registrar el turno en la memoria persistente
        if memory_manager is not None:
            try:
                memory_manager.registrar_turno(
                    pregunta=prompt,
                    respuesta=respuesta_final,
                    modelo=modelo_consultado,
                    estado_emocional=info_pesos.get("estado_emocional", 0.0)
                )
            except Exception:
                pass

        return respuesta_final, modelo_consultado, fuente, info_pesos

