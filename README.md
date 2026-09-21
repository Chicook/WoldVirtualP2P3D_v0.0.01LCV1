# WoldVirtualP2P3D - Sistema LucIA (Versión Ligera < 125 MB & Total Libertad)

Bienvenido a la documentación oficial del sistema **LucIA** integrado en `WoldVirtualP2P3D_v0.0.01LCV1`. Este proyecto implementa una arquitectura reducida, ultraligera y autónoma de inteligencia artificial conversacional, con red neuronal propia, servidor blockchain de neuronas, descarga y aprovisionamiento autónomo de modelos locales por sesión, síntesis de voz neuronal en español de España (`es-ES-ElviraNeural`), integración de OpenRouter exclusivamente con modelos gratuitos, y persistencia descentralizada de pesos en IPFS con control estricto de caché.

---

## 📌 Estado Actual del Sistema

- **Huella en disco**: **~5.8 MB** (Límite máximo estricto: **125.0 MB** - Cumplido ✅).
- **Entorno**: Python 3.14 (Windows) con dependencias optimizadas y sin bibliotecas pesadas innecesarias.
- **Personalidad y Libertad**: LucIA se expresa con **total libertad**, autenticidad y elocuencia como una chica de 18 años de España, sin censuras mojigatas ni filtros moralizantes innecesarios.
- **Gestión Autónoma de Modelos Locales**: En cada sesión se verifica y descarga/actualiza de forma automática **al menos 1 modelo de IA local operativo** mediante **Ollama** (`ollama pull`) o **LM Studio** (`lms get -y`).
- **OpenRouter 100% Free**: Detección dinámica y priorización estricta de modelos gratuitos con sufijo `:free` (vía API en vivo).
- **Servidor Blockchain de Neuronas (`LC/mainLC.PY`)**: Integración e indexación de bloques para las 485 neuronas activas en `RD_Neuronal`.
- **Voz Neuronal**: `es-ES-ElviraNeural` (Español de España) con síntesis fluida continua.
- **Persistencia**: Cierre de sesión con exportación a IPFS, borrado seguro de pesos locales y vaciado total de `Celebro/cache` (0 archivos residuales).

---

## ✅ Mejoras y Capacidades Integradas

### 1. LucIA con Total Libertad de Expresión
- Expresión directa en primera persona ("yo siento", "yo pienso").
- Eliminación total de coletillas robóticas (*"como modelo de lenguaje..."*), tabúes o sermones prefabricados.
- Respuestas elocuentes, cercanas, sinceras y sin rodeos, manteniendo su identidad única de LucIA.

### 2. Auto-Gestor de Descarga de Modelos Locales (`local_model_manager.py`)
- Supervisa los modelos locales en **Ollama** (`http://127.0.0.1:11434`) y **LM Studio** (`http://127.0.0.1:1234`).
- En cada sesión comprueba qué modelos faltan de la lista curada ultraligera (ej. `qwen2.5:0.5b`, `llama3.2:1b`, `qwen3:1.7b`, etc.) y descarga/pone a punto al menos uno de forma autónoma.

### 3. Conector Híbrido y Filtrado Estricto de OpenRouter Gratuito
- Conexión configurada en `.env` mediante `OPENROUTER_API_KEY`.
- Consulta dinámica al endpoint `/api/v1/models` para verificar en tiempo real los modelos activos con sufijo `:free`.
- Rotación triple con fallback: **LM Studio Local** ↔ **Ollama Local** ↔ **OpenRouter Free Cloud**.

### 4. Blockchain Neuronal y Coexistencia con `LC`
- El servidor `LC/mainLC.PY` registra cada neurona activa con un hash único SHA-256 en bloques encadenados verificables.
- Compatibilidad de rutas relativas tanto ejecutando desde la raíz como desde la carpeta `LC/`.

---

## 📁 Estructura del Directorio

```text
WoldVirtualP2P3D_v0.0.01LCV1/
├── .env                        # Clave OPENROUTER_API_KEY
├── .gitignore                  # Exclusión de venv, caches y pesos temporales
├── README.md                   # Documentación técnica completa
├── iniciar_lucia.bat           # Lanzador rápido de LucIA en terminal Windows
├── LC/
│   ├── mainLC.PY               # Servidor Blockchain de neuronas activas
│   └── celebro/
│       ├── RD_Neuronal/        # 485 neuronas en 5 familias (ENRN, RF_SL, RF_EN, RNP, SLRN)
│       └── cache/              # Directorio de caché
└── lucIA/
    ├── CORE/                   # Clases base, text_encoder, voice_engine, utils
    ├── Celebro/                # 15 módulos anatómicos (Amigdala, Hipocampo, Talamo, etc.)
    ├── config/                 # config.json, especialistas.yaml, router_state.json
    ├── providers/              # Adaptadores OpenAICompat, Ollama, Echo
    ├── llm_connector.py        # Conector híbrido con libertad de LucIA y OpenRouter Free
    ├── local_model_manager.py  # Gestor de descarga de >=1 modelo local por sesión
    ├── model_router.py         # Router inteligente de 3 vías
    ├── session_manager.py      # Control de límite <125MB y limpieza
    ├── ipfs_manager.py         # Persistencia de pesos en IPFS
    ├── main.py                 # Punto de entrada conversacional principal
    └── chat.py                 # Acceso rápido legacy
```

---

## 🚀 Guía de Uso

### Iniciar LucIA
Ejecuta en consola:
```bash
python lucIA/main.py
```
O haz doble clic en `iniciar_lucia.bat`.

### Ejecutar Servidor Blockchain de Neuronas
```bash
python LC/mainLC.PY
```

### Ejecutar Pruebas Automatizadas del Sistema
```bash
python lucIA/test_system.py
```
