# WoldVirtualP2P3D - Sistema LucIA (Versión Ligera < 125 MB)

Bienvenido a la documentación oficial del sistema **LucIA** integrado en `WoldVirtualP2P3D_v0.0.01`. Este proyecto implementa una arquitectura reducida, ultraligera y autónoma de inteligencia artificial conversacional, con red neuronal propia, síntesis de voz neuronal en español de España (`es-ES-ElviraNeural`), persistencia descentralizada de pesos en IPFS y control estricto de caché.

---

## 📌 Estado Actual del Sistema

- **Huella en disco**: **~11.65 MB** (Límite máximo estricto: **125.0 MB** - Cumplido ✅).
- **Entorno**: Python 3.14 (Windows) con dependencias optimizadas y sin bibliotecas pesadas innecesarias (`torch` o `scipy`).
- **Voz Neuronal**: `es-ES-ElviraNeural` (Español de España) con reproducción nativa mediante MCI Windows y síntesis `edge_tts`.
- **Conexión Híbrida**: LM Studio Local (`127.0.0.1:1234`) y OpenRouter Cloud Gratuito con rotación por consulta.
- **Persistencia**: Cierre de sesión con exportación a IPFS, borrado seguro de pesos locales y vaciado total de `Celebro/cache` (0 archivos residuales).

---

## ✅ Todo lo Realizado

### 1. Flujo Cognitivo: Modelos Externos asimilados a Pesos y Voz Propia
- **Consulta distribuida**: Cada pregunta se remite a los modelos locales en **LM Studio** (`127.0.0.1:1234`) y/o a modelos gratuitos de **OpenRouter** (`openrouter.ai/free`).
- **Conversión sináptica a pesos en `Celebro`**:
  - La respuesta técnica recibida es vectorizada mediante `TextEncoder` (vector 4D) y proyectada por las neuronas `ENRN` (`EN2_Xavier`, `EN8_ReLU`) y `RF_SL` (`NeuronaMemoriaBase`).
  - Se calculan las variaciones sinápticas en tiempo real ($\Delta\text{pesos}$) y el estado emocional (`[-1.0, 1.0]`).
  - Muestra en consola la telemetría cognitiva: `🧠 [Celebro]: Vector semántico=[...] | Δpesos=... | Emoción=...`.
- **Síntesis con palabras propias de LucIA**:
  - LucIA no reproduce mecánicamente el LLM base: con el estado sináptico asimilado en `Celebro`, formula su respuesta final **con sus propias palabras, tono e identidad de chica de 18 años de España**.
  - La respuesta se pronuncia fluidamente mediante la voz neuronal `es-ES-ElviraNeural`.
- **Retroalimentación continua**: La interacción completa actualiza la neurona de memoria asociativa en `Celebro`.

### 2. Motor de Síntesis de Voz Neuronal (`es-ES-ElviraNeural`)
- **Voz oficial**: Asignada `es-ES-ElviraNeural` femenina en español de España.
- **Síntesis fluida continua**: Longitud de fragmentos configurada en **1500 caracteres** (conforme a `app_config.py` de `inteligenciaartificial_lucIA`), permitiendo que respuestas extensas se descarguen y reproduzcan en una sola pista sin pausas artificiales ni micro-cortes.
- **Limpieza profunda de texto para TTS**: Supresión automática de bloques `<think>`, asteriscos de markdown, viñetas (`* `, `- `, `+ `), tablas y listas numéricas para una dicción natural y suave.
- **Cancelación instantánea (`cancel_speech`)**: Cada vez que el usuario formula una nueva consulta, cualquier locución anterior se interrumpe inmediatamente en milisegundos para evitar solapamientos en el dispositivo de audio MCI.

### 3. Conector Híbrido de Modelos y System Prompt Inflexible en Español
- **Rotación inteligente**:
  - **LM Studio Local**: Conexión a `http://127.0.0.1:1234/v1`, con detección automática de modelos ligeros disponibles.
  - **OpenRouter Cloud Gratuito**: Rotación entre modelos libres (`nvidia/nemotron-3.5-lightning:free`, `google/gemma-4-26b-a4b-it:free`, etc.) según `especialistas.yaml`.
- **System Prompt Mandatorio en Español de España**:
  - Regla estricta de idioma: Prohibición absoluta de emitir texto, preámbulos o pensamientos internos en inglés.
  - Regla de identidad: LucIA no adopta nombres de modelos base y se expresa siempre en primera persona.
- **Filtro Anti-Monólogo**: Eliminación garantizada de trazas de razonamiento en inglés (como *"- User asked:"*, *"Persona:"*, *"Voice/format:"*).

### 4. Conexión Real con IPFS y Gestión de Memoria
- **Persistencia IPFS Real y Fallback Descentralizado**:
  - Detección del daemon local HTTP IPFS en `http://127.0.0.1:5001`.
  - Subida directa vía multipart `/api/v0/add?pin=true` con fijado real (Pinning) en la red descentralizada.
  - Generación estándar de CIDv0 (`Qm...`) y registro completo en `ipfs_manifest.json`.
  - **Borrado local estricto**: Al transferir los pesos a IPFS, el archivo `pesos_activos.npz` se elimina de disco, preservando el tamaño del sistema < 125 MB (actualmente ~5.7 MB).
- **Vaciado total de `Celebro/cache`**: Redirección de todo el bytecode a `Celebro/cache` y vaciado del 100% de los archivos residuales al cerrar sesión.

### 5. Configuración y Eliminación de Advertencias
- **Directorio de configuración `lucIA/config/`**:
  - `config.json`: Define `only_free: true`, voz predeterminada y modelo principal.
  - `especialistas.yaml`: Mapeo de modelos y candidatos libres de OpenRouter.
- **Arranque limpio en `RF_EN`**: Importación dinámica y silenciosa de los submódulos `RFENRN1` a `RFENRN10` en `lucIA/Celebro/RF_EN/__init__.py`, eliminando los `UserWarning` previos causados por la ausencia de `scipy` y `torch`.

---

## 📁 Estructura del Directorio `lucIA/`

```text
lucIA/
├── CORE/
│   ├── base.py                 # Clases base de neuronas y sistema LucIASystem
│   ├── initializers.py         # Inicializadores He, Xavier y LeCun
│   ├── text_encoder.py         # Extractor de vectores semánticos 4D
│   ├── utils.py                # Utilidades de normalización y convergencia
│   └── voice_engine.py         # Motor TTS es-ES-ElviraNeural y control MCI
├── Celebro/
│   ├── ENRN/                   # 10 neuronas de entrada funcionales (Xavier, ReLU, etc.)
│   ├── RF_SL/                  # 10 módulos de memoria supervisada Hebbiana
│   ├── RF_EN/                  # 10 módulos de aprendizaje por refuerzo
│   ├── RNP/                    # Optimizadores de pesos neuronales (AdamW, Lion, SAM)
│   ├── SLRN/                   # Redes de aprendizaje supervisado
│   ├── cache/                  # Directorio centralizado de compilación __pycache__
│   └── conversor_pesos.py      # Conversor cognitivo de consultas y memoria a pesos
├── config/
│   ├── config.json             # Configuración general y voz
│   └── especialistas.yaml      # Modelos gratuitos y clave OpenRouter
├── llm_connector.py            # Conector híbrido LM Studio + OpenRouter con filtro anti-inglés
├── session_manager.py          # Control de límites de espacio (125MB) y purga de caché
├── ipfs_manager.py             # Transferencia a IPFS y borrado de pesos locales
├── main.py                     # Punto de entrada interactivo conversacional
└── __init__.py                 # Exportación de componentes principales
```

---

## 🚀 Guía de Uso Rápido

### Iniciar LucIA en Modo Conversacional
Desde la terminal en el directorio del proyecto:

```bash
python lucIA/main.py
```

### Comandos Disponibles en Chat
- `/modo local` : Forzar uso de modelos locales de LM Studio (`127.0.0.1:1234`).
- `/modo cloud` : Forzar uso de modelos gratuitos de OpenRouter.
- `/modo auto`  : Modo híbrido inteligente (prioriza local con fallback a cloud).
- `/voz on`     : Activa la síntesis de voz con `es-ES-ElviraNeural`.
- `/voz off`    : Desactiva la voz y responde solo por texto en terminal.
- `/estado`     : Muestra el estado sináptico de `Celebro`, neuronas activas y tamaño del sistema.
- `salir` / `exit`: Despedida por voz, subida de pesos a IPFS, borrado local y vaciado de caché.

---

## ⏳ Tareas Pendientes / Roadmap Futuro

1. **Integración con Avatar 3D (Godot & Blender)**:
   - Conectar los sockets TCP en los puertos `9876` (Blender) y `9877` (Godot) para sincronizar expresiones emocionales en tiempo real (`happy`, `worried`, `neutral`) en base al valor neural calculado por `Celebro`.
2. **Sincronización de Visemas/Labios (Lip-sync)**:
   - Mapear las emisiones de fonemas de `edge_tts` hacia los visemas del modelo 3D en Godot para lograr sincronización labial realista.
3. **Red P2P Distribuida (WoldVirtualP2P3D)**:
   - Propagar los CIDs de IPFS generados en cada cierre de sesión a través de la red peer-to-peer con otros nodos del mundo virtual.
4. **Ciclo de Auto-Entrenamiento en Background (LifeLoop)**:
   - Integrar un hilo en segundo plano de baja prioridad que revise periódicamente las interacciones pasadas y ajuste los pesos de los módulos `RF_EN` durante tiempos de inactividad del usuario.
