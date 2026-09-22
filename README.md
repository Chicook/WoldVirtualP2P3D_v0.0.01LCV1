# WoldVirtualP2P3D - Sistema Cognitivo LucIA (v0.0.01LCV1 - 2026)

Plataforma distribuida de computación cognitiva neuronal, consenso blockchain inmutable y persistencia descentralizada IPFS, con inferencia híbrida (modelos locales vía Ollama/LM Studio y catálogo gratuito de OpenRouter), motor de voz neuronal y diseño de terminal interactiva cyber-bioluminiscente.

---

## 📌 Resumen de lo Realizado Hasta el Momento

El proyecto ha evolucionado desde una estructura heredada hacia una arquitectura modular, unificada y optimizada bajo el paquete central **`LC`** (*LucIA Cognitive*):

### 1. 🧠 Red Neuronal Distribuida (50 Neuronas Activas)
Se estructuraron e integraron 50 neuronas activas organizadas en 5 subsistemas especializados (`LC/celebro/red_neuronal/`):
- **ENRN (10 neuronas)**: *Entrada Recurrente No Lineal*. Manejo de percepción sensorial, codificación y transformaciones no lineales primarias (`EN1_RN.py` a `EN10_RN.py`).
- **RF_SL (10 neuronas)**: *Aprendizaje Supervisado y Residual*. Procesamiento y regularización supervisada (`RFSL1_1.py` a `RFSL1_10.py`).
- **RF_EN (10 neuronas)**: *Retroalimentación y Refuerzo Dinámico*. Modulación adaptativa y aprendizaje por refuerzo continuo (`RFEN1_RN_1.py` a `RFEN1_RN_10.py`).
- **RNP (10 neuronas)**: *Plasticidad Sináptica y Memoria Asociativa*. Métodos adaptativos de gradientes, compuertas de atención y retención (`RN11.py` a `RN14_PuertasAtencion.py`).
- **SLRN (10 neuronas)**: *Optimizadores y Convergencia Sináptica Temporal*. Algoritmos de convergencia y optimización adaptativa (`SL11.py` a `SL12.py`).

### 2. ⛓️ Blockchain Neuronal y Consenso PoNL (`BKSVCB.py`)
- Servidor blockchain inmutable con consenso **PoNL** (*Proof of Neural Learning*).
- Estructura criptográfica con hashing doble SHA-256, cálculo de árbol de transacciones (**Merkle Root**) y dificultad de minado ajustable.
- **Transducción Ledger ➔ Pesos**: Al arrancar, transforma el historial de transacciones y bloques del `blockchain_ledger.json` en tensores sinápticos activos en memoria.
- **Persistencia en cada bloque**: Registro de transacciones neuronales (prompts, respuestas, deltas de pesos y firmas del modelo).
- **Servidor HTTP JSON-RPC / REST** embebido (puerto por defecto `8545`) para consulta de bloques, minado y estado del consenso.

### 3. 🌐 Persistencia Descentralizada IPFS (`CMFG/ipfs_manager.py`)
- Gestor de conexión al nodo IPFS Kubo RPC (`http://127.0.0.1:5001`).
- Mapeo y registro de CIDs en `ipfs_manifest.json`.
- Checkpoints atómicos: al finalizar sesión o minar bloques, se realiza el pinning automático del estado neuronal y el ledger blockchain a IPFS con recuperación de hash de contenido (CID).

### 4. ⚡ Transducción y Dinámica de Pesos Vivos (`CMFG/PSNRCV.py` y `pesos_vivos.py`)
- **Motor Matemático de Alta Precisión**:
  - Ortogonalización polar **Muon Newton-Schulz 5 (NS-5)**.
  - Estimación espectral de gradiente **GSNR** (*Gradient Signal-to-Noise Ratio*).
  - Cálculo de **Entropía de Shannon** para medir diversidad sináptica y dispersión de información.
  - Detección autónoma de deriva, saturación de gradientes y corrección dinámica de tensores.
- Persistencia local en tiempo real dentro del directorio `LC/celebro/PSNRL/`.

### 5. 🤖 SubSistema de Inferencia Gratuita y Modelos Locales (`IAFREE.py` e `IAlocal.json`)
- **IAFREE**: Conector inteligente a la API de OpenRouter enfocado exclusivamente en modelos sin costo (`:free`), como `google/gemma-4-31b-it:free`, `nvidia/nemotron-3-super-120b-a12b:free`, `qwen/qwen3.8-27b:free`, entre otros.
  - Rotación y failover automático ante códigos HTTP 429 (*Rate Limit*) o 404.
  - Inyección de contexto cognitivo de las 50 neuronas activas en los prompts.
  - Registro de consumo con balance garantizado en `$0.00 USD`.
- **Modelos Locales**: Configuración en `LC/modelosIAlocal/IAlocal.json` para integración sin conexión con servidores Ollama / LM Studio locales.

### 6. 🎨 Motor de Estilos Visuales y Terminal Cyber-Bioluminiscente (`STYLOS.py`)
- Sistema de diseño de consola estilo *Cyber-Bioluminiscente* (Electric Cyan `#00F5FF`, Violeta Neón `#9D00FF`, Ámbar Solar `#FFAA00`, Esmeralda Vivo `#00E676`).
- Renderizado de cajas y tarjetas con bordes redondeados Unicode (`╭─╮╰─╯│`).
- Formateo enriquecido de Markdown con resaltado de sintaxis, listas y tablas.
- Banners de telemetría de turno en vivo (duración en ms, modelo activo, deriva Muon, CID de IPFS y transacciones pendientes).

### 7. 🗣️ Motor de Síntesis de Voz Neuronal (`voice_engine.py`)
- Voz neuronal oficial `es-ES-ElviraNeural` (vía Microsoft Edge/Speech Services o fallback por `System.Speech`).
- Modulador prosódico y de expresividad juvenil espontánea.
- Sanitización de texto para locución limpia (filtrado de markdown, enlaces y código).
- Cancelación y parada instantánea mediante `winmm.dll (MCI)` sin bloquear la interfaz.

### 8. 🎛️ Orquestador Central Unificado (`mainLCSTM.py`)
- Punto de entrada maestro del sistema cognitivo.
- Carga segura de variables de entorno desde `.env`.
- Inicialización coordinada de los subsistemas:
  1. *Blockchain BKSVCB*
  2. *Conversor PSNRCV (50 neuronas)*
  3. *Transducción Ledger ➔ Pesos*
  4. *Sesión y subsistema IAFREE*
- Bucle interactivo por turnos con comandos integrados (`estado`, `modelos`, `minar`, `modelo <id>`, `ayuda`, `salir`).
- Protocolo de cierre ordenado con minado de transacciones pendientes y sincronización final a IPFS.

---

## 📁 Estructura del Directorio

```text
WoldVirtualP2P3D_v0.0.01LCV1/
├── .env                       # Credenciales locales (ignorado por git)
├── .gitignore                  # Reglas de exclusión de git
├── README.md                  # Descripción del proyecto y avances
└── LC/
    ├── mainLCSTM.py           # Orquestador maestro del sistema LucIA
    ├── modelosIAlocal/
    │   └── IAlocal.json       # Configuración de endpoints locales (Ollama/LM Studio)
    └── celebro/
        ├── __init__.py        # Exportaciones del núcleo cognitivo y catálogo neuronal
        ├── BKSVCB.py          # Servidor blockchain, bloques y consenso PoNL
        ├── blockchain_ledger.json # Ledger histórico de bloques y transacciones
        ├── PSNRL/             # Checkpoints locales de pesos sinápticos activos
        ├── CMFG/              # Celebro Model & Feed Gateway
        │   ├── ipfs_manager.py    # Conexión Kubo RPC y subida a IPFS
        │   ├── ipfs_manifest.json # Manifiesto de CIDs y hashes SHA-256
        │   ├── PSNRCV.py          # Conversor de diálogos a pesos y motor matemático
        │   ├── pesos_vivos.py     # Monitor de telemetría sináptica en terminal
        │   └── SBSTM/             # SubSistema de Sesión, Voz, Estilos y LLM Free
        │       ├── IAFREE.py          # Inferencia $0.00 vía OpenRouter :free
        │       ├── RPLC.py            # Filtro y procesador de voz/tono propio
        │       ├── SNSBSTNPRB.py      # Gestor de sesión P2P
        │       ├── STYLOS.py          # Motor visual de terminal ANSI / Unicode
        │       └── voice_engine.py    # Síntesis TTS juvenil con interrupción MCI
        └── red_neuronal/      # Paquete con las 50 neuronas activas
            ├── ENRN/          # 10 neuronas de entrada recurrente
            ├── RF_SL/         # 10 neuronas de aprendizaje supervisado
            ├── RF_EN/         # 10 neuronas de refuerzo adaptativo
            ├── RNP/           # 10 neuronas de plasticidad y optimizadores
            └── SLRN/          # 10 neuronas de convergencia temporal
```

---

## 🚀 Requisitos e Inicio Rápido

### Requisitos Previos
- **Python 3.10+** (recomendado Python 3.11 o 3.12).
- Dependencias estándar y científicas: `numpy`, `edge-tts` (opcional para voz neuronal de alta calidad).
- Nodo **IPFS Kubo** local corriendo en `http://127.0.0.1:5001` (opcional, opera con fallback si no está disponible).
- Clave de API de OpenRouter configurada en `.env` (si se usa `IAFREE`):
  ```env
  OPENROUTER_API_KEY=tu_api_key_aqui
  ```

### Ejecución
Para iniciar el orquestador maestro interactivo:
```bash
python LC/mainLCSTM.py
```

### Comandos en la Consola de LucIA
- `estado` / `status`: Muestra la tarjeta integral de salud (bloques, neuronas activas, deriva Muon, costo $0.00).
- `modelos` / `free`: Lista los modelos gratuitos disponibles en el catálogo de OpenRouter.
- `modelo <nombre_o_id>`: Selecciona un modelo específico para inferencia.
- `minar` / `mine`: Fuerza el minado inmediato de transacciones sinápticas pendientes en un nuevo bloque.
- `ayuda` / `help`: Muestra el panel interactivo de ayuda de comandos.
- `salir` / `exit`: Consolida el estado, mina el bloque final y sincroniza con IPFS.
