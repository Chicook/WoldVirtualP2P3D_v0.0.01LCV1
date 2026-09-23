# WoldVirtualP2P3D - Sistema Cognitivo LucIA (v0.0.01LCV1 - 2026)

Plataforma distribuida de computación cognitiva neuronal, consenso blockchain inmutable y persistencia descentralizada IPFS, con inferencia híbrida (modelos locales vía Ollama/LM Studio y catálogo gratuito de OpenRouter), motor de voz neuronal y diseño de terminal interactiva cyber-bioluminiscente.

---

## Estado actual verificable de LucIA

> **Fuente factual:** `LC/ARCHITECTURE_CHANGELOG.json` y las comprobaciones estructurales ejecutadas por `LC.compatibility`. Esta sección distingue entre código presente, capacidad parcialmente integrada y comportamiento validado de extremo a extremo.

- **Versión de arquitectura:** `2026.3.1-refactor-2`.
- **Cambios registrados:** 7.
- **Capacidades registradas:** 8.
- **Estado operativo de LucIA:** observacional y reportorial. LucIA puede leer el manifiesto, diagnosticar el estado y comunicar limitaciones, pero no modifica por sí misma el código ni la configuración.
- **Último cierre conocido:** el cierre informa correctamente `refactor parcial_omitido` cuando HRCTRC omite módulos vivos. El conteo observado más reciente fue de 5 omisiones.

### Capacidades verificadas

- `imports.compatibility`: resuelve aliases heredados hacia módulos activos sin cargar respaldos de `CHG`.
- `diagnostics.factual-context`: inyecta el manifiesto factual en el contexto de LucIA y evita inventar changelogs.
- `diagnostics.closure-status`: el resumen de cierre distingue entre refactor aplicado, parcial omitido y fallido, e informa el número de omisiones.

### Capacidades parciales

- `runtime.local-ai-config`: `LC/modelosIAlocal/IAlocal.json` existe y define Ollama, LM Studio y modo offline; todavía no está verificado el failover automático completo.
- `blockchain.transaction-safety`: las transacciones pendientes se confirman después de minar y persistir el bloque; no están validados todos los escenarios de reorg, fork y concurrencia.
- `session.ollama-config`: `SNSBSTNPRB` lee y normaliza la configuración de Ollama; faltan pruebas de timeout, TLS, autenticación y extremos de configuración.
- `purger.path-safety`: `logger` y `CHG_DIR` están definidos en las partes del purgador; faltan pruebas de carga, dry-run y auditoría prolongada.

### Capacidad bloqueada

`live-module-refactor` permanece en estado `blocked`. HRCTRC no aplica refactors a módulos vivos sin una prueba previa aprobada. En el último cierre fueron omitidos:

- `LC/celebro/BKSVCB.py`
- `LC/celebro/CMFG/SBSTM/INTEGRACIONRF.py`
- `LC/celebro/CMFG/SBSTM/SNSBSTNPRB.py`
- `LC/celebro/CMFG/ipfs_manager.py`
- `LC/celebro/red_neuronal/RF_EN/RFEN1_RN_4.py`

Estas omisiones son deliberadas: no deben describirse como refactors aplicados hasta superar sus pruebas y el mecanismo de aprobación correspondiente.

### Comprobaciones actuales

- `compatibility_layer`: **OK**.
- `local_ai_config`: **OK** como comprobación de presencia y estructura básica; no equivale a una prueba end-to-end.
- `transaction_safety_fix`: **OK** como comprobación de la implementación presente; no cubre todavía todos los escenarios de consenso.
- `architecture_tests_available`: **OK**; existen pruebas de arquitectura.
- `architecture_tests_verified`: **PENDIENTE**; el arranque de LucIA no ejecuta automáticamente toda la suite.
- `purgador_logger`: **OK**.

No se deben afirmar como implementados o verificados, sin pruebas adicionales, WAL, `fsync`, reorg automático, cron seguro, mitigación CVE, balanceo entre backends, validación completa de JSON Schema ni tests end-to-end.

---

## Pendientes de desarrollo y plan de cierre

El orden siguiente prioriza observabilidad y seguridad antes de desbloquear módulos vivos.

### Fase 0 — Línea base reproducible

1. Ejecutar la suite de arquitectura en el entorno real con todas sus dependencias.
2. Guardar el resultado, duración, versión de Python y dependencias utilizadas.
3. Separar en los informes `tests disponibles`, `tests ejecutados` y `tests aprobados`.
4. Hacer que el cierre de LucIA marque `architecture_tests_verified=OK` únicamente cuando exista un resultado válido de esa ejecución.

**Criterio de cierre:** informe reproducible y verificable; ningún diagnóstico debe afirmar que la suite pasa si no se ha ejecutado.

### Fase 1 — Configuración y failover de IA local

1. Validar `IAlocal.json` contra un esquema versionado.
2. Implementar health-check real para Ollama y LM Studio con timeout explícito.
3. Definir la política de selección: backend primario, fallback, modo offline y recuperación.
4. Registrar en el diagnóstico qué backend está vivo y cuál fue utilizado.
5. Probar reinicio, backend inexistente, timeout, respuesta inválida y recuperación posterior.

**Criterio de cierre:** una prueba end-to-end demuestra la transición Ollama → LM Studio → offline sin intervención manual y sin perder el contexto de sesión.

### Fase 2 — Endurecimiento de la sesión Ollama

1. Validar endpoint, modelo, timeout, TLS, autenticación y `keep_alive`.
2. Rechazar configuraciones incompletas con mensajes accionables.
3. Definir qué parámetros pueden cambiarse en caliente y cuáles requieren reinicio.
4. Añadir pruebas de endpoint caído, modelo inexistente y respuesta lenta.

**Criterio de cierre:** `SNSBSTNPRB` informa una configuración normalizada y falla de forma controlada ante cada error previsto.

### Fase 3 — Seguridad transaccional y persistencia

1. Añadir recuperación del mempool después de reinicio.
2. Diseñar persistencia atómica de transacciones pendientes y bloques.
3. Probar doble gasto, concurrencia, reinicio durante persistencia y bloques huérfanos.
4. Definir y probar la política de reorg/fork antes de declarar esa capacidad verificada.
5. Medir la latencia y el comportamiento ante errores de disco.

**Criterio de cierre:** las pruebas demuestran que no se pierden transacciones pendientes ni se confirman estados que no estén persistidos.

### Fase 4 — Purgador seguro y auditable

1. Añadir modo `dry-run` antes de cualquier purga programada.
2. Validar que cada ruta objetivo permanezca dentro del directorio permitido.
3. Probar directorios vacíos, archivos bloqueados, enlaces, rutas inexistentes y grandes volúmenes.
4. Centralizar el logger y conservar auditoría más allá de la rotación corta.
5. Registrar en el informe qué se eliminó, qué se omitió y por qué.

**Criterio de cierre:** una ejecución de prueba no destructiva y una ejecución controlada producen informes completos y no permiten salir del ámbito autorizado.

### Fase 5 — Desbloqueo progresivo de módulos vivos

Aplicar el mismo ciclo a cada módulo, uno por uno:

1. Capturar el comportamiento actual con pruebas de contrato.
2. Crear una prueba de regresión específica para el cambio.
3. Ejecutar el refactor en una copia o rama aislada.
4. Comparar imports, interfaces, estado persistido y rendimiento.
5. Aprobar el resultado antes de permitir que HRCTRC lo aplique al sistema vivo.
6. Mantener rollback y registrar el resultado en el manifiesto.

Orden recomendado: `INTEGRACIONRF.py`, `SNSBSTNPRB.py`, `BKSVCB.py`, `ipfs_manager.py` y, por último, `RFEN1_RN_4.py`.

**Criterio de cierre:** cada módulo tiene pruebas aprobadas, rollback documentado y aparece como `verified` o `partial` con evidencia; nunca se cambia a `verified` solo porque el proceso terminó sin excepción.

### Fase 6 — Diagnóstico y documentación continua

1. Actualizar `LC/ARCHITECTURE_CHANGELOG.json` en cada cambio real.
2. Mantener el diagnóstico de LucIA alineado con los logs de cierre.
3. No registrar como capacidad verificada una función que solo esté definida en configuración.
4. Conservar la diferencia entre estado estructural, prueba unitaria y prueba end-to-end.

**Criterio de cierre:** el manifiesto, el cierre de sesión, los tests y este README describen el mismo estado.

---

## Diseño y capacidades declaradas del sistema

La sección siguiente describe la arquitectura y las funciones previstas o integradas en el código. Para conocer qué está verificado actualmente, prevalece siempre la sección [Estado actual verificable de LucIA](#estado-actual-verificable-de-lucia) y el manifiesto `LC/ARCHITECTURE_CHANGELOG.json`.

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
        │   ├── ipfs_manager.py    # Conexión Kubo RPC y subida a IPFS (pin confirmado, 500 entradas + spill)
        │   ├── neural_math.py     # Motor matemático único: Muon NS-5, SOAP, GSNR, Shannon, SemanticEncoderSHA
        │   ├── PSNRCV.py          # Conversor de diálogos a pesos (trust-ratio por neurona, importlib)
        │   ├── pesos_vivos.py     # Monitor de telemetría sináptica en terminal
        │   └── SBSTM/             # SubSistema de Sesión, Voz, Estilos y LLM Free
        │       ├── IAFREE.py          # Inferencia $0.00 vía OpenRouter :free (caché disco, backoff exp.)
        │       ├── RPLC.py            # Filtro y procesador de voz/tono propio
        │       ├── SNSBSTNPRB.py      # Gestor de sesión P2P (peers + sincronizar_ledger)
        │       ├── STYLOS.py          # Motor visual de terminal ANSI / Unicode
        │       └── voice_engine.py    # Síntesis TTS juvenil con interrupción MCI
        └── red_neuronal/      # Paquete con las 50 neuronas activas
            ├── common/          # bases.py, init_utils.py, buffers.py (compartidos)
            ├── ENRN/          # 10 neuronas de entrada recurrente
            ├── RF_SL/         # 10 neuronas de aprendizaje supervisado
            ├── RF_EN/         # 10 neuronas de refuerzo (+ _bases/_buffers/_optimizers/_serializers)
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
- `pin`: Sube PSNRL a IPFS (solo borra local con pin confirmado; resto queda `pendiente_pin`).
- `restaurar <CID>`: Recupera pesos desde IPFS (daemon > CLI > vault > manifiesto).
- `benchmark`: Mide latencia (1 llamada corta por modelo, timeout 10 s).
- `exportar-pesos`: Checkpoint manual de las 50 neuronas en PSNRL.
- `ayuda` / `help`: Muestra el panel interactivo de ayuda de comandos.
- `salir` / `exit`: Consolida el estado, mina el bloque final y sincroniza con IPFS.

### P2P mínimo
`SesionNeuronalP2P.agregar_peer(url)` + `sincronizar_ledger()` adoptan la cadena más larga vía `GET /blocks` del puerto 8545 (+reintentos). Variable `LUCIA_BKS_PORT` configura el puerto.

### Tests y CI
```bash
pip install -e .[dev]
python -m pytest LC/celebro/test -q --timeout=60
```
CI en `.github/workflows/ci.yml` (py3.10/11/12, ubuntu+windows).