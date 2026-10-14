# Sesión de desarrollo — Teleporte entre islas 3D + P2P (2026-10-14)

Directorio de trabajo: `VersionDe_sesion/CONTRRF/TLLDCD/src/MTBGT/WoldVirtualP2P3D/`
Base intacta: `ChunkManager.gd` (red P2P), `userbase.gd` (movimiento), `ISLA_FINAL.tscn`,
`ApplyMaterial.gd`. Todo lo nuevo es aditivo.

## 1. Archivos nuevos
- `ESCENAS/sceneOPTM/TeleportManager.gd` (+ `.uid`)
  Historial persistente de islas (visitable aunque el dueño se vaya), detección por
  nodos (válida en clientes, donde `active_islands` está vacío), `teleport_to(index)`,
  `get_islands()`, `get_max_index()`, `current_island_index(pos)`, `mark_owner_left()`.
- `ESCENAS/sceneOPTM/TeleportPanel.gd` (+ `.uid`)
  `CanvasLayer` lateral (`layer=100`): botón `Isla N [T]` + etiqueta de isla actual,
  panel colapsable (tecla `T`), lista de botones por isla con marca `>` en la actual.
- `ESCENAS/sceneOPTM/PlayerSync.gd` (+ `.uid`)
  Réplicas visuales de jugadores remotos (cápsula azul + etiqueta `Jugador <id>`),
  sincronización de posición por RPC no fiable cada 0.1s con interpolación.

## 2. Edits aditivos a archivos existentes
- `ESCENAS/sceneOPTM/ISLA_FINAL.tscn`: nodos nuevos `TeleportManager`, `TeleportPanel`,
  `PlayerSync` + sus `ext_resource` con UID. Nada más tocado.
- `ESCENAS/sceneOPTM/ChunkManager.gd` (funciones originales intactas):
  - `spawned_indices` + guarda anti-duplicados en `_spawn_island_for_player`.
  - `server_disconnected`/`peer_disconnected` → migración de anfitrión: un cliente
    superviviente asume el servidor en el puerto 8910 (si el puerto está ocupado,
    reintenta como cliente).
  - `_on_peer_disconnected` ya no borra el registro: las islas huérfanas se conservan.
  - `_send_orphan_islands()` + reserva con claves ficticias: los nuevos jugadores
    reciben también las islas huérfanas.
  - Apretón de manos `_claim_island` / `_adopt_or_register` (margen 2.5s):
    las reconexiones conservan su isla (1 isla = 1 usuario, sin islas extra).
  - Limpieza de IDs obsoletos tras migración/promoción.
- `PlayerSync`: limpieza total de proxies al caer el servidor (los IDs viejos caducan).

## 3. Bugs encontrados y corregidos
1. Panel no visible → faltaban `.uid`, `layer=100`, anclas explícitas, `_input` en vez
   de `_unhandled_input`, `is_connected` de Godot 3 → `Signal.is_connected`.
2. Cliente no veía islas → `active_islands` solo vive en el servidor; detección por
   nodos + filtros que descartaban la isla 0 y duplicados con mismo nombre.
3. Avatares invisibles → no existía sincronización de jugadores; creado `PlayerSync`.
4. Nueva instancia sin isla tras caída del usuario 1 → migración de anfitrión.
5. Usuario 3 sin 3 islas → reenvío de huérfanas + no borrar registro.
6. 3 islas extra al migrar con 4 usuarios → handshake de reclamación.
7. Etiqueta `Jugador 1` mezclada tras migrar → limpieza de proxies + depuración de IDs.
8. Falta de indicador visual → botón/etiqueta con `Isla N` actual.

## 4. Pendiente / mejoras futuras
- [ ] Sincronizar teleporte en red (hoy es local; stub RPC reservado en `TeleportManager`).
- [ ] Probar con 5+ peers y salidas en distinto orden; revisar carrera si dos clientes
      promocionan a la vez (resuelto por puerto ocupado, pero verificar).
- [ ] Reducir el margen de 2.5s de `_adopt_or_register` o hacerlo adaptativo.
- [ ] Marcar islas `(libre)` cuando el dueño se va (hoy `owner_offline` no se activa).
- [ ] Estilo visual del panel (colores, fuente) y posición configurable.
- [ ] Persistencia entre sesiones (guardar `island_history` en disco).
- [ ] Decidir qué ocurre si el anfitrión migrado también se va en cadena (migración
      en cascada, ya contemplada pero sin prueba de estrés).
- [ ] Verificación en build exportado (hasta ahora solo pruebas en editor con
      instancias locales en `127.0.0.1:8910`).
