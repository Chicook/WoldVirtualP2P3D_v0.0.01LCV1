# RESUMEN DE SESIÓN — WoldVirtualP2P3D (metaverso)

Fecha: 2026-10-16. Proyecto: `VersionDe_sesion/CONTRRF/TLLDCD/src/MTBGT/WoldVirtualP2P3D`.

## 1. Flujo de entrada (nuevo)
- `gdscrip/Boot.gd` (nuevo): arranque en 4 fases.
  1. **REGISTRO** con avatar (`scenesconstr/registro.tscn` + `gdscrip/registro.gd`): preview 3D del humanoide hm1 de frente (idle), campo nick (mín. 3 letras), guardado en `user://registro.cfg`. Emite `registro_completado`; si se ejecuta suelto cambia de escena a `mnpr.tscn`.
  2. **ZONA DEL PORTAL** (`scenesconstr/entornodeprueba.tscn`): suelo físico real, marca roja emisiva + `Area3D Portal` (`gdscrip/portal_csgbox5.gd`, señal `portal_cruzado`; en uso suelto cambia a `ISLA3Dblock.tscn`). El avatar aparece mirando al portal (rotado 180°).
  3. **VENTANA DE CARGA** ("Conectando a la red P2P...") mientras el mundo instancia `ISLA3Dblock` y la red ENet (puerto 8910) conecta.
  4. **ISLA**: al llegar `ChunkManager.mi_isla_lista` se cierra la carga. Cámara recolocada con `snap_camera()`.
- `WoldVirtual3DP2P.tscn`: ahora solo entorno + luz + script Boot (la isla se instancia en diferido).

## 2. Avatar humanoide en el metaverso
- `ISLA3Dblock.tscn`: instancia `USERBASE` cambiada de `user/userbase.tscn` (cápsula) a `.../ctr_tscn_userBS/scenesconstr/userbaseh1.tscn` (hm1 + AnimationTree). Nombre `USERBASE`, `unique_id` y spawn intactos → ChunkManager/PlayerSync/TeleportManager lo encuentran igual.
- `userbaseh1.tscn` del metaverso: rutas `res://` corregidas (apuntaban a `res://Import_UserAnimados/...` inexistente), escena limpia (solo jugador + cámara).
- `userbaseh_1.gd` refactorizado en 9 regiones: cámara orbital con retardo (`follow_speed 6.0`, el cuerpo gira al instante y se VE girar), entrada invertida 180° porque hm1 mira a +Z, sprint AltGr, `snap_camera()` para teleporte/spawn, modo `es_clon_remoto` + `fija_estado_remoto(0-3)` para réplicas.
- Clones remotos (`PlayerSync.gd`): ya no son cápsulas; instancia `userbaseh1.tscn` sin cámara, con nick del registro y animación replicada (idle/andar/correr/aire). RPC `_net_position(pos, rot_y, nick, estado)`. Antes salían medio enterrados (malla centrada en pies) y sin giro.

## 3. Red P2P e islas (cambios compatibles)
- `ChunkManager.gd`: señal `mi_isla_lista`; `_claim_island(idx, nick)`; `owner_names` (peer→nick) e `island_owners` (isla→[dueño, nick], vive en TODOS los peers); el nick viaja en `_spawn_island_for_player(id, index, nick)`; `_nick_unico()` sufija repetidos (`Ana#2`).
- `TeleportManager.gd`: `get_islands()` devuelve `index/position/offline/nombre/uuid`; `_uuid_corto()` determinista de 6 caracteres (sin 0/O/1/I/L); `_nombre_dueno()` prefiere `island_owners` (funciona en clientes, que no ven `active_islands`).
- `TeleportPanel.gd`: filas `Isla {nombre} ({uuid})`, etiqueta superior con los 3 valores, barra superior con `HBoxContainer` + recorte (sin solapes), panel a 300px.
- Avisos del analizador corregidos (división entera, Variant, parámetros/variables sin uso) + accesos entre scripts por `connect("señal")`/`call()`/`get()`.

## 4. Escenas de prueba
- `mnpr.tscn` (proyecto UserAnimados): suelo físico real `StaticBody3D` (el CSG daba colisión poco fiable), jugador a y=1.0, marcas sin colisión.
- `userbaseh1.tscn` (UserAnimados): `Camera3D` devuelta a raíz (estaba bajo `User1H1` y rompía `$Camera3D`).

## 5. Deuda técnica
1. `registro.tscn`, `mnpr.tscn`, `entornodeprueba.tscn` y `mnprCPOF` del metaverso aún con rutas `res://Import_UserAnimados/...` rotas (no afectan al juego, solo a pruebas sueltas). `registro.tscn` y `entornodeprueba.tscn` ya corregidos; verificar `mnpr.tscn`/`mnprCPOF`.
2. `userbase.tscn` + `gdscrip/userbase.gd` (cápsula) quedan como respaldo sin uso: decidir borrar o conservar.
3. `PlayerSync` no limpia `owner_names`/`island_owners` de peers definitivamente caídos (nicks fantasma con sufijo hasta reinicio).
4. Tras migración de anfitrión, el nuevo host pierde `owner_names` (lo reconstruye con claims, pero islas huérfanas quedan con "—").
5. Réplicas remotas: sin strafe/atrás/jump diferenciados (solo idle/andar/correr/aire) y sin interpolación de rotación.
6. Doble luz/entorno: `userbaseh1` no trae, pero `registro.tscn` y principal sí tienen los suyos; revisar sobreexposición al convivir instancias.
7. Sin pruebas automatizadas; todo verificado manual en consola (yaw, nodos, colisiones).
8. `ApplyMaterial.gd` inerte y `P2PC#/` sin revisar: confirmar si se usan.

## 6. Hoja de ruta: selector y personalización de avatares (usuarios únicos)
Objetivo: cada usuario elige y personaliza su avatar; su clon se ve igual en todos los peers.

- **Fase 1 — Catálogo**: carpeta `avatars/` con variantes (hm1, hm2...). Cada avatar = `.tscn` + thumbnail `.png` + `avatar.cfg` (nombre, colores base, escala). El registro muestra carrusel con flechas + nombre; guarda `avatar_id` en `user://registro.cfg`.
- **Fase 2 — Personalización**: color de piel/ropa/pelo (paleta de 8-12 `StandardMaterial3D` aplicados por nombre de malla: `Cabeza/Torso/...`), altura/escala (0.95–1.05), nick visible. Guardar `dict` de personalización en cfg.
- **Fase 3 — Réplica fiel**: el RPC lleva `avatar_id + personalización`; `PlayerSync` instancia el avatar correspondiente y aplica materiales/escala al clon. Versiones distintas de catálogo: fallback a hm1 + aviso.
- **Fase 4 — Identidad única**: UUID de usuario persistente (generado en primer registro, guardado en cfg) independiente del nick y del peer_id; el panel muestra `Isla {nick} ({uuid6})` y la etiqueta del clon usa nick+uuid corto. Resolver colisiones de nick en servidor (ya existe `_nick_unico`).
- **Fase 5 — Persistencia**: guardar isla/nick/avatar en servidor o archivo compartido para reconexiones (hoy `island_history` es solo runtime).
