extends Node3D
## PlayerSync: replica visual de jugadores remotos (no toca ChunkManager ni userbase).
## Cada peer emite posicion, rotacion, nick y estado; los demas muestran
## un clon con el avatar humanoide hm1 (idle/andar/correr/aire).

const SYNC_INTERVAL := 0.1
const AVATAR_SCENE: PackedScene = preload("res://ESCENAS/sceneOPTM/constructor/UA/UserAnimados/Import_UserAnimados/ctr_tscn_userBS/scenesconstr/userbaseh1.tscn")

var _userbase: CharacterBody3D = null
var _proxies: Dictionary = {} # peer_id(int) -> Node3D clon
var _targets: Dictionary = {} # peer_id(int) -> Vector3 objetivo
var _estados: Dictionary = {} # peer_id(int) -> int estado animacion
var _nick := ""
var _timer := 0.0

func _ready() -> void:
	_userbase = get_parent().get_node_or_null("USERBASE") as CharacterBody3D
	_nick = _leer_nick()
	multiplayer.peer_connected.connect(_on_peer_connected)
	multiplayer.peer_disconnected.connect(_on_peer_disconnected)
	# Al caer el servidor todos los IDs caducan (el nuevo anfitrion sera 1 y los
	# reconectados reciben IDs nuevos): borrar proxies para no mezclar etiquetas.
	multiplayer.server_disconnected.connect(_clear_all_proxies)
	# Crear proxies para peers ya conectados (caso de union tardia)
	for id in multiplayer.get_peers():
		_ensure_proxy(id)

func _process(delta: float) -> void:
	# Suavizar clones hacia su objetivo + fijar su animación
	for id in _proxies.keys():
		var proxy: Node3D = _proxies[id]
		if not is_instance_valid(proxy):
			continue
		var target: Vector3 = _targets.get(id, proxy.position)
		proxy.position = proxy.position.lerp(target, minf(1.0, delta * 8.0))
		if proxy.has_method("fija_estado_remoto"):
			proxy.call("fija_estado_remoto", int(_estados.get(id, 0)), delta)
	# Emitir estado propio a intervalos
	_timer += delta
	if _timer < SYNC_INTERVAL:
		return
	_timer = 0.0
	if _userbase != null and multiplayer.multiplayer_peer != null:
		var st := multiplayer.multiplayer_peer.get_connection_status()
		if st == MultiplayerPeer.CONNECTION_CONNECTED:
			rpc("_net_position", _userbase.position, _userbase.rotation.y, _nick, _estado_propio())

func _on_peer_connected(id: int) -> void:
	_ensure_proxy(id)

func _on_peer_disconnected(id: int) -> void:
	_remove_proxy(id)

# Limpieza total tras caida del servidor: los IDs antiguos ya no valen.
func _clear_all_proxies() -> void:
	for id in _proxies.keys():
		if is_instance_valid(_proxies[id]):
			(_proxies[id] as Node).queue_free()
	_proxies.clear()
	_targets.clear()
	_estados.clear()

func _remove_proxy(id: int) -> void:
	# Se elimina el avatar pero la isla queda (la conserva TeleportManager).
	if _proxies.has(id):
		(_proxies[id] as Node).queue_free()
		_proxies.erase(id)
		_targets.erase(id)
		_estados.erase(id)

func _ensure_proxy(id: int) -> void:
	if id == multiplayer.get_unique_id():
		return
	if _proxies.has(id):
		return
	# Clon con el avatar humanoide (sin cámara ni control propio).
	var root: CharacterBody3D = AVATAR_SCENE.instantiate() as CharacterBody3D
	root.name = "RemotePlayer_%d" % id
	root.set("es_clon_remoto", true)
	get_parent().add_child(root)
	var cam := root.get_node_or_null("Camera3D")
	if cam:
		cam.queue_free()
	# Aparece junto a nuestro avatar hasta recibir primera posicion real
	if _userbase != null:
		root.position = _userbase.position + Vector3(2, 0, 0)
	_proxies[id] = root
	_targets[id] = root.position
	_estados[id] = 0
	# Etiqueta flotante con el nick (o id si no hay nick)
	var label := Label3D.new()
	label.name = "NickLabel"
	label.text = "Jugador %d" % id
	label.position = Vector3(0, 2.1, 0)
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	root.add_child(label)

@rpc("any_peer", "unreliable")
func _net_position(pos: Vector3, rot_y: float, nick: String, estado: int) -> void:
	var sender := multiplayer.get_remote_sender_id()
	if sender == 0:
		return
	_ensure_proxy(sender)
	_targets[sender] = pos
	_estados[sender] = estado
	var proxy: Node3D = _proxies[sender]
	proxy.rotation.y = rot_y
	var label := proxy.get_node_or_null("NickLabel") as Label3D
	if label:
		label.text = nick if nick != "" else ("Jugador %d" % sender)


## Estado de animación propio (0=idle, 1=andar, 2=correr, 3=aire)
## leído del USERBASE para que los clones lo repliquen.
func _estado_propio() -> int:
	if _userbase == null:
		return 0
	if not is_instance_valid(_userbase) or not _userbase.is_on_floor():
		return 3
	var anim := int(_userbase.get("current_animation"))
	if anim == 2 or anim == 3:
		return 2
	if anim == 1 or anim == 6 or anim == 7:
		return 1
	return 0


## Lee el nick guardado en el registro (user://registro.cfg).
func _leer_nick() -> String:
	var cfg := ConfigFile.new()
	if cfg.load("user://registro.cfg") == OK:
		return str(cfg.get_value("jugador", "nick", ""))
	return ""
