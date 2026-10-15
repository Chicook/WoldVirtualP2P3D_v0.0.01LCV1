extends Node3D
## PlayerSync: replica visual de jugadores remotos (no toca ChunkManager ni userbase).
## Cada peer emite su posicion; los demas muestran un proxy capsula de otro color.

const SYNC_INTERVAL := 0.1

var _userbase: CharacterBody3D = null
var _proxies: Dictionary = {} # peer_id(int) -> MeshInstance3D proxy
var _targets: Dictionary = {} # peer_id(int) -> Vector3 objetivo
var _timer := 0.0

func _ready() -> void:
	_userbase = get_parent().get_node_or_null("USERBASE") as CharacterBody3D
	multiplayer.peer_connected.connect(_on_peer_connected)
	multiplayer.peer_disconnected.connect(_on_peer_disconnected)
	# Al caer el servidor todos los IDs caducan (el nuevo anfitrion sera 1 y los
	# reconectados reciben IDs nuevos): borrar proxies para no mezclar etiquetas.
	multiplayer.server_disconnected.connect(_clear_all_proxies)
	# Crear proxies para peers ya conectados (caso de union tardia)
	for id in multiplayer.get_peers():
		_ensure_proxy(id)

func _process(delta: float) -> void:
	# Suavizar proxies hacia su objetivo
	for id in _proxies.keys():
		var proxy: Node3D = _proxies[id]
		var target: Vector3 = _targets.get(id, proxy.position)
		proxy.position = proxy.position.lerp(target, minf(1.0, delta * 8.0))
	# Emitir posicion propia a intervalos
	_timer += delta
	if _timer < SYNC_INTERVAL:
		return
	_timer = 0.0
	if _userbase != null and multiplayer.multiplayer_peer != null:
		var st := multiplayer.multiplayer_peer.get_connection_status()
		if st == MultiplayerPeer.CONNECTION_CONNECTED:
			rpc("_net_position", _userbase.position)

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

func _remove_proxy(id: int) -> void:
	# Se elimina el avatar pero la isla queda (la conserva TeleportManager).
	if _proxies.has(id):
		(_proxies[id] as Node).queue_free()
		_proxies.erase(id)
		_targets.erase(id)

func _ensure_proxy(id: int) -> void:
	if id == multiplayer.get_unique_id():
		return
	if _proxies.has(id):
		return
	var body := MeshInstance3D.new()
	body.name = "RemotePlayer_%d" % id
	var capsule := CapsuleMesh.new()
	capsule.radius = 0.4
	capsule.height = 1.8
	body.mesh = capsule
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.2, 0.6, 1.0)
	body.material_override = mat
	get_parent().add_child(body)
	# Aparece junto a nuestro avatar hasta recibir primera posicion real
	if _userbase != null:
		body.position = _userbase.position + Vector3(2, 0, 0)
	_proxies[id] = body
	_targets[id] = body.position
	# Etiqueta flotante con el id
	var label := Label3D.new()
	label.text = "Jugador %d" % id
	label.position = Vector3(0, 2.2, 0)
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	body.add_child(label)

@rpc("any_peer", "unreliable")
func _net_position(pos: Vector3) -> void:
	var sender := multiplayer.get_remote_sender_id()
	if sender == 0:
		return
	_ensure_proxy(sender)
	_targets[sender] = pos
