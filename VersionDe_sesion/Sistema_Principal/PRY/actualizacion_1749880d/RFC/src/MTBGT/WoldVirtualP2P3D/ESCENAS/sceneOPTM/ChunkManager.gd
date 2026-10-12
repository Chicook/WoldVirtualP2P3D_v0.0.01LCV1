extends Node3D

@export var chunk_size: float = 199.5
@export var islands_per_row: int = 10

@onready var original_island = $ISLA3DCHUNK
@onready var original_ocean = $Oceano3D
@onready var userbase = $USERBASE

var peer = ENetMultiplayerPeer.new()
var next_island_index = 0
var active_islands = {} # Diccionario: player_id -> island_index

func _ready():
	# Escondemos la isla original hasta saber si somos host o cliente
	original_island.hide()
	original_ocean.hide()
	
	multiplayer.connected_to_server.connect(_on_connected_ok)
	multiplayer.connection_failed.connect(_on_connected_fail)
	multiplayer.peer_connected.connect(_on_peer_connected)
	
	# Intentar conectarse como cliente
	print("Buscando red P2P...")
	peer.create_client("127.0.0.1", 8910)
	multiplayer.multiplayer_peer = peer

func _on_connected_ok():
	print("Conectado al mundo P2P exitosamente.")

func _on_connected_fail():
	print("Nadie alojando. Creando servidor local P2P...")
	peer = ENetMultiplayerPeer.new()
	peer.create_server(8910, 32)
	multiplayer.multiplayer_peer = peer
	
	# El servidor registra su propia isla
	_register_player(multiplayer.get_unique_id())

func _on_peer_connected(id):
	if multiplayer.is_server():
		# Enviar a este nuevo jugador las islas que ya existen
		for p_id in active_islands:
			rpc_id(id, "_spawn_island_for_player", p_id, active_islands[p_id])
			
		# Registrar y crear la nueva isla para el jugador que acaba de entrar
		_register_player(id)

func _register_player(id):
	active_islands[id] = next_island_index
	rpc("_spawn_island_for_player", id, next_island_index)
	next_island_index += 1

@rpc("authority", "call_local", "reliable")
func _spawn_island_for_player(id: int, index: int):
	print("Generando isla ", index, " para el jugador ", id)
	
	var row = index / islands_per_row
	var col = index % islands_per_row
	
	var pos_x = 0.0
	var pos_z = -row * chunk_size
	
	if row % 2 == 0:
		pos_x = -col * chunk_size
	else:
		pos_x = -(islands_per_row - 1 - col) * chunk_size
		
	# Instanciar nueva isla
	var new_island = original_island.duplicate()
	new_island.show()
	add_child(new_island)
	new_island.position.x += pos_x
	new_island.position.z += pos_z
	
	var new_ocean = original_ocean.duplicate()
	new_ocean.show()
	add_child(new_ocean)
	new_ocean.position.x += pos_x
	new_ocean.position.z += pos_z
	
	# Si esta isla es para MÍ, muevo MI avatar al centro de esta isla
	if id == multiplayer.get_unique_id():
		userbase.position.x = pos_x
		userbase.position.z = pos_z
