extends Node3D
## Arranque del metaverso:
## 1. REGISTRO con avatar (nick).
## 2. ZONA DEL PORTAL: el avatar camina y lo atraviesa.
## 3. Ventana de carga hasta que la red P2P conecta y aparece MI isla.
## 4. Se cierra la carga y queda el mundo jugable.

const REGISTRO_SCENE: PackedScene = preload("res://ESCENAS/sceneOPTM/constructor/UA/UserAnimados/Import_UserAnimados/ctr_tscn_userBS/scenesconstr/registro.tscn")
const PORTAL_SCENE: PackedScene = preload("res://ESCENAS/sceneOPTM/constructor/UA/UserAnimados/Import_UserAnimados/ctr_tscn_userBS/scenesconstr/entornodeprueba.tscn")
const MUNDO_SCENE: PackedScene = preload("res://ESCENAS/sceneOPTM/constructor/chunk3DBLOCK/ISLA3Dblock.tscn")

var _registro: Control
var _zona_portal: Node3D
var _carga: CanvasLayer
var _carga_label: Label
var _puntos := 0
var _tiempo := 0.0
var _cargando := false


func _ready() -> void:
	_mostrar_registro()


func _process(delta: float) -> void:
	if not _cargando or _carga_label == null:
		return
	_tiempo += delta
	if _tiempo >= 0.5:
		_tiempo = 0.0
		_puntos = (_puntos + 1) % 4
		_carga_label.text = "Conectando a la red P2P" + ".".repeat(_puntos)


func _mostrar_registro() -> void:
	_registro = REGISTRO_SCENE.instantiate() as Control
	add_child(_registro)
	_registro.connect("registro_completado", _on_registro_completado)


func _on_registro_completado(_nick: String) -> void:
	if _registro:
		_registro.queue_free()
		_registro = null
	_mostrar_zona_portal()


func _mostrar_zona_portal() -> void:
	_zona_portal = PORTAL_SCENE.instantiate() as Node3D
	add_child(_zona_portal)
	var portal: Area3D = _zona_portal.get_node("Portal") as Area3D
	portal.connect("portal_cruzado", _on_portal_cruzado)


func _on_portal_cruzado() -> void:
	if _zona_portal:
		_zona_portal.queue_free()
		_zona_portal = null
	_mostrar_carga()
	var mundo: Node3D = MUNDO_SCENE.instantiate() as Node3D
	mundo.name = "ISLA3Dblock"
	add_child(mundo)
	mundo.connect("mi_isla_lista", _on_mi_isla_lista)


func _mostrar_carga() -> void:
	_cargando = true
	_carga = CanvasLayer.new()
	_carga.layer = 200
	add_child(_carga)
	var fondo := ColorRect.new()
	fondo.set_anchors_preset(Control.PRESET_FULL_RECT)
	fondo.color = Color(0.05, 0.06, 0.09, 1)
	_carga.add_child(fondo)
	var centro := CenterContainer.new()
	centro.set_anchors_preset(Control.PRESET_FULL_RECT)
	_carga.add_child(centro)
	_carga_label = Label.new()
	_carga_label.text = "Conectando a la red P2P"
	_carga_label.add_theme_font_size_override("font_size", 28)
	centro.add_child(_carga_label)


func _on_mi_isla_lista() -> void:
	_cargando = false
	if _carga:
		_carga.queue_free()
		_carga = null
