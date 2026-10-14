extends CanvasLayer
## TeleportPanel: UI lateral colapsable (tecla T o boton). Solo usa TeleportManager.

var manager: Node = null
var panel: PanelContainer = null
var list_box: VBoxContainer = null
var toggle_btn: Button = null
var location_lbl: Label = null
var _userbase: Node3D = null
var _expanded := true
var _cooldown := 0.0
var _loc_timer := 0.0
var _current_island := -2 # ultimo indice mostrado (-2 = sin inicializar)

func _ready() -> void:
	layer = 100
	visible = true
	_build_ui()
	# Conexion diferida para asegurar que TeleportManager ya hizo _ready
	call_deferred("_late_connect")

func _late_connect() -> void:
	manager = get_parent().get_node_or_null("TeleportManager")
	if manager != null:
		if manager.has_signal("islands_updated"):
			if not manager.islands_updated.is_connected(_on_islands_updated):
				manager.connect("islands_updated", _on_islands_updated)
		if manager.has_method("get_islands"):
			_on_islands_updated(manager.get_islands())
			return
	# Si aun no hay manager, mostrar algo igualmente para verificar visibilidad
	_show_debug()

func _build_ui() -> void:
	# Boton siempre visible arriba-izquierda con anclas explicitas
	toggle_btn = Button.new()
	toggle_btn.text = "Isla ? [T]"
	toggle_btn.set_anchors_preset(Control.PRESET_TOP_LEFT)
	toggle_btn.position = Vector2(10, 10)
	toggle_btn.size = Vector2(120, 36)
	toggle_btn.visible = true
	toggle_btn.pressed.connect(_toggle_panel)
	add_child(toggle_btn)

	# Etiqueta con la isla actual, a la derecha del boton
	location_lbl = Label.new()
	location_lbl.text = "Isla ?"
	location_lbl.set_anchors_preset(Control.PRESET_TOP_LEFT)
	location_lbl.position = Vector2(138, 16)
	location_lbl.add_theme_font_size_override("font_size", 20)
	add_child(location_lbl)

	panel = PanelContainer.new()
	panel.set_anchors_preset(Control.PRESET_TOP_LEFT)
	panel.position = Vector2(10, 54)
	panel.size = Vector2(200, 320)
	panel.custom_minimum_size = Vector2(200, 320)
	panel.visible = true
	add_child(panel)

	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size = Vector2(200, 320)
	scroll.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	panel.add_child(scroll)

	list_box = VBoxContainer.new()
	list_box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(list_box)

	var lbl := Label.new()
	lbl.text = "Cargando islas..."
	lbl.name = "DebugLabel"
	list_box.add_child(lbl)

func _show_debug() -> void:
	pass # el label "Cargando islas..." ya demuestra visibilidad

func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_T:
			_toggle_panel()

func _toggle_panel() -> void:
	_expanded = not _expanded
	if panel != null:
		panel.visible = _expanded

func _process(delta: float) -> void:
	if _cooldown > 0.0:
		_cooldown -= delta
	_loc_timer += delta
	if _loc_timer >= 0.3:
		_loc_timer = 0.0
		_refresh_location()

# Actualiza boton y etiqueta con la isla bajo el avatar (Isla 0, Isla 1...).
func _refresh_location() -> void:
	if manager == null or not manager.has_method("current_island_index"):
		return
	if _userbase == null or not is_instance_valid(_userbase):
		_userbase = get_parent().get_node_or_null("USERBASE") as Node3D
		if _userbase == null:
			return
	var idx: int = manager.current_island_index(_userbase.position)
	if idx == _current_island:
		return
	_current_island = idx
	var txt := "Isla %d" % idx if idx >= 0 else "Mar"
	toggle_btn.text = "%s [T]" % txt
	location_lbl.text = txt
	_mark_current_button(idx)

func _on_islands_updated(islands: Array) -> void:
	if list_box == null:
		return
	for c in list_box.get_children():
		c.queue_free()
	if islands.is_empty():
		var lbl := Label.new()
		lbl.text = "Buscando islas..."
		list_box.add_child(lbl)
		return
	for entry in islands:
		var idx: int = int(entry["index"])
		var offline: bool = bool(entry.get("offline", false))
		var b := Button.new()
		b.text = "Isla %d%s" % [idx, " (libre)" if offline else ""]
		b.tooltip_text = "Teletransportarse a Isla %d" % idx
		b.custom_minimum_size = Vector2(180, 32)
		b.set_meta("island_index", idx)
		b.set_meta("offline", offline)
		b.pressed.connect(_on_goto.bind(idx))
		list_box.add_child(b)
	_mark_current_button(_current_island)

# Marca con "> " el boton de la isla actual en la lista.
func _mark_current_button(idx: int) -> void:
	if list_box == null:
		return
	for c in list_box.get_children():
		if c is Button and c.has_meta("island_index"):
			var b_idx: int = int(c.get_meta("island_index"))
			var base := "Isla %d" % b_idx
			if bool(c.get_meta("offline", false)):
				base += " (libre)"
			c.text = ("> " + base) if b_idx == idx else base

func _on_goto(index: int) -> void:
	if _cooldown > 0.0 or manager == null:
		return
	_cooldown = 0.5
	manager.teleport_to(index)
