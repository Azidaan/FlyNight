extends Node3D

## Уродливый играбельный срез. Кубы вместо моделей, текст вместо панелей.
## Смотреть не на что — играть можно. Это точка первого плейтеста.

const SLOT_X := [0.0, 1.7, 3.4]
const SLOT_Z := [-2.0, -3.6, -5.2]

var state: GameState
var tracker: FakeTracker
var tracks: Array[Track] = []
var selected: int = 0
var log_lines: Array[String] = []

@onready var hud: Label = $HUD/Text
@onready var slots: Node3D = $Slots

func _ready() -> void:
	_new_night(randi() % 100000)

func _new_night(s: int) -> void:
	state = GameState.new(s)
	tracker = FakeTracker.new(s + 1)
	tracks = tracker.update(SceneView.from_state(state))
	selected = 0
	log_lines = ["Ночь началась. Сид %d" % s]
	_redraw()

func _knight_tracks() -> Array[Track]:
	var out: Array[Track] = []
	for tr in tracks:
		if tr.kind == &"knight":
			out.append(tr)
	return out

func _selected_track_id() -> int:
	var kt := _knight_tracks()
	if kt.is_empty():
		return -1
	return kt[selected % kt.size()].track_id

func _unhandled_key_input(event: InputEvent) -> void:
	if not event.is_pressed() or event.is_echo():
		return

	if event.keycode == KEY_ESCAPE:
		get_tree().quit(0)
		return

	if state.finished:
		if event.keycode == KEY_SPACE:
			_new_night(randi() % 100000)
		return

	if event.keycode == KEY_TAB:
		var kt := _knight_tracks()
		if not kt.is_empty():
			selected = (selected + 1) % kt.size()
			_redraw()
		return

	var r: ActionResult = null
	match event.keycode:
		KEY_Q: r = Actions.restock(state, PN.Potion.HEALING)
		KEY_W: r = Actions.restock(state, PN.Potion.STRENGTH)
		KEY_E: r = Actions.restock(state, PN.Potion.COURAGE)
		KEY_L: r = Actions.lantern(state)
		KEY_R: r = Actions.rock(state, tracks)
		KEY_T: r = Actions.recapture(state, tracker)
		KEY_1, KEY_2, KEY_3, KEY_4:
			r = Actions.give(state, tracks, _selected_track_id(), event.keycode - KEY_1)
		_: return

	if r == null:
		return
	if not r.ok:
		_log("отказ: " + r.reason)
		_redraw()
		return

	_log_events(r.events)
	_log_events(Turn.world_turn(state, tracker).events)
	tracks = tracker.update(SceneView.from_state(state))
	_redraw()

func _log(line: String) -> void:
	log_lines.append(line)
	while log_lines.size() > 7:
		log_lines.pop_front()

func _log_events(events: Array[GameEvent]) -> void:
	for e in events:
		_log("  " + String(e.kind) + " " + str(e.data))

func _redraw() -> void:
	_redraw_scene()
	hud.text = "\n".join(_hud_lines())

func _redraw_scene() -> void:
	for c in slots.get_children():
		c.queue_free()

	for k in state.knights:
		if k.slot < 0:
			continue
		var box := MeshInstance3D.new()
		var mesh := BoxMesh.new()
		mesh.size = Vector3(0.6, 1.6, 0.4)
		box.mesh = mesh
		box.position = Vector3(SLOT_X[k.slot], 0.8, SLOT_Z[k.slot])
		var mat := StandardMaterial3D.new()
		mat.albedo_color = _cloak_color(k.cloak)
		box.material_override = mat
		slots.add_child(box)

	if state.wolf_distance >= 0:
		var wolf := MeshInstance3D.new()
		var wmesh := BoxMesh.new()
		wmesh.size = Vector3(0.9, 0.5, 0.4)
		wolf.mesh = wmesh
		wolf.position = Vector3(-2.5, 0.3, -1.0 - state.wolf_distance * 0.8)
		var wmat := StandardMaterial3D.new()
		wmat.albedo_color = Color(0.5, 0.1, 0.1)
		wolf.material_override = wmat
		slots.add_child(wolf)

func _cloak_color(cloak: int) -> Color:
	match cloak:
		PN.Cloak.RED: return Color(0.75, 0.2, 0.2)
		PN.Cloak.BLUE: return Color(0.2, 0.35, 0.75)
		PN.Cloak.GREEN: return Color(0.2, 0.6, 0.3)
		PN.Cloak.YELLOW: return Color(0.8, 0.7, 0.2)
		PN.Cloak.PURPLE: return Color(0.5, 0.3, 0.7)
	return Color.WHITE

func _hud_lines() -> Array[String]:
	var lines: Array[String] = []
	lines.append("ход %d/%d   монеты %d   промахи %d/%d   дождь %.2f   тьма %.2f   фонарь %s" % [
		state.turn, PN.NIGHT_TURNS, state.coins, state.misses, PN.MAX_MISSES,
		state.rain(), state.dark(), "горит" if state.lantern_on() else "нет"])

	var shelf: Array[String] = []
	for i in range(state.counter.size()):
		shelf.append("%d:%s" % [i + 1, PN.potion_name(state.counter[i])])
	lines.append("прилавок: " + (", ".join(shelf) if not shelf.is_empty() else "пусто"))
	lines.append("сундук: лечения %d, силы %d, храбрости %d" % [
		state.chest[PN.Potion.HEALING], state.chest[PN.Potion.STRENGTH],
		state.chest[PN.Potion.COURAGE]])
	lines.append("волк: " + ("нет" if state.wolf_distance < 0 else "дистанция %d" % state.wolf_distance))
	lines.append("")
	lines.append("--- ПОКАЗАНИЯ ТРЕКЕРА ---")

	var sel := _selected_track_id()
	var state_names := ["ЗАХВАТ", "ДРЕЙФ ", "ПОТЕРЯ"]
	for tr in tracks:
		var mark := ">" if tr.track_id == sel else " "
		if tr.kind == &"knight":
			var order_txt := "—" if tr.state == PN.TrackState.LOST \
				else PN.potion_name(tr.remembered_order)
			lines.append("%s #%-3d %s %.2f  плащ %-11s хочет %s" % [
				mark, tr.track_id, state_names[tr.state], tr.conf,
				PN.cloak_name(tr.remembered_cloak), order_txt])
		else:
			lines.append("%s #%-3d %s %.2f  [%s]" % [
				mark, tr.track_id, state_names[tr.state], tr.conf, tr.kind])

	lines.append("")
	lines.append("--- ПРАВДА МИРА (для отладки, в готовой игре этого не будет) ---")
	for k in state.knights:
		lines.append("   рыцарь %d  слот %d  плащ %-11s хочет %-11s терпение %d" % [
			k.id, k.slot, PN.cloak_name(k.cloak), PN.potion_name(k.order), k.patience])

	lines.append("")
	lines.append("TAB выбрать цель · 1-4 отдать зелье с прилавка · Q/W/E достать из сундука")
	lines.append("L фонарь · R камень · T перезахват · ESC выход")

	if state.finished:
		lines.append("")
		lines.append("═══ %s ═══  монет %d  ·  ПРОБЕЛ — новая ночь" % [
			"РАССВЕТ" if state.outcome == "dawn" else "ПРОВАЛ", state.coins])

	lines.append("")
	lines.append_array(log_lines)
	return lines
