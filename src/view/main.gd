extends Node3D

## Играбельный срез. Кубы вместо моделей, но рамки трекинга настоящие —
## именно они, а не красота, делают игру понятной.

## Габариты в единицах пака: персонаж ~3.1 в высоту, скелет ~2.2.
const KNIGHT_SIZE := Vector3(1.5, 3.1, 1.2)
const WOLF_SIZE := Vector3(1.4, 2.2, 1.0)
const COUNTER_SLOT_X := [-1.5, -0.5, 0.5, 1.5]

var state: GameState
var tracker: FakeTracker
var tracks: Array[Track] = []
var selected: int = 0
var log_lines: Array[String] = []
var _node_of: Dictionary = {}   # object_id -> Node3D
var _actors: Dictionary = {}    # id покупателя -> Node3D, переживает ход
var _undead: Node3D = null
var _passers: Array[Node3D] = []
var _passer_timer := 3.0

## Длительности переходов. Приход дольше ухода: гость идёт от края кадра,
## а уходящий уже стоит рядом.
const WALK_ARRIVE := 2.6
const WALK_SHIFT := 1.1
const WALK_LEAVE := 3.2
const PASSER_PERIOD := 7.0
const PASSER_TRAVEL := 26.0
const PASSER_LIMIT := 3

@onready var camera: Camera3D = $Camera3D
@onready var world: Node3D = $World
@onready var atmosphere: Node3D = $Atmosphere
@onready var world_env: WorldEnvironment = $WorldEnvironment
@onready var moon: DirectionalLight3D = $Moon
@onready var slots: Node3D = $Slots
## Зелья и монеты неподвижны — их держим отдельно и пересоздаём целиком,
## чтобы не мешать актёрам, которые обязаны пережить ход.
var props_holder: Node3D
@onready var boxes: Control = $HUD/Boxes
@onready var status: Label = $HUD/Status
@onready var panel: Label = $HUD/Panel
@onready var actions_label: Label = $HUD/Actions
@onready var status_bg: ColorRect = $HUD/StatusBg
@onready var panel_bg: ColorRect = $HUD/PanelBg

func _ready() -> void:
	SceneBuilder.build(world)
	atmosphere.setup(world_env.environment, moon)
	props_holder = Node3D.new()
	props_holder.name = "Props"
	slots.add_child(props_holder)
	_new_night(randi() % 100000)
	# Режим автоснимка: даёт посмотреть на сцену, не гоняя игру руками.
	#   engine\Godot.exe --path . -- --shot
	if "--shot" in OS.get_cmdline_user_args():
		_shoot_and_quit()

func _shoot_and_quit() -> void:
	# Наполняем сцену: на нулевом ходу очередь пуста и смотреть не на что.
	for i in range(6):
		Turn.world_turn(state, tracker)
	Actions.restock(state, PN.Potion.HEALING)
	Actions.restock(state, PN.Potion.STRENGTH)
	Actions.restock(state, PN.Potion.COURAGE)
	state.wolf_distance = 3
	tracks = tracker.update(SceneView.from_state(state))
	_redraw()
	await get_tree().create_timer(1.2).timeout
	var img := get_viewport().get_texture().get_image()
	var err := img.save_png("res://tools/shot.png")
	print("снимок: ", "res://tools/shot.png" if err == OK else "ОШИБКА %d" % err)
	get_tree().quit(0)

func _new_night(s: int) -> void:
	# Новая ночь — новые люди: старых актёров убираем разом, без проводов.
	for c in slots.get_children():
		if c != props_holder:
			c.queue_free()
	_actors.clear()
	_node_of.clear()
	_passers.clear()
	_undead = null

	state = GameState.new(s)
	tracker = FakeTracker.new(s + 1)
	tracks = tracker.update(SceneView.from_state(state))
	selected = 0
	log_lines = ["Ночь началась. Сид %d." % s]
	_redraw()

# ── ввод ──────────────────────────────────────────────────────

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
			_log("Цель: трек #%d." % kt[selected].track_id)
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
		_log("✗ Нельзя: %s." % r.reason)
		_redraw()
		return

	_log_events(r.events)
	_log_events(Turn.world_turn(state, tracker).events)
	tracks = tracker.update(SceneView.from_state(state))
	if selected >= _knight_tracks().size():
		selected = 0
	_redraw()

# ── треки ─────────────────────────────────────────────────────

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

# ── лог по-человечески ────────────────────────────────────────

func _log(line: String) -> void:
	log_lines.append(line)
	while log_lines.size() > 6:
		log_lines.pop_front()

func _log_events(events: Array[GameEvent]) -> void:
	for e in events:
		var line := _describe(e)
		if line != "":
			_log(line)

func _describe(e: GameEvent) -> String:
	var d := e.data
	match e.kind:
		&"POTION_PLACED":
			return "Достал зелье %s на прилавок." % PN.potion_name(d["potion"])
		&"POTION_HANDED":
			return "✓ Рыцарь %d получил зелье %s — то, что просил." % [
				d["knight"], PN.potion_name(d["potion"])]
		&"POTION_REFUSED":
			return "✗ Рыцарю %d не подошло зелье %s. Он злится, терпение −3." % [
				d["knight"], PN.potion_name(d["potion"])]
		&"KNIGHT_LEAVES_HAPPY":
			return "Рыцарь %d уходит довольный." % d["knight"]
		&"KNIGHT_LEAVES_ANGRY":
			return "✗✗ Рыцарь %d устал ждать и ушёл ни с чем. ПРОМАХ." % d["knight"]
		&"KNIGHT_ARRIVES":
			return "К прилавку подошёл рыцарь %d (плащ %s)." % [
				d["knight"], PN.cloak_name(d["cloak"])]
		&"COIN_THROWN":
			return "Монета покатилась по доскам — успей увидеть её трекером."
		&"COIN_COLLECTED":
			return "✓ Монета подобрана."
		&"COIN_LOST":
			return "✗ Монета укатилась в темноту — трекер её потерял."
		&"LANTERN_LIT":
			return "Зажёг фонарь на %d хода. Видно лучше, но волк осмелеет." % d["turns"]
		&"LANTERN_DIED":
			return "Фонарь погас."
		&"ROCK_THROWN":
			return "Бросил камень."
		&"ROCK_MISSED":
			return "✗ Промах — трекер потерял волка, кидал вслепую. Ход потрачен."
		&"WOLF_APPEARS":
			return "В темноте загорелись глаза. Волк близко."
		&"WOLF_STEPS":
			return "Волк подкрался ближе (дистанция %d)." % d["distance"]
		&"WOLF_STEALS":
			if d["potion"] < 0:
				return "✗ Волк дошёл до прилавка, но красть было нечего."
			return "✗ Волк утащил зелье %s!" % PN.potion_name(d["potion"])
		&"WOLF_FLEES":
			return "Волк скрылся в темноте."
		&"TRACKER_RESET":
			return "Перезахват: трекер сопоставил всё заново. Номера сменились."
		&"DAWN":
			return "═══ РАССВЕТ. Заработано монет: %d. ═══" % d["coins"]
		&"NIGHT_FAILED":
			return "═══ ТРИ ПРОМАХА. Ночь провалена. Монет: %d. ═══" % d["coins"]
	return ""

# ── отрисовка ─────────────────────────────────────────────────

func _redraw() -> void:
	# Небо, свет и дождь ведутся теми же числами, что роняют уверенность
	# трекера, — картинка и механика не расходятся по определению.
	atmosphere.set_weather(state.rain(), state.dark(), state.turn, PN.NIGHT_TURNS)
	_sync_scene()
	status.text = _status_text()
	panel.text = _panel_text()
	actions_label.text = _actions_text()
	_push_boxes()
	# Фоны подгоняются под содержимое: при длинном логе текст вылезал за блок.
	# Отложено на кадр — до перерисовки Label ещё не знает своей высоты.
	call_deferred("_fit_backgrounds")

func _fit_backgrounds() -> void:
	status_bg.size.y = _text_height(status) + 20.0
	var wanted: float = _text_height(panel) + 20.0
	panel_bg.size.y = wanted
	panel.size.y = wanted - 20.0

func _text_height(label: Label) -> float:
	# get_content_height() у Label нет — считаем по строкам с учётом переноса
	return float(label.get_line_count()) * float(label.get_line_height())

## Сцена больше не пересобирается каждый ход — она синхронизируется.
## Актёры переживают ход и переходят между позициями, иначе никакой ходьбы
## быть не может: покупатель, которого сносят и создают заново, всегда
## возникает из ниоткуда.
func _sync_scene() -> void:
	_sync_customers()
	_sync_undead()
	_sync_props()

func _sync_customers() -> void:
	var alive := {}
	for k in state.knights:
		if k.slot < 0:
			continue
		alive[k.id] = true
		var target: Vector3 = SceneBuilder.SLOTS[k.slot]
		var actor: Node3D = _actors.get(k.id)

		if actor == null:
			# Приходит со стороны замка, из-за края кадра.
			actor = CustomerActor.spawn(k.id, _cloak_color(k.cloak))
			actor.position = SceneBuilder.road_entry()
			slots.add_child(actor)
			_actors[k.id] = actor
			_attach_truth(actor)
			_walk(actor, target, WALK_ARRIVE, true)
		elif actor.position.distance_to(target) > 0.25:
			_walk(actor, target, WALK_SHIFT, true)

		_node_of[k.id] = actor
		_update_truth(actor, k)

	for id in _actors.keys():
		if alive.has(id):
			continue
		var leaving: Node3D = _actors[id]
		_actors.erase(id)
		_node_of.erase(id)
		_walk_away(leaving)

func _sync_undead() -> void:
	if state.wolf_distance < 0:
		if _undead != null and is_instance_valid(_undead):
			_walk_away(_undead)
		_undead = null
		_node_of.erase(-1)
		return

	# Выходит из леса по ту сторону дороги и идёт к лавке, пересекая её.
	var target: Vector3 = SceneBuilder.WOLF_LANE \
		- SceneBuilder.WOLF_STEP * float(state.wolf_distance)

	if _undead == null or not is_instance_valid(_undead):
		_undead = Props.skeleton()
		_undead.position = SceneBuilder.WOLF_LANE \
			- SceneBuilder.WOLF_STEP * float(SceneBuilder.WOLF_SPAWN_DISTANCE)
		slots.add_child(_undead)

	if _undead.position.distance_to(target) > 0.25:
		_walk(_undead, target, WALK_SHIFT, true)
	_node_of[-1] = _undead

## Зелья и монеты неподвижны, их проще пересоздавать.
func _sync_props() -> void:
	for c in props_holder.get_children():
		c.queue_free()

	for i in range(state.counter.size()):
		var bottle := Props.bottle(state.counter[i])
		bottle.position = SceneBuilder.counter_spot(i)
		props_holder.add_child(bottle)

	for coin_id in state.coins_in_flight:
		var coin := Props.coin()
		coin.position = Vector3(-0.5, SceneBuilder.COUNTER_TOP + 0.15, -1.5)
		props_holder.add_child(coin)
		_node_of[coin_id] = coin

# ── ходьба ────────────────────────────────────────────────────

func _walk(actor: Node3D, to: Vector3, duration: float, face_counter: bool) -> void:
	var old = actor.get_meta("walk_tween", null)
	if old is Tween and old.is_valid():
		old.kill()

	var anim := actor.find_child("AnimationPlayer", true, false) as AnimationPlayer
	if anim != null and anim.has_animation("Walking_A"):
		anim.play("Walking_A")

	var dir := to - actor.position
	if dir.length() > 0.05:
		actor.rotation.y = atan2(dir.x, dir.z)

	var tw := create_tween()
	tw.tween_property(actor, "position", to, duration) \
		.set_trans(Tween.TRANS_SINE).set_ease(Tween.EASE_IN_OUT)
	actor.set_meta("walk_tween", tw)
	tw.finished.connect(func() -> void:
		if not is_instance_valid(actor):
			return
		if anim != null and anim.has_animation("Idle"):
			anim.play("Idle")
		if face_counter:
			var back := Vector3.ZERO - actor.position
			actor.rotation.y = atan2(back.x, back.z))

func _walk_away(actor: Node3D) -> void:
	var old = actor.get_meta("walk_tween", null)
	if old is Tween and old.is_valid():
		old.kill()

	var anim := actor.find_child("AnimationPlayer", true, false) as AnimationPlayer
	if anim != null and anim.has_animation("Walking_A"):
		anim.play("Walking_A")

	var exit := SceneBuilder.road_exit()
	var dir := exit - actor.position
	if dir.length() > 0.05:
		actor.rotation.y = atan2(dir.x, dir.z)

	var tw := create_tween()
	tw.tween_property(actor, "position", exit, WALK_LEAVE)
	tw.tween_callback(actor.queue_free)

# ── прохожие ──────────────────────────────────────────────────

## Просто идут мимо: из замка прочь и обратно в замок. К лавке отношения
## не имеют, трекер их не берёт — это фон, от которого дорога живая.
func _process(delta: float) -> void:
	var live: Array[Node3D] = []
	for p in _passers:
		if is_instance_valid(p):
			live.append(p)
	_passers = live

	if state == null or state.finished:
		return

	_passer_timer -= delta
	if _passer_timer > 0.0:
		return

	# В сумерках и на рассвете дорога людная, глубокой ночью почти пуста.
	_passer_timer = PASSER_PERIOD / maxf(0.18, 1.0 - state.dark())
	if _passers.size() < PASSER_LIMIT:
		_spawn_passer()

func _spawn_passer() -> void:
	var to_castle := randf() < 0.5
	var from_p: Vector3 = SceneBuilder.road_exit() if to_castle \
		else SceneBuilder.road_entry()
	var to_p: Vector3 = SceneBuilder.road_entry() if to_castle \
		else SceneBuilder.road_exit()

	# Каждый держится своей стороны дороги, иначе встречные проходят насквозь.
	var yaw := SceneBuilder.road_yaw(0.0)
	var lane := Vector3(cos(yaw), 0.0, -sin(yaw)) * (1.7 if to_castle else -1.7)

	var actor := CustomerActor.spawn(randi() % 997, _cloak_color(randi() % PN.Cloak.size()))
	actor.position = from_p + lane
	slots.add_child(actor)
	_passers.append(actor)

	var anim := actor.find_child("AnimationPlayer", true, false) as AnimationPlayer
	if anim != null and anim.has_animation("Walking_A"):
		anim.play("Walking_A")

	var target := to_p + lane
	var dir := target - actor.position
	actor.rotation.y = atan2(dir.x, dir.z)

	var tw := create_tween()
	tw.tween_property(actor, "position", target, PASSER_TRAVEL)
	tw.tween_callback(actor.queue_free)

# ── табличка правды ───────────────────────────────────────────

func _attach_truth(actor: Node3D) -> void:
	var truth := Label3D.new()
	truth.name = "Truth"
	truth.font_size = 40
	truth.pixel_size = 0.0038
	truth.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	truth.modulate = Color(0.66, 0.66, 0.72)
	truth.outline_modulate = Color(0, 0, 0, 0.9)
	truth.outline_size = 10
	truth.no_depth_test = true
	# У ног, а не над головой: сверху висит подпись трекера, и две надписи
	# об одном персонаже дрались за одно место. Смещения вперёд нет —
	# иначе она бы качалась при развороте актёра.
	truth.position = Vector3(0.0, 0.3, 0.0)
	actor.add_child(truth)

func _update_truth(actor: Node3D, k: Knight) -> void:
	var truth := actor.get_node_or_null("Truth") as Label3D
	if truth == null:
		return
	truth.text = "правда: %s / %s\nтерпение %d" % [
		PN.cloak_name(k.cloak), PN.potion_name(k.order), k.patience]

func _make_box(size: Vector3, colour: Color) -> MeshInstance3D:
	var node := MeshInstance3D.new()
	var mesh := BoxMesh.new()
	mesh.size = size
	node.mesh = mesh
	var mat := StandardMaterial3D.new()
	mat.albedo_color = colour
	node.material_override = mat
	return node

func _cloak_color(cloak: int) -> Color:
	match cloak:
		PN.Cloak.RED: return Color(0.78, 0.19, 0.19)
		PN.Cloak.BLUE: return Color(0.19, 0.34, 0.78)
		PN.Cloak.GREEN: return Color(0.19, 0.6, 0.31)
		PN.Cloak.YELLOW: return Color(0.83, 0.72, 0.2)
		PN.Cloak.PURPLE: return Color(0.5, 0.28, 0.7)
	return Color.WHITE

func _push_boxes() -> void:
	var list: Array = []
	for tr in tracks:
		var node: Node3D = _node_of.get(tr.object_id)
		if node == null:
			continue
		# У всех моделей пака точка отсчёта в ногах, рамку центрируем сами.
		var size := KNIGHT_SIZE
		match tr.kind:
			&"wolf": size = WOLF_SIZE
			&"coin": size = Vector3(0.5, 0.4, 0.5)
		var centre := node.global_position
		centre.y += size.y * 0.5
		list.append({"pos": centre, "size": size, "track": tr})
	boxes.set_entries(camera, list, _selected_track_id())

# ── тексты панелей ────────────────────────────────────────────

func _status_text() -> String:
	var lines: Array[String] = []
	lines.append("НОЧЬ  ход %d из %d" % [state.turn, PN.NIGHT_TURNS])
	lines.append("монеты   %d" % state.coins)
	lines.append("промахи  %d из %d" % [state.misses, PN.MAX_MISSES])
	lines.append("")
	lines.append("дождь  %s" % _bar(state.rain()))
	lines.append("тьма   %s" % _bar(state.dark()))
	lines.append("фонарь %s" % ("горит, %d хода" % state.lantern_turns if state.lantern_on() else "погашен"))
	lines.append("волк   %s" % ("нет" if state.wolf_distance < 0 else "дистанция %d" % state.wolf_distance))
	return "\n".join(lines)

func _bar(v: float) -> String:
	var filled := int(round(v * 5.0))
	return "█".repeat(filled) + "░".repeat(5 - filled) + " %.2f" % v

func _panel_text() -> String:
	var lines: Array[String] = []
	lines.append("ПРИЛАВОК")
	if state.counter.is_empty():
		lines.append("  пусто — доставай из сундука")
	else:
		for i in range(state.counter.size()):
			lines.append("  [%d] %s" % [i + 1, PN.potion_name(state.counter[i])])
	lines.append("")
	lines.append("СУНДУК")
	lines.append("  [Q] лечения   %d" % state.chest[PN.Potion.HEALING])
	lines.append("  [W] силы      %d" % state.chest[PN.Potion.STRENGTH])
	lines.append("  [E] храбрости %d" % state.chest[PN.Potion.COURAGE])
	lines.append("")
	lines.append("СОБЫТИЯ")
	for l in log_lines:
		lines.append("  " + l)
	return "\n".join(lines)

func _actions_text() -> String:
	if state.finished:
		return "═══ %s ═══   монет %d   ·   ПРОБЕЛ — новая ночь   ·   ESC — выход" % [
			"РАССВЕТ" if state.outcome == "dawn" else "ПРОВАЛ", state.coins]

	var sel := _selected_track_id()
	var kt := _knight_tracks()
	var head := "нет целей"
	if sel >= 0:
		var tr := _track_by_id(sel)
		if tr != null:
			head = "цель: рамка #%d — трекер уверяет, что он хочет %s" % [
				tr.track_id, PN.potion_name(tr.remembered_order)]

	var parts: Array[String] = []
	for i in range(state.counter.size()):
		parts.append("[%d] отдать %s" % [i + 1, PN.potion_name(state.counter[i])])
	if parts.is_empty():
		parts.append("прилавок пуст")

	var lines: Array[String] = []
	lines.append("%s        [TAB] следующая цель из %d" % [head, kt.size()])
	lines.append("   ".join(parts))
	lines.append("[Q][W][E] достать из сундука   [L] фонарь   [R] камень   [T] перезахват   [ESC] выход")
	return "\n".join(lines)

func _track_by_id(id: int) -> Track:
	for tr in tracks:
		if tr.track_id == id:
			return tr
	return null
