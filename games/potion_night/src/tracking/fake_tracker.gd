class_name FakeTracker
extends TrackSource

## Имитация трекера. Считает уверенность по формуле из раздела 5 спеки и
## иногда путает ID соседних по очереди рыцарей.
##
## Подмена обязана быть честной, а не рандомным наказанием: при склейке треков
## меняется только привязка (object_id), а запомненный признак (remembered_cloak)
## остаётся прежним. Игрок видит «плащ красный» в панели и синий плащ на экране —
## и может уличить трекер, если не торопится.

var noise_amplitude: float = 0.05
var _seed: int
var _next_track_id: int = 1
var _by_object: Dictionary = {}   # object_id -> Track

func _init(p_seed: int) -> void:
	_seed = p_seed

func update(view: SceneView) -> Array[Track]:
	var out: Array[Track] = []
	var seen: Dictionary = {}

	for o in view.objects:
		seen[o.object_id] = true
		var tr: Track = _by_object.get(o.object_id)
		if tr == null:
			tr = Track.new(_next_track_id, o.kind, o.object_id)
			tr.remembered_cloak = o.cloak
			tr.remembered_order = o.order
			_next_track_id += 1
			_by_object[o.object_id] = tr
		else:
			tr.age += 1

		tr.conf = clampf(confidence(view, o) + _noise(view.turn, tr.track_id), 0.0, 1.0)
		tr.refresh_state()
		if tr.state != PN.TrackState.LOST:
			tr.last_known_pos = o.position
		out.append(tr)

	for key in _by_object.keys():
		if not seen.has(key):
			_by_object.erase(key)

	return out

func confidence(view: SceneView, o: SceneObject) -> float:
	var dark_factor := 0.25 if view.lantern_on else 1.0
	var c := 1.0
	c -= 0.30 * view.rain
	c -= 0.35 * view.dark * dark_factor
	c -= 0.20 * o.dist_norm
	c -= 0.25 * o.occlusion
	return clampf(c, 0.0, 1.0)

func _noise(turn: int, track_id: int) -> float:
	if noise_amplitude <= 0.0:
		return 0.0
	var h := absi(hash("%d:%d:%d" % [_seed, turn, track_id]))
	return (float(h % 1000) / 1000.0) * (2.0 * noise_amplitude) - noise_amplitude

## Чистое решение о подмене. Бросок кубика передаётся параметром, поэтому самая
## коварная механика игры проверяется точными assert'ами без всякого RNG.
static func should_switch(conf_a: float, conf_b: float, roll: float) -> bool:
	if conf_a >= 0.60 or conf_b >= 0.60:
		return false
	if conf_a < PN.CONF_DRIFTING or conf_b < PN.CONF_DRIFTING:
		return false
	return roll < 0.25 * (1.0 - minf(conf_a, conf_b))

func track_for(object_id: int) -> Track:
	for tr in _by_object.values():
		if tr.object_id == object_id:
			return tr
	return null

func track_by_id(track_id: int) -> Track:
	for tr in _by_object.values():
		if tr.track_id == track_id:
			return tr
	return null

## Меняет местами привязку двух треков, сохраняя запомненные признаки.
func force_switch(object_a: int, object_b: int) -> void:
	var ta := track_for(object_a)
	var tb := track_for(object_b)
	if ta == null or tb == null or ta == tb:
		return
	var tmp := ta.object_id
	ta.object_id = tb.object_id
	tb.object_id = tmp
	ta.switched = not ta.switched
	tb.switched = not tb.switched
	_by_object[ta.object_id] = ta
	_by_object[tb.object_id] = tb

## Пересопоставляет треки с объектами заново и снимает все подмены.
## Номера при этом меняются — как у настоящего трекера после потери.
func recapture(view: SceneView) -> void:
	_by_object.clear()
	for o in view.objects:
		var tr := Track.new(_next_track_id, o.kind, o.object_id)
		tr.remembered_cloak = o.cloak
		_next_track_id += 1
		_by_object[o.object_id] = tr

## Прогоняет подмены между соседями по очереди. Инвариант «только соседи»
## держится структурой цикла, а не проверкой внутри.
func roll_switches(view: SceneView, rng: RandomNumberGenerator) -> void:
	for slot in range(PN.QUEUE_SLOTS - 1):
		var a := _object_in_slot(view, slot)
		var b := _object_in_slot(view, slot + 1)
		if a < 0 or b < 0:
			continue
		var ta := track_for(a)
		var tb := track_for(b)
		if ta == null or tb == null:
			continue
		if should_switch(ta.conf, tb.conf, rng.randf()):
			force_switch(a, b)

	_roll_passer_switches(view, rng)

## Подмена покупателя ПРОХОЖИМ. При обслуживании по одному соседей в очереди
## нет и путать не с кем — а настоящий трекер ошибается именно в потоке
## посторонних. Прохожий, подошедший к прилавку, и есть такой поток.
func _roll_passer_switches(view: SceneView, rng: RandomNumberGenerator) -> void:
	for guest in view.objects:
		if guest.kind != &"knight":
			continue
		var tg := track_for(guest.object_id)
		if tg == null:
			continue

		for other in view.objects:
			if other.kind != &"passer":
				continue
			var tp := track_for(other.object_id)
			if tp == null:
				continue
			if should_switch(tg.conf, tp.conf, rng.randf()):
				force_switch(guest.object_id, other.object_id)
				return   # одной подмены за ход достаточно

func _object_in_slot(view: SceneView, slot: int) -> int:
	for o in view.objects:
		if o.kind == &"knight" and o.slot == slot:
			return o.object_id
	return -1
