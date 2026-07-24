class_name Turn
extends RefCounted

## Ход мира: то, что происходит после каждой разыгранной карты.

const ARRIVAL_CHANCE := 0.45
const WOLF_BASE_CHANCE := 0.10
const WOLF_DARK_FACTOR := 0.25
const WOLF_START_DISTANCE := 6

## Монета живёт один ход. Держится трек — подобрана, потерян — укатилась.
## Так трекинг влияет на доход напрямую, без отдельного действия.
static func resolve_coins(s: GameState, tracks: Array[Track], r: ActionResult) -> void:
	for coin_id in s.coins_in_flight:
		var found: Track = null
		for tr in tracks:
			if tr.kind == &"coin" and tr.object_id == coin_id:
				found = tr
				break
		if found != null and found.state != PN.TrackState.LOST:
			s.coins += 1
			r.emit(&"COIN_COLLECTED", {"coin": coin_id})
		else:
			r.emit(&"COIN_LOST", {"coin": coin_id})
	s.coins_in_flight.clear()

static func world_turn(s: GameState, tracker: FakeTracker) -> ActionResult:
	var r := ActionResult.new()
	if s.finished:
		return r

	# Волк, сбежавший этим же ходом, не может тут же вернуться: иначе игрок
	# увидел бы WOLF_FLEES и следом WOLF_APPEARS в одном ходу.
	var wolf_was_absent := s.wolf_distance < 0

	resolve_coins(s, tracker.update(SceneView.from_state(s)), r)
	_tick_patience(s, r)
	_tick_lantern(s, r)
	_step_wolf(s, r)
	_advance_queue(s, r)
	_maybe_arrive(s, r)
	if wolf_was_absent:
		_maybe_wolf_appears(s, r)

	s.turn += 1

	var view := SceneView.from_state(s)
	tracker.update(view)
	tracker.roll_switches(view, s.rng)

	_check_end(s, r)
	return r

static func _tick_patience(s: GameState, r: ActionResult) -> void:
	# обход по копии: удаление из массива во время итерации пропускает элементы
	for k in s.knights.duplicate():
		k.patience -= 1
		if k.patience <= 0:
			s.knights.erase(k)
			s.misses += 1
			r.emit(&"KNIGHT_LEAVES_ANGRY", {"knight": k.id, "slot": k.slot})

static func _tick_lantern(s: GameState, r: ActionResult) -> void:
	if s.lantern_turns > 0:
		s.lantern_turns -= 1
		if s.lantern_turns == 0:
			r.emit(&"LANTERN_DIED", {})

static func _step_wolf(s: GameState, r: ActionResult) -> void:
	if s.wolf_distance < 0:
		return
	s.wolf_distance -= 2 if s.lantern_on() else 1
	if s.wolf_distance > 0:
		r.emit(&"WOLF_STEPS", {"distance": s.wolf_distance})
		return

	if not s.counter.is_empty():
		var idx := s.rng.randi_range(0, s.counter.size() - 1)
		var stolen: int = s.counter[idx]
		s.counter.remove_at(idx)
		r.emit(&"WOLF_STEALS", {"potion": stolen})
	else:
		r.emit(&"WOLF_STEALS", {"potion": -1})
	s.wolf_distance = -1
	r.emit(&"WOLF_FLEES", {})

static func _advance_queue(s: GameState, r: ActionResult) -> void:
	var occupied := s.knights.duplicate()
	occupied.sort_custom(func(a, b): return a.slot < b.slot)
	for i in range(occupied.size()):
		if occupied[i].slot != i:
			occupied[i].slot = i
			r.emit(&"KNIGHT_ADVANCES", {"knight": occupied[i].id, "slot": i})

static func _maybe_arrive(s: GameState, r: ActionResult) -> void:
	var slot := s.free_queue_slot()
	if slot < 0:
		return
	var kinds := s.available_potion_types()
	if kinds.is_empty():
		return
	if s.rng.randf() >= ARRIVAL_CHANCE:
		return

	var k := Knight.new(
		s.next_knight_id,
		kinds[s.rng.randi_range(0, kinds.size() - 1)],
		s.rng.randi_range(0, PN.Cloak.size() - 1),
		s.rng.randi_range(6, 9))
	k.slot = slot
	s.next_knight_id += 1
	s.knights.append(k)
	r.emit(&"KNIGHT_ARRIVES", {"knight": k.id, "slot": slot, "cloak": k.cloak})

static func _maybe_wolf_appears(s: GameState, r: ActionResult) -> void:
	if s.wolf_distance >= 0:
		return
	if s.rng.randf() < WOLF_BASE_CHANCE + WOLF_DARK_FACTOR * s.dark():
		s.wolf_distance = WOLF_START_DISTANCE
		r.emit(&"WOLF_APPEARS", {"distance": WOLF_START_DISTANCE})

static func _check_end(s: GameState, r: ActionResult) -> void:
	if s.misses >= PN.MAX_MISSES:
		s.finished = true
		s.outcome = "failed"
		r.emit(&"NIGHT_FAILED", {"coins": s.coins})
	elif s.turn >= PN.NIGHT_TURNS:
		s.finished = true
		s.outcome = "dawn"
		r.emit(&"DAWN", {"coins": s.coins})
