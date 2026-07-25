class_name Actions
extends RefCounted

## Пять карт в руке игрока. Каждая либо тратит ход и меняет состояние,
## либо отказывает и не трогает ничего.

const LANTERN_TURNS := 3
const WRONG_POTION_PENALTY := 3

static func restock(s: GameState, potion: int) -> ActionResult:
	if s.counter.size() >= PN.COUNTER_SLOTS:
		return ActionResult.fail("прилавок полон")
	if s.chest.get(potion, 0) <= 0:
		return ActionResult.fail("в сундуке пусто")

	s.chest[potion] -= 1
	s.counter.append(potion)

	var r := ActionResult.new()
	r.emit(&"POTION_PLACED", {"potion": potion, "slot": s.counter.size() - 1})
	return r

## Цель выбирается ПО НОМЕРУ ТРЕКА. Здесь подмена ID впервые кусает игрока:
## трек ведёт к object_id, а не к тому, кого игрок видел под этой рамкой.
static func give(s: GameState, tracks: Array[Track], track_id: int, counter_index: int) -> ActionResult:
	if counter_index < 0 or counter_index >= s.counter.size():
		return ActionResult.fail("на прилавке нет такого зелья")

	var target: Track = null
	for tr in tracks:
		if tr.track_id == track_id:
			target = tr
			break
	if target == null or target.kind != &"knight":
		return ActionResult.fail("нет такого трека")

	var knight := s.knight_by_id(target.object_id)
	if knight == null:
		return ActionResult.fail("трек ведёт в пустоту")

	var potion: int = s.counter[counter_index]
	s.counter.remove_at(counter_index)

	var r := ActionResult.new()
	if potion == knight.order:
		s.knights.erase(knight)
		r.emit(&"POTION_HANDED", {"knight": knight.id, "potion": potion})
		r.emit(&"KNIGHT_LEAVES_HAPPY", {"knight": knight.id, "slot": knight.slot})
		# Монета не начисляется здесь: она становится объектом со своим треком
		# и подбирается ходом мира. Иначе трекинг не влиял бы на доход.
		var coin_id := s.next_coin_id
		s.next_coin_id += 1
		s.coins_in_flight.append(coin_id)
		r.emit(&"COIN_THROWN", {"knight": knight.id, "coin": coin_id})
	else:
		# Зелье потрачено впустую, но рыцарь никуда не уходит: времени
		# на обслуживание не отведено, он дождётся нужного.
		s.misses += 1
		r.emit(&"POTION_REFUSED", {"knight": knight.id, "potion": potion})
	return r

static func lantern(s: GameState) -> ActionResult:
	if s.lantern_on():
		return ActionResult.fail("фонарь уже горит")
	s.lantern_turns = LANTERN_TURNS
	var r := ActionResult.new()
	r.emit(&"LANTERN_LIT", {"turns": LANTERN_TURNS})
	return r

static func rock(s: GameState, tracks: Array[Track]) -> ActionResult:
	if s.wolf_distance < 0:
		return ActionResult.fail("волка не видно")

	var wolf: Track = null
	for tr in tracks:
		if tr.kind == &"wolf":
			wolf = tr
			break

	var r := ActionResult.new()
	r.emit(&"ROCK_THROWN", {})
	if wolf == null or wolf.state == PN.TrackState.LOST:
		r.emit(&"ROCK_MISSED", {})
		return r

	s.wolf_distance = -1
	r.emit(&"WOLF_FLEES", {})
	return r

static func recapture(s: GameState, tracker: FakeTracker) -> ActionResult:
	tracker.recapture(SceneView.from_state(s))
	var r := ActionResult.new()
	r.emit(&"TRACKER_RESET", {})
	return r
