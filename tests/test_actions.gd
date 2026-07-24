extends RefCounted

func _has(events: Array, kind: StringName) -> bool:
	for e in events:
		if e.kind == kind:
			return true
	return false

func _tracked(g: GameState) -> Array:
	var ft := FakeTracker.new(5)
	ft.noise_amplitude = 0.0
	return [ft, ft.update(SceneView.from_state(g))]

func run(t: AssertBag) -> void:
	# ── события и результат ───────────────────────────────────
	var e := GameEvent.new(&"POTION_PLACED", {"potion": PN.Potion.HEALING})
	t.eq(e.kind, &"POTION_PLACED", "событие помнит вид")
	t.eq(e.data["potion"], PN.Potion.HEALING, "событие несёт данные")

	var bad := ActionResult.fail("нет места")
	t.check(not bad.ok, "провальный результат помечен")
	t.eq(bad.reason, "нет места", "причина сохранена")

	# ── пополнение ────────────────────────────────────────────
	var s := GameState.new(1)
	var r := Actions.restock(s, PN.Potion.HEALING)
	t.check(r.ok, "пополнение с полным сундуком проходит")
	t.eq(s.counter.size(), 1, "зелье легло на прилавок")
	t.eq(s.chest[PN.Potion.HEALING], PN.CHEST_PER_TYPE - 1, "сундук уменьшился")
	t.eq(r.events.size(), 1, "породилось одно событие")
	t.eq(r.events[0].kind, &"POTION_PLACED", "событие про выкладку")

	for i in range(3):
		Actions.restock(s, PN.Potion.HEALING)
	t.eq(s.counter.size(), PN.COUNTER_SLOTS, "прилавок заполнен")
	var full := Actions.restock(s, PN.Potion.HEALING)
	t.check(not full.ok, "на полный прилавок не положить")
	t.eq(s.counter.size(), PN.COUNTER_SLOTS, "состояние не изменилось при отказе")
	t.eq(s.chest[PN.Potion.HEALING], PN.CHEST_PER_TYPE - 4, "и сундук тоже не тронут")

	var s2 := GameState.new(2)
	s2.chest[PN.Potion.STRENGTH] = 0
	t.check(not Actions.restock(s2, PN.Potion.STRENGTH).ok, "из пустого сундука не достать")
	t.eq(s2.counter.size(), 0, "прилавок не тронут")

	# ── выдача: верное зелье верному рыцарю ───────────────────
	var g := GameState.new(3)
	var k1 := Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 8)
	k1.slot = 0
	var k2 := Knight.new(2, PN.Potion.STRENGTH, PN.Cloak.BLUE, 8)
	k2.slot = 1
	g.knights.append_array([k1, k2])
	g.counter.append_array([PN.Potion.HEALING, PN.Potion.STRENGTH])

	var pack := _tracked(g)
	var ft: FakeTracker = pack[0]
	var tracks: Array[Track] = pack[1]
	var t1: int = ft.track_for(1).track_id

	var good := Actions.give(g, tracks, t1, 0)
	t.check(good.ok, "выдача проходит")
	t.eq(g.knights.size(), 1, "довольный рыцарь ушёл")
	t.eq(g.counter.size(), 1, "зелье списано с прилавка")
	t.check(_has(good.events, &"COIN_THROWN"), "довольный бросает монету")
	t.eq(g.coins_in_flight.size(), 1, "монета в полёте")
	t.eq(g.coins, 0, "но в кошелёк ещё не легла")

	# ── выдача: не то зелье ───────────────────────────────────
	var g2 := GameState.new(4)
	var k3 := Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 8)
	k3.slot = 0
	g2.knights.append(k3)
	g2.counter.append(PN.Potion.COURAGE)
	var p2 := _tracked(g2)
	var tr2: Array[Track] = p2[1]
	var wrong := Actions.give(g2, tr2, tr2[0].track_id, 0)
	t.check(wrong.ok, "выдача не того зелья всё равно тратит ход")
	t.eq(g2.knights[0].patience, 5, "терпение упало на три")
	t.eq(g2.counter.size(), 0, "зелье потрачено впустую")
	t.check(_has(wrong.events, &"POTION_REFUSED"), "породилось событие отказа")
	t.eq(g2.coins_in_flight.size(), 0, "монеты нет")

	# ── подмена уводит зелье не тому ──────────────────────────
	var g3 := GameState.new(5)
	var a := Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 8)
	a.slot = 0
	var b := Knight.new(2, PN.Potion.STRENGTH, PN.Cloak.BLUE, 8)
	b.slot = 1
	g3.knights.append_array([a, b])
	g3.counter.append(PN.Potion.HEALING)
	var p3 := _tracked(g3)
	var ft3: FakeTracker = p3[0]
	var tr3: Array[Track] = p3[1]
	var id_a: int = ft3.track_for(1).track_id

	ft3.force_switch(1, 2)
	var swapped := Actions.give(g3, tr3, id_a, 0)
	t.check(swapped.ok, "выдача по подменённому треку проходит")
	t.eq(g3.knight_by_id(1).patience, 8, "первый рыцарь ничего не получил")
	t.eq(g3.knight_by_id(2).patience, 5, "зелье ушло второму и не подошло ему")

	t.check(not Actions.give(g3, tr3, 9999, 0).ok, "по чужому номеру трека выдать нельзя")
	t.check(not Actions.give(g3, tr3, id_a, 77).ok, "с несуществующего слота прилавка тоже")

	# ── фонарь ────────────────────────────────────────────────
	var g4 := GameState.new(6)
	var lr := Actions.lantern(g4)
	t.check(lr.ok, "фонарь зажигается")
	t.eq(g4.lantern_turns, Actions.LANTERN_TURNS, "горит три хода")
	t.check(g4.lantern_on(), "фонарь считается горящим")
	t.check(_has(lr.events, &"LANTERN_LIT"), "событие поджига")
	t.check(not Actions.lantern(g4).ok, "горящий фонарь не поджечь снова")

	# ── камень ────────────────────────────────────────────────
	var g5 := GameState.new(7)
	var none: Array[Track] = []
	t.check(not Actions.rock(g5, none).ok, "без волка камень не бросить")

	g5.wolf_distance = 3
	var wt := Track.new(50, &"wolf", -1)
	wt.conf = 0.5
	wt.refresh_state()
	var live: Array[Track] = [wt]
	var hit := Actions.rock(g5, live)
	t.check(hit.ok, "по дрейфующему треку попадание засчитано")
	t.eq(g5.wolf_distance, -1, "волк убежал")
	t.check(_has(hit.events, &"WOLF_FLEES"), "событие бегства")

	var g6 := GameState.new(8)
	g6.wolf_distance = 2
	var lostw := Track.new(51, &"wolf", -1)
	lostw.conf = 0.1
	lostw.refresh_state()
	var dead: Array[Track] = [lostw]
	var miss := Actions.rock(g6, dead)
	t.check(miss.ok, "промах — тоже потраченный ход, а не отказ")
	t.eq(g6.wolf_distance, 2, "волк на месте")
	t.check(_has(miss.events, &"ROCK_MISSED"), "событие промаха")

	# ── перезахват ────────────────────────────────────────────
	var g7 := GameState.new(9)
	var q1 := Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 8)
	q1.slot = 0
	var q2 := Knight.new(2, PN.Potion.STRENGTH, PN.Cloak.BLUE, 8)
	q2.slot = 1
	g7.knights.append_array([q1, q2])
	var p7 := _tracked(g7)
	var ft7: FakeTracker = p7[0]
	ft7.force_switch(1, 2)
	t.check(ft7.track_for(1).switched, "подмена сделана")

	var rec := Actions.recapture(g7, ft7)
	t.check(rec.ok, "перезахват проходит")
	t.check(not ft7.track_for(1).switched, "подмены сняты")
	t.eq(ft7.track_for(1).remembered_cloak, PN.Cloak.RED, "признак снова верен")
	t.check(_has(rec.events, &"TRACKER_RESET"), "событие сброса")
