extends RefCounted

func _has(events: Array, kind: StringName) -> bool:
	for e in events:
		if e.kind == kind:
			return true
	return false

func _tracker(s: int = 77) -> FakeTracker:
	var ft := FakeTracker.new(s)
	ft.noise_amplitude = 0.0
	return ft

func run(t: AssertBag) -> void:
	# ── монеты в полёте ───────────────────────────────────────
	var cs := GameState.new(21)
	cs.coins_in_flight.append(101)
	var v := SceneView.from_state(cs)
	var coin_found := false
	for o in v.objects:
		if o.kind == &"coin":
			coin_found = true
			t.eq(o.object_id, 101, "монета попала в сцену со своим номером")
	t.check(coin_found, "монета видна трекеру")

	var clear := GameState.new(22)
	clear.coins_in_flight.append(7)
	var tracks := _tracker().update(SceneView.from_state(clear))
	var r0 := ActionResult.new()
	Turn.resolve_coins(clear, tracks, r0)
	t.eq(clear.coins, 1, "монета зачтена в ясную погоду")
	t.eq(clear.coins_in_flight.size(), 0, "список очищен")
	t.check(_has(r0.events, &"COIN_COLLECTED"), "событие подбора")

	var murk := GameState.new(23)
	murk.coins_in_flight.append(8)
	var lostc := Track.new(1, &"coin", 8)
	lostc.conf = 0.1
	lostc.refresh_state()
	var dead: Array[Track] = [lostc]
	var r1 := ActionResult.new()
	Turn.resolve_coins(murk, dead, r1)
	t.eq(murk.coins, 0, "потерянный трек — монеты нет")
	t.eq(murk.coins_in_flight.size(), 0, "но список всё равно очищен")
	t.check(_has(r1.events, &"COIN_LOST"), "событие потери")

	# ── терпение и уход ───────────────────────────────────────
	var s := GameState.new(31)
	# номер заведомо далёкий: автоматически приходящие рыцари нумеруются
	# с next_knight_id, и совпадение номеров спутало бы проверку
	var k := Knight.new(900, PN.Potion.HEALING, PN.Cloak.RED, 2)
	k.slot = 0
	s.knights.append(k)

	Turn.world_turn(s, _tracker())
	t.eq(s.turn, 1, "номер хода вырос")
	t.eq(s.knight_by_id(900).patience, 1, "терпение упало")

	var r := Turn.world_turn(s, _tracker())
	t.eq(s.knight_by_id(900), null, "терпение кончилось, рыцарь ушёл")
	t.eq(s.misses, 1, "засчитан промах")
	t.check(_has(r.events, &"KNIGHT_LEAVES_ANGRY"), "событие ухода")

	# ── фонарь догорает ───────────────────────────────────────
	var s2 := GameState.new(32)
	s2.lantern_turns = 1
	var r2 := Turn.world_turn(s2, _tracker())
	t.eq(s2.lantern_turns, 0, "фонарь погас")
	t.check(_has(r2.events, &"LANTERN_DIED"), "событие погасания")

	# ── волк ──────────────────────────────────────────────────
	var s3 := GameState.new(33)
	s3.wolf_distance = 4
	Turn.world_turn(s3, _tracker())
	t.eq(s3.wolf_distance, 3, "волк сделал шаг")
	s3.lantern_turns = 2
	Turn.world_turn(s3, _tracker())
	t.eq(s3.wolf_distance, 1, "при фонаре волк шагает вдвое")

	var s4 := GameState.new(34)
	s4.wolf_distance = 1
	s4.counter.append(PN.Potion.HEALING)
	var r4 := Turn.world_turn(s4, _tracker())
	t.eq(s4.counter.size(), 0, "зелье украдено")
	t.eq(s4.wolf_distance, -1, "волк убежал с добычей")
	t.check(_has(r4.events, &"WOLF_STEALS"), "событие кражи")

	# волк доходит до пустого прилавка и всё равно убегает
	var s4b := GameState.new(340)
	s4b.wolf_distance = 1
	var r4b := Turn.world_turn(s4b, _tracker())
	t.eq(s4b.wolf_distance, -1, "с пустого прилавка волк тоже уходит")
	t.check(_has(r4b.events, &"WOLF_STEALS"), "событие всё равно есть")

	# ── провал по трём промахам ───────────────────────────────
	var s5 := GameState.new(35)
	s5.misses = 2
	var k5 := Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 1)
	k5.slot = 0
	s5.knights.append(k5)
	var r5 := Turn.world_turn(s5, _tracker())
	t.eq(s5.misses, 3, "третий промах")
	t.check(s5.finished, "ночь закончена")
	t.eq(s5.outcome, "failed", "исход — провал")
	t.check(_has(r5.events, &"NIGHT_FAILED"), "событие провала")

	# ── рассвет ───────────────────────────────────────────────
	var s6 := GameState.new(36)
	s6.turn = PN.NIGHT_TURNS - 1
	var r6 := Turn.world_turn(s6, _tracker())
	t.check(s6.finished, "ночь закончена по времени")
	t.eq(s6.outcome, "dawn", "исход — рассвет")
	t.check(_has(r6.events, &"DAWN"), "событие рассвета")

	var before := s6.turn
	Turn.world_turn(s6, _tracker())
	t.eq(s6.turn, before, "после конца ход не идёт")

	# ── очередь подтягивается ─────────────────────────────────
	var s7 := GameState.new(37)
	var far := Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 9)
	far.slot = 2
	s7.knights.append(far)
	Turn.world_turn(s7, _tracker())
	t.eq(s7.knight_by_id(1).slot, 0, "рыцарь подошёл к прилавку")

	# ── заказы только из доступного ───────────────────────────
	var s8 := GameState.new(38)
	s8.chest[PN.Potion.STRENGTH] = 0
	s8.chest[PN.Potion.COURAGE] = 0
	for i in range(20):
		Turn.world_turn(s8, _tracker())
	var all_healing := true
	for kk in s8.knights:
		if kk.order != PN.Potion.HEALING:
			all_healing = false
	t.check(all_healing, "заказ только из того, что физически есть")

	# ── детерминизм ───────────────────────────────────────────
	var a := GameState.new(4242)
	var b := GameState.new(4242)
	var fa := _tracker(1)
	var fb := _tracker(1)
	for i in range(15):
		Turn.world_turn(a, fa)
		Turn.world_turn(b, fb)
	t.eq(a.knights.size(), b.knights.size(), "одинаковый сид — одинаковая ночь")
	t.eq(a.wolf_distance, b.wolf_distance, "волк ведёт себя одинаково")
	t.eq(a.misses, b.misses, "промахи совпадают")
	t.eq(a.next_knight_id, b.next_knight_id, "рыцари приходили в том же порядке")
