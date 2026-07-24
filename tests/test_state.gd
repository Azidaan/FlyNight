extends RefCounted

func run(t: AssertBag) -> void:
	var k := Knight.new(7, PN.Potion.HEALING, PN.Cloak.RED, 8)
	t.eq(k.id, 7, "рыцарь помнит свой номер")
	t.eq(k.patience, 8, "терпение задано")
	t.eq(k.slot, -1, "новый рыцарь ещё не в очереди")

	var s := GameState.new(12345)
	t.eq(s.turn, 0, "ночь начинается с нулевого хода")
	t.eq(s.coins, 0, "монет нет")
	t.eq(s.misses, 0, "промахов нет")
	t.eq(s.counter.size(), 0, "прилавок пуст")
	t.eq(s.chest[PN.Potion.HEALING], PN.CHEST_PER_TYPE, "сундук полон")
	t.eq(s.wolf_distance, -1, "волка нет")
	t.eq(s.lantern_turns, 0, "фонарь погашен")
	t.check(not s.lantern_on(), "фонарь не считается горящим")
	t.check(not s.finished, "ночь не закончена")
	t.eq(s.coins_in_flight.size(), 0, "монет в полёте нет")

	t.eq(s.free_queue_slot(), 0, "первый слот очереди свободен")
	var k1 := Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 8)
	k1.slot = 0
	s.knights.append(k1)
	t.eq(s.free_queue_slot(), 1, "занятый слот пропускается")

	t.eq(s.knight_in_slot(0).id, 1, "рыцарь находится по слоту")
	t.eq(s.knight_in_slot(2), null, "пустой слот даёт null")
	t.eq(s.knight_by_id(1).id, 1, "рыцарь находится по номеру")
	t.eq(s.knight_by_id(99), null, "несуществующий номер даёт null")

	# очередь заполняется до отказа
	for i in range(2):
		var extra := Knight.new(10 + i, PN.Potion.HEALING, PN.Cloak.BLUE, 8)
		extra.slot = i + 1
		s.knights.append(extra)
	t.eq(s.free_queue_slot(), -1, "полная очередь не даёт свободных слотов")

	# доступные типы
	var a := GameState.new(2)
	t.eq(a.available_potion_types().size(), 3, "в начале доступны все три типа")
	a.chest[PN.Potion.STRENGTH] = 0
	t.eq(a.available_potion_types().size(), 2, "пустой в сундуке тип выпадает")
	a.counter.append(PN.Potion.STRENGTH)
	t.eq(a.available_potion_types().size(), 3, "но лежащий на прилавке снова доступен")
	a.chest[PN.Potion.HEALING] = 0
	a.chest[PN.Potion.COURAGE] = 0
	a.counter.clear()
	t.eq(a.available_potion_types().size(), 0, "когда всё кончилось — пусто")

	var s2 := GameState.new(12345)
	t.almost(s2.weather.rain_offset, s.weather.rain_offset, "одинаковый сид — одинаковая погода")
	var s3 := GameState.new(54321)
	t.check(absf(s3.weather.rain_offset - s.weather.rain_offset) > 0.0, "разный сид — разная погода")
