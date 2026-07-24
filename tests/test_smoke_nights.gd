extends RefCounted

## Смоук: ночь всегда заканчивается, бездействие проигрывает, игра проходима.
## Стоит до играбельного среза — зависший цикл и мёртвый баланс ловятся здесь
## дешевле, чем это станет видно глазами.

func _play_idle(seed_value: int) -> GameState:
	var g := GameState.new(1000 + seed_value)
	var ft := FakeTracker.new(2000 + seed_value)
	var guard := 0
	while not g.finished and guard < PN.NIGHT_TURNS + 5:
		Turn.world_turn(g, ft)
		guard += 1
	return g

## Бот играет честно: смотрит ТОЛЬКО на то, что показывает трекер
## (remembered_order), и не подглядывает в настоящие заказы рыцарей.
func _play_bot(seed_value: int) -> Dictionary:
	var g := GameState.new(3000 + seed_value)
	var ft := FakeTracker.new(4000 + seed_value)
	var refused := 0
	var guard := 0

	while not g.finished and guard < PN.NIGHT_TURNS + 5:
		var tracks := ft.update(SceneView.from_state(g))
		var acted := false

		for tr in tracks:
			if tr.kind != &"knight" or tr.state == PN.TrackState.LOST:
				continue
			var idx := g.counter.find(tr.remembered_order)
			if idx >= 0:
				var res := Actions.give(g, tracks, tr.track_id, idx)
				if res.ok:
					for e in res.events:
						if e.kind == &"POTION_REFUSED":
							refused += 1
					acted = true
					break

		if not acted:
			for tr in tracks:
				if tr.kind == &"knight" and tr.remembered_order >= 0:
					if Actions.restock(g, tr.remembered_order).ok:
						acted = true
						break

		if not acted:
			if not Actions.restock(g, PN.Potion.HEALING).ok:
				Actions.lantern(g)

		Turn.world_turn(g, ft)
		guard += 1

	return {"state": g, "refused": refused}

func run(t: AssertBag) -> void:
	# ── бездействие ───────────────────────────────────────────
	var stuck := 0
	var idle_dawns := 0
	var idle_fails := 0

	for s in range(50):
		var g := _play_idle(s)
		if not g.finished:
			stuck += 1
		elif g.outcome == "dawn":
			idle_dawns += 1
		else:
			idle_fails += 1

	t.eq(stuck, 0, "ни одна ночь не зависла")
	t.eq(idle_dawns + idle_fails, 50, "у каждой ночи есть исход")
	t.eq(idle_dawns, 0, "ничего не делая, ночь пережить нельзя")

	# ── бот играет ────────────────────────────────────────────
	var bot_dawns := 0
	var bot_fails := 0
	var bot_stuck := 0
	var total_refused := 0
	var total_coins := 0

	for s in range(50):
		var res := _play_bot(s)
		var g: GameState = res["state"]
		total_refused += res["refused"]
		total_coins += g.coins
		if not g.finished:
			bot_stuck += 1
		elif g.outcome == "dawn":
			bot_dawns += 1
		else:
			bot_fails += 1

	t.eq(bot_stuck, 0, "ни одна ночь с ботом не зависла")
	t.eq(bot_dawns + bot_fails, 50, "у каждой ночи с ботом есть исход")
	t.check(bot_dawns > 0, "игру можно пройти: хотя бы одна ночь дожила до рассвета")
	t.check(total_coins > 0, "бот заработал хоть сколько-то монет")

	# Отказы возникают, только когда трекер соврал: бот всегда отдаёт ровно то,
	# что написано в панели. Если отказов ноль — подмена ID ни на что не влияет.
	t.check(total_refused > 0, "подмена ID реально приводит к неверной выдаче")

	print("  смоук: бездействие %d провалов; бот %d рассветов / %d провалов, %d монет, %d ошибочных выдач"
		% [idle_fails, bot_dawns, bot_fails, total_coins, total_refused])
