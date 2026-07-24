extends RefCounted

func run(t: AssertBag) -> void:
	var tr := Track.new(1, &"knight", 7)
	t.eq(tr.track_id, 1, "трек помнит номер")
	t.eq(tr.kind, &"knight", "трек помнит класс")
	t.eq(tr.object_id, 7, "трек привязан к объекту")
	t.check(not tr.switched, "новый трек не подменён")
	t.eq(tr.remembered_cloak, -1, "признак ещё не зафиксирован")
	t.eq(tr.age, 0, "новый трек только что родился")

	tr.conf = 0.9
	tr.refresh_state()
	t.eq(tr.state, PN.TrackState.LOCKED, "0.9 это захват")
	tr.conf = 0.5
	tr.refresh_state()
	t.eq(tr.state, PN.TrackState.DRIFTING, "0.5 это дрейф")
	tr.conf = 0.2
	tr.refresh_state()
	t.eq(tr.state, PN.TrackState.LOST, "0.2 это потеря")

	var o := SceneObject.new(7, &"knight", 0.5, 0.0)
	o.cloak = PN.Cloak.BLUE
	t.eq(o.object_id, 7, "объект помнит номер")
	t.almost(o.dist_norm, 0.5, "дистанция задана")

	var v := SceneView.new(0.3, 0.8, false)
	v.objects.append(o)
	t.eq(v.objects.size(), 1, "сцена содержит объект")
	t.almost(v.rain, 0.3, "сцена помнит дождь")
	t.almost(v.dark, 0.8, "сцена помнит темень")
	t.check(not v.lantern_on, "фонарь погашен")

	var src := TrackSource.new()
	t.eq(src.update(v).size(), 0, "базовый источник ничего не выдаёт")

	# сборка сцены из состояния мира
	var s := GameState.new(400)
	for i in range(3):
		var k := Knight.new(i + 1, PN.Potion.HEALING, PN.Cloak.RED, 8)
		k.slot = i
		s.knights.append(k)
	var built := SceneView.from_state(s)
	t.eq(built.objects.size(), 3, "три рыцаря попали в сцену")

	var dists := {}
	for obj in built.objects:
		dists[obj.slot] = obj.dist_norm
	t.almost(dists[0], 0.0, "слот у прилавка — дистанция ноль")
	t.almost(dists[1], 0.5, "средний слот — половина")
	t.almost(dists[2], 1.0, "хвост очереди — единица")

	# рыцарь вне очереди в сцену не попадает
	var off := Knight.new(99, PN.Potion.HEALING, PN.Cloak.RED, 8)
	s.knights.append(off)
	t.eq(SceneView.from_state(s).objects.size(), 3, "рыцарь без слота невидим")

	# волк и монета
	s.wolf_distance = 4
	s.coins_in_flight.append(101)
	var full := SceneView.from_state(s)
	t.eq(full.objects.size(), 5, "волк и монета добавились")
	for obj in full.objects:
		if obj.kind == &"wolf":
			t.almost(obj.occlusion, 0.35, "волк перекрыт кромкой леса")
		elif obj.kind == &"coin":
			t.almost(obj.occlusion, 0.30, "монета перекрыта прилавком")
			t.almost(obj.dist_norm, 0.0, "монета падает у прилавка")
