extends RefCounted

func _clear_view() -> SceneView:
	var v := SceneView.new(0.0, 0.0, false)
	var o := SceneObject.new(1, &"knight", 0.0, 0.0)
	o.cloak = PN.Cloak.RED
	o.slot = 0
	v.objects.append(o)
	return v

func _tracker() -> FakeTracker:
	var ft := FakeTracker.new(999)
	ft.noise_amplitude = 0.0
	return ft

func run(t: AssertBag) -> void:
	# ── уверенность ───────────────────────────────────────────
	var tracks := _tracker().update(_clear_view())
	t.eq(tracks.size(), 1, "один объект — один трек")
	t.almost(tracks[0].conf, 1.0, "в ясную погоду уверенность полная")
	t.eq(tracks[0].state, PN.TrackState.LOCKED, "ясная погода — захват")
	t.eq(tracks[0].remembered_cloak, PN.Cloak.RED, "признак зафиксирован при захвате")

	# худшие условия: 1.0 - 0.30 - 0.35 - 0.20 = 0.15
	var worst := SceneView.new(1.0, 1.0, false)
	var wo := SceneObject.new(1, &"knight", 1.0, 0.0)
	wo.cloak = PN.Cloak.RED
	wo.slot = 2
	worst.objects.append(wo)
	var bad := _tracker().update(worst)
	t.almost(bad[0].conf, 0.15, "худшие условия роняют уверенность до 0.15")
	t.eq(bad[0].state, PN.TrackState.LOST, "худшие условия — потеря")

	# фонарь: 1.0 - 0.35*1*0.25 = 0.9125
	var lit := SceneView.new(0.0, 1.0, true)
	var lo := SceneObject.new(1, &"knight", 0.0, 0.0)
	lo.cloak = PN.Cloak.RED
	lo.slot = 0
	lit.objects.append(lo)
	t.almost(_tracker().update(lit)[0].conf, 0.9125, "фонарь гасит темноту вчетверо")

	# без фонаря то же самое: 1.0 - 0.35 = 0.65
	var unlit := SceneView.new(0.0, 1.0, false)
	var uo := SceneObject.new(1, &"knight", 0.0, 0.0)
	uo.cloak = PN.Cloak.RED
	uo.slot = 0
	unlit.objects.append(uo)
	t.almost(_tracker().update(unlit)[0].conf, 0.65, "без фонаря темнота бьёт вчетверо сильнее")

	var a := FakeTracker.new(42).update(_clear_view())
	var b := FakeTracker.new(42).update(_clear_view())
	t.almost(a[0].conf, b[0].conf, "одинаковый сид воспроизводим")

	var ids := _tracker()
	var first := ids.update(_clear_view())
	var second := ids.update(_clear_view())
	t.eq(first[0].track_id, second[0].track_id, "трек сохраняет номер между ходами")
	t.eq(second[0].age, 1, "возраст трека растёт")

	# исчезнувший объект теряет трек
	var gone := ids.update(SceneView.new(0.0, 0.0, false))
	t.eq(gone.size(), 0, "объект пропал — трека нет")
	t.eq(ids.track_for(1), null, "и в таблице его тоже нет")

	# ── решение о подмене: чистая функция, без RNG ────────────
	t.check(not FakeTracker.should_switch(0.8, 0.5, 0.0), "уверенный трек не подменяется")
	t.check(not FakeTracker.should_switch(0.5, 0.7, 0.0), "хватает одного уверенного")
	t.check(not FakeTracker.should_switch(0.2, 0.5, 0.0), "потерянный трек не подменяется")
	t.check(not FakeTracker.should_switch(0.5, 0.2, 0.0), "потерянный второй тоже")
	t.check(FakeTracker.should_switch(0.4, 0.5, 0.0), "нулевой бросок всегда проходит")
	t.check(not FakeTracker.should_switch(0.4, 0.5, 0.99), "высокий бросок не проходит")

	# p = 0.25 * (1 - min(0.4, 0.5)) = 0.15
	t.check(FakeTracker.should_switch(0.4, 0.5, 0.14), "бросок ниже порога проходит")
	t.check(not FakeTracker.should_switch(0.4, 0.5, 0.16), "бросок выше порога не проходит")

	# ── подмена меняет привязку, но не признак ────────────────
	var pair := SceneView.new(0.9, 0.9, false)
	for i in range(2):
		var o := SceneObject.new(i + 1, &"knight", 0.5, 0.0)
		o.cloak = PN.Cloak.RED if i == 0 else PN.Cloak.BLUE
		o.slot = i
		pair.objects.append(o)

	var ft := FakeTracker.new(7)
	ft.noise_amplitude = 0.0
	ft.update(pair)
	var cloak_a: int = ft.track_for(1).remembered_cloak
	var tid_a: int = ft.track_for(1).track_id
	t.eq(cloak_a, PN.Cloak.RED, "первый трек запомнил красный плащ")

	ft.force_switch(1, 2)
	t.eq(ft.track_for(2).track_id, tid_a, "трек первого рыцаря теперь указывает на второго")
	t.eq(ft.track_by_id(tid_a).object_id, 2, "и наоборот, по номеру трека")
	t.eq(ft.track_by_id(tid_a).remembered_cloak, cloak_a,
		"признак остался прежним — в этом вся подлость подмены")
	t.check(ft.track_by_id(tid_a).switched, "флаг подмены выставлен")

	# ── перезахват чинит ──────────────────────────────────────
	ft.recapture(pair)
	t.eq(ft.track_for(1).object_id, 1, "перезахват вернул верную привязку")
	t.check(not ft.track_for(1).switched, "перезахват снял флаг")
	t.eq(ft.track_for(1).remembered_cloak, PN.Cloak.RED, "признак перезаписан заново и верен")

	# ── подмена только между соседями ─────────────────────────
	var three := SceneView.new(0.9, 0.9, false)
	for i in range(3):
		var o := SceneObject.new(i + 1, &"knight", 0.5, 0.0)
		o.cloak = i
		o.slot = i
		three.objects.append(o)

	var swapped_far := false
	for s in range(200):
		var f := FakeTracker.new(s)
		f.noise_amplitude = 0.0
		f.update(three)
		var rng := RandomNumberGenerator.new()
		rng.seed = s
		f.roll_switches(three, rng)
		# рыцари 1 и 3 стоят через одного — их склейка невозможна
		var t1 := f.track_for(1)
		if t1 != null and t1.track_id == 3:
			swapped_far = true
	t.check(not swapped_far, "рыцари через одного никогда не склеиваются")

	# при ясной погоде подмен нет вообще
	var calm := SceneView.new(0.0, 0.0, false)
	for i in range(2):
		var o := SceneObject.new(i + 1, &"knight", 0.0, 0.0)
		o.cloak = i
		o.slot = i
		calm.objects.append(o)
	var any_switch := false
	for s in range(200):
		var f := FakeTracker.new(s)
		f.noise_amplitude = 0.0
		f.update(calm)
		var rng := RandomNumberGenerator.new()
		rng.seed = s
		f.roll_switches(calm, rng)
		if f.track_for(1).switched:
			any_switch = true
	t.check(not any_switch, "в ясную погоду трекер не путается ни разу за 200 прогонов")
