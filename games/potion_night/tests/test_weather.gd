extends RefCounted

func run(t: AssertBag) -> void:
	var w := Weather.new(0.0)

	t.almost(w.dark_at(0), 0.35, "в начале ночи темнота базовая")
	t.almost(w.dark_at(15), 0.90, "к середине ночи пик темноты")
	t.almost(w.dark_at(30), 0.35, "к рассвету темнота спадает")

	t.check(w.rain_at(15) > w.rain_at(0), "дождь усиливается к середине")
	t.check(w.rain_at(15) <= 1.0, "дождь не превышает единицы")
	t.check(w.rain_at(0) >= 0.0, "дождь не уходит в минус")

	# обе шкалы не выходят за границы ни на одном ходу ночи
	for turn in range(PN.NIGHT_TURNS + 1):
		var d := w.dark_at(turn)
		var r := w.rain_at(turn)
		if d < 0.0 or d > 1.0:
			t.check(false, "темнота вышла за границы на ходу %d: %f" % [turn, d])
		if r < 0.0 or r > 1.0:
			t.check(false, "дождь вышел за границы на ходу %d: %f" % [turn, r])
	t.check(true, "обе шкалы держатся в [0,1] всю ночь")

	var a := Weather.new(0.1)
	var b := Weather.new(-0.1)
	t.check(a.rain_at(10) != b.rain_at(10), "разный сдвиг даёт разные ночи")

	var rng := RandomNumberGenerator.new()
	rng.seed = 555
	var o1 := Weather.offset_from_rng(rng)
	rng.seed = 555
	var o2 := Weather.offset_from_rng(rng)
	t.almost(o1, o2, "сдвиг воспроизводим по сиду")
	t.check(absf(o1) <= 0.15, "сдвиг не выходит за 0.15")
