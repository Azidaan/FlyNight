class_name Weather
extends RefCounted

## Погода как чистая функция номера хода. Обе шкалы идут по синусоиде с пиком
## к середине ночи: сначала темнеет и льёт всё сильнее, под утро отпускает.
## Сдвиг дождя берётся из сида — так ночи отличаются друг от друга.

var rain_offset: float

func _init(offset: float) -> void:
	rain_offset = offset

func dark_at(turn: int) -> float:
	var t := float(turn) / float(PN.NIGHT_TURNS)
	return clampf(0.35 + 0.55 * sin(PI * t), 0.0, 1.0)

func rain_at(turn: int) -> float:
	var t := float(turn) / float(PN.NIGHT_TURNS)
	var shifted := clampf(t - rain_offset, 0.0, 1.0)
	return clampf(0.15 + 0.75 * sin(PI * shifted), 0.0, 1.0)

static func offset_from_rng(rng: RandomNumberGenerator) -> float:
	return rng.randf_range(-0.15, 0.15)
