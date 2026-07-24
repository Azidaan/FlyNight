class_name SceneView
extends RefCounted

## Срез сцены, который скармливается трекеру. Настоящий YOLO из проекта 3
## получит вместо этого кадр из буфера — интерфейс TrackSource от этого
## не поменяется.

var objects: Array[SceneObject] = []
var rain: float
var dark: float
var lantern_on: bool
var turn: int = 0

func _init(p_rain: float, p_dark: float, p_lantern: bool) -> void:
	rain = p_rain
	dark = p_dark
	lantern_on = p_lantern

static func from_state(s: GameState) -> SceneView:
	var v := SceneView.new(s.rain(), s.dark(), s.lantern_on())
	v.turn = s.turn

	for k in s.knights:
		if k.slot < 0:
			continue
		# слот 0 у самого прилавка, слот 2 в хвосте очереди
		var dist := float(k.slot) / float(PN.QUEUE_SLOTS - 1)
		var o := SceneObject.new(k.id, &"knight", dist, 0.0)
		o.cloak = k.cloak
		o.order = k.order
		o.slot = k.slot
		v.objects.append(o)

	if s.wolf_distance >= 0:
		# волк у кромки леса — перекрыт ветками, оттого и не берётся
		v.objects.append(SceneObject.new(-1, &"wolf", 1.0, 0.35))

	for coin_id in s.coins_in_flight:
		# монета лежит низко и частично закрыта прилавком
		v.objects.append(SceneObject.new(coin_id, &"coin", 0.0, 0.30))

	return v
