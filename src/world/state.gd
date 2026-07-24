class_name GameState
extends RefCounted

## Состояние одной ночи целиком. Ни одной ноды Godot: всё это гоняется headless.

var seed_value: int
var rng := RandomNumberGenerator.new()
var weather: Weather

var turn: int = 0
var knights: Array[Knight] = []
var counter: Array[int] = []          # PN.Potion, не длиннее PN.COUNTER_SLOTS
var chest: Dictionary = {}            # PN.Potion -> int
var coins: int = 0
var misses: int = 0
var lantern_turns: int = 0
var wolf_distance: int = -1           # -1 = волка нет
var next_knight_id: int = 1
var coins_in_flight: Array[int] = []
var next_coin_id: int = 100

var finished: bool = false
var outcome: String = ""              # "" | "dawn" | "failed"

func _init(p_seed: int) -> void:
	seed_value = p_seed
	rng.seed = p_seed
	weather = Weather.new(Weather.offset_from_rng(rng))
	for p in [PN.Potion.HEALING, PN.Potion.STRENGTH, PN.Potion.COURAGE]:
		chest[p] = PN.CHEST_PER_TYPE

func rain() -> float:
	return weather.rain_at(turn)

func dark() -> float:
	return weather.dark_at(turn)

func lantern_on() -> bool:
	return lantern_turns > 0

func free_queue_slot() -> int:
	for i in range(PN.QUEUE_SLOTS):
		if knight_in_slot(i) == null:
			return i
	return -1

func knight_in_slot(slot: int) -> Knight:
	for k in knights:
		if k.slot == slot:
			return k
	return null

func knight_by_id(id: int) -> Knight:
	for k in knights:
		if k.id == id:
			return k
	return null

## Типы, которые ещё физически есть в наличии. Заказы раздаются только среди них —
## поэтому ситуация «рыцарь просит то, чего нет» невозможна по построению,
## а не потому что мы про неё не забыли.
func available_potion_types() -> Array[int]:
	var out: Array[int] = []
	for p in [PN.Potion.HEALING, PN.Potion.STRENGTH, PN.Potion.COURAGE]:
		if chest[p] > 0 or counter.has(p):
			out.append(p)
	return out
