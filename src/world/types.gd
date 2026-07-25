class_name PN
extends RefCounted

## Перечисления и все числа баланса игры. Держатся в одном месте намеренно:
## раздел 13 спеки требует, чтобы баланс правился без раскопок по коду —
## числа взяты с потолка и будут крутиться по итогам плейтеста.

enum Potion { HEALING, STRENGTH, COURAGE }
enum Cloak { RED, BLUE, GREEN, YELLOW, PURPLE }
enum TrackState { LOCKED, DRIFTING, LOST }
enum ActionKind { GIVE, RESTOCK, LANTERN, ROCK, RECAPTURE }

const COUNTER_SLOTS := 4
## Обслуживание по одному: подошёл, спросил, получил, ушёл. Очередей нет.
const QUEUE_SLOTS := 1
const CHEST_PER_TYPE := 6
const NIGHT_TURNS := 30
## Ошибочные выдачи считаются для отчёта, но ночь не обрывают: игра спокойная,
## ограничений по времени обслуживания нет.
const MAX_MISSES := 3

const CONF_LOCKED := 0.70
const CONF_DRIFTING := 0.35

static func potion_name(p: int) -> String:
	match p:
		Potion.HEALING: return "лечения"
		Potion.STRENGTH: return "силы"
		Potion.COURAGE: return "храбрости"
	return "?"

static func cloak_name(c: int) -> String:
	match c:
		Cloak.RED: return "красный"
		Cloak.BLUE: return "синий"
		Cloak.GREEN: return "зелёный"
		Cloak.YELLOW: return "жёлтый"
		Cloak.PURPLE: return "фиолетовый"
	return "?"

static func state_from_conf(conf: float) -> int:
	if conf >= CONF_LOCKED:
		return TrackState.LOCKED
	if conf >= CONF_DRIFTING:
		return TrackState.DRIFTING
	return TrackState.LOST
