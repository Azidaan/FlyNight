class_name ActionResult
extends RefCounted

## Итог применения карты.
##
## ok = false означает «ход НЕ засчитан, состояние не тронуто» — например,
## попытка положить зелье на полный прилавок. Промах камнем при этом ok = true:
## ход потрачен, и это и есть наказание. Иначе можно было бы кидать бесплатно,
## пока не попадёшь.

var ok: bool = true
var reason: String = ""
var events: Array[GameEvent] = []

static func fail(p_reason: String) -> ActionResult:
	var r := ActionResult.new()
	r.ok = false
	r.reason = p_reason
	return r

func emit(kind: StringName, data: Dictionary = {}) -> void:
	events.append(GameEvent.new(kind, data))
