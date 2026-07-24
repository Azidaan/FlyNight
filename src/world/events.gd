class_name GameEvent
extends RefCounted

## Событие для аниматора. Содержимое намеренно нетипизировано: view разбирает
## его по kind, а мир не должен знать, как это будет нарисовано.

var kind: StringName
var data: Dictionary

func _init(p_kind: StringName, p_data: Dictionary = {}) -> void:
	kind = p_kind
	data = p_data
