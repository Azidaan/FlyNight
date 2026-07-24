class_name SceneObject
extends RefCounted

## Объект глазами трекера: где он, насколько далеко, насколько перекрыт.

var object_id: int
var kind: StringName
var dist_norm: float
var occlusion: float
var cloak: int = -1             # PN.Cloak, только для рыцарей
var slot: int = -1              # позиция в очереди, только для рыцарей
var position: Vector3 = Vector3.ZERO

func _init(p_id: int, p_kind: StringName, p_dist: float, p_occl: float) -> void:
	object_id = p_id
	kind = p_kind
	dist_norm = p_dist
	occlusion = p_occl
