class_name Track
extends RefCounted

## Один трек — то, что игрок видит вместо объекта.
##
## Ключевая пара полей: object_id — настоящая привязка, которой игрок не видит;
## remembered_cloak — признак, зафиксированный в момент захвата и показываемый
## в панели. После подмены они расходятся, и это единственный способ уличить
## трекер во лжи.

var track_id: int
var kind: StringName            # &"knight" | &"wolf" | &"coin"
var object_id: int
var conf: float = 1.0
var state: int = PN.TrackState.LOCKED
var switched: bool = false
# Признак и заказ фиксируются в момент захвата и НЕ пересчитываются от текущей
# привязки. Без этого подмена была бы бессмысленной: панель показывала бы заказ
# того рыцаря, к которому трек ведёт сейчас, и выданное зелье всегда подходило бы.
var remembered_cloak: int = -1  # PN.Cloak
var remembered_order: int = -1  # PN.Potion
var last_known_pos: Vector3 = Vector3.ZERO
var age: int = 0

func _init(p_track_id: int, p_kind: StringName, p_object_id: int) -> void:
	track_id = p_track_id
	kind = p_kind
	object_id = p_object_id

func refresh_state() -> void:
	state = PN.state_from_conf(conf)
