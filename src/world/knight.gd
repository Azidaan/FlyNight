class_name Knight
extends RefCounted

## Рыцарь глазами мира: настоящий номер, настоящий заказ, настоящий плащ.
## Игрок ничего этого напрямую не видит — только то, что скажет трекер.

var id: int
var order: int          # PN.Potion
var cloak: int          # PN.Cloak — единственный видимый признак,
                        # по которому можно уличить трекер в подмене
var patience: int
var slot: int = -1      # -1 = вне очереди

func _init(p_id: int, p_order: int, p_cloak: int, p_patience: int) -> void:
	id = p_id
	order = p_order
	cloak = p_cloak
	patience = p_patience
