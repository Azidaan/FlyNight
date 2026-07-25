class_name SceneBuilder
extends RefCounted

## Постройка неподвижной части мира из KayKit Dungeon Remastered (CC0).
##
## Пак живёт в сетке 4 единицы на тайл: пол 4x4, стена 4x4x1, стол 4 в длину,
## персонаж ~3.1 в высоту. Все координаты ниже — в этих же единицах.

const TILE := 4.0

const FLOOR := preload("res://assets/props/floor_dirt_large.glb")
const WALL := preload("res://assets/props/wall.glb")
const WALL_WINDOW := preload("res://assets/props/wall_archedwindow_open.glb")
const WALL_CORNER := preload("res://assets/props/wall_corner.glb")
const TABLE := preload("res://assets/props/table_long_tablecloth.glb")
const CHEST := preload("res://assets/props/chest.glb")
const BARREL := preload("res://assets/props/barrel_large.glb")
const BARRELS_SMALL := preload("res://assets/props/barrel_small_stack.glb")
const CRATES := preload("res://assets/props/crates_stacked.glb")
const TORCH := preload("res://assets/props/torch_mounted.glb")
const BANNER_RED := preload("res://assets/props/banner_patternA_red.glb")
const BANNER_BLUE := preload("res://assets/props/banner_patternA_blue.glb")
const STOOL := preload("res://assets/props/stool.glb")

## Высота столешницы: на неё ставятся зелья.
const COUNTER_TOP := 1.0

## Куда встают покупатели. Очередь уходит вглубь и вправо — так три слота
## реально отличаются по дистанции, и падение уверенности трекера
## с расстоянием выглядит обоснованным, а не выдуманным.
const SLOTS := [
	Vector3(-0.8, 0.0, -3.0),
	Vector3(1.8, 0.0, -5.2),
	Vector3(4.2, 0.0, -7.4),
]

const WOLF_LANE_X := -5.0

static func build(root: Node3D) -> void:
	_floor(root)
	_back_wall(root)
	_counter(root)
	_dressing(root)

static func _floor(root: Node3D) -> void:
	for ix in range(-2, 3):
		for iz in range(-3, 2):
			_put(root, FLOOR, Vector3(ix * TILE, 0.0, iz * TILE))

static func _back_wall(root: Node3D) -> void:
	var z := -11.0
	for ix in range(-2, 3):
		var piece: PackedScene = WALL_WINDOW if ix == 0 or ix == 2 else WALL
		_put(root, piece, Vector3(ix * TILE, 0.0, z))
		# Второй ярус: одиночная стена низкая, и над ней зияла пустота.
		_put(root, WALL, Vector3(ix * TILE, TILE, z))

	_put(root, WALL_CORNER, Vector3(-3 * TILE + 1.0, 0.0, z), PI * 0.5)
	_put(root, WALL_CORNER, Vector3(3 * TILE - 1.0, 0.0, z), 0.0)

	_put(root, BANNER_RED, Vector3(-4.0, 2.4, z + 0.6))
	_put(root, BANNER_BLUE, Vector3(4.0, 2.4, z + 0.6))

	# Факелы на дальней стене: единственный свет, кроме фонаря на прилавке.
	for x in [-8.0, 8.0]:
		_put(root, TORCH, Vector3(x, 2.6, z + 0.6))
		var glow := OmniLight3D.new()
		glow.position = Vector3(x, 2.8, z + 1.2)
		glow.light_color = Color(1.0, 0.72, 0.38)
		glow.light_energy = 2.2
		glow.omni_range = 9.0
		root.add_child(glow)

static func _counter(root: Node3D) -> void:
	# Стол в паке длинный вдоль Z — разворачиваем поперёк взгляда.
	_put(root, TABLE, Vector3(0.0, 0.0, 0.0), PI * 0.5)
	_put(root, CHEST, Vector3(4.4, 0.0, 0.4), -PI * 0.35)
	_put(root, STOOL, Vector3(-3.4, 0.0, 1.0))

static func _dressing(root: Node3D) -> void:
	_put(root, BARREL, Vector3(-4.6, 0.0, -0.6))
	_put(root, BARRELS_SMALL, Vector3(6.2, 0.0, -1.8), PI * 0.2)
	_put(root, CRATES, Vector3(-7.0, 0.0, -3.4), -PI * 0.15)
	_put(root, BARREL, Vector3(7.6, 0.0, -5.0))
	_put(root, CRATES, Vector3(-8.2, 0.0, -7.6), PI * 0.4)

static func _put(root: Node3D, scene: PackedScene, pos: Vector3, yaw: float = 0.0) -> Node3D:
	var node: Node3D = scene.instantiate()
	node.position = pos
	node.rotation.y = yaw
	root.add_child(node)
	return node
