class_name SceneBuilder
extends RefCounted

## Постройка неподвижной части мира.
##
## Два пака в разных масштабах. Dungeon Remastered — ближний план, сетка
## 4 единицы на тайл, персонаж ~3.1 в высоту. Medieval Hexagon — дальний план,
## там дерево всего 1.2, поэтому его модели ставятся далеко и укрупняются.
## Все координаты ниже — в единицах ближнего плана.

const TILE := 4.0

const FLOOR := preload("res://assets/props/floor_dirt_large.glb")
const TABLE := preload("res://assets/props/table_long_tablecloth.glb")
const CHEST := preload("res://assets/props/chest.glb")
const BARREL := preload("res://assets/props/barrel_large.glb")
const BARRELS_SMALL := preload("res://assets/props/barrel_small_stack.glb")
const CRATES := preload("res://assets/props/crates_stacked.glb")
const TORCH := preload("res://assets/props/torch_mounted.glb")
const STOOL := preload("res://assets/props/stool.glb")

const HILLS_NEAR := preload("res://assets/hex/nature/hills_C_trees.gltf")
const HILLS_MID := preload("res://assets/hex/nature/hills_B_trees.gltf")
const HILLS_FAR := preload("res://assets/hex/nature/hills_A_trees.gltf")
const MOUNTAIN_A := preload("res://assets/hex/nature/mountain_A_grass_trees.gltf")
const MOUNTAIN_C := preload("res://assets/hex/nature/mountain_C_grass_trees.gltf")
const TREES_A_LARGE := preload("res://assets/hex/nature/trees_A_large.gltf")
const TREES_A_MED := preload("res://assets/hex/nature/trees_A_medium.gltf")
const TREES_B_LARGE := preload("res://assets/hex/nature/trees_B_large.gltf")
const TREES_B_MED := preload("res://assets/hex/nature/trees_B_medium.gltf")
const TREE_A := preload("res://assets/hex/nature/tree_single_A.gltf")
const TREE_B := preload("res://assets/hex/nature/tree_single_B.gltf")
const ROCK_A := preload("res://assets/hex/nature/rock_single_A.gltf")

const CASTLE := preload("res://assets/hex/castle/building_castle_red.gltf")
const TOWER_A := preload("res://assets/hex/castle/building_tower_A_red.gltf")
const TOWER_B := preload("res://assets/hex/castle/building_tower_B_red.gltf")

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
	_ground(root)
	_castle(root)
	_ridges(root)
	_forest(root)
	_counter(root)
	_dressing(root)

static func _ground(root: Node3D) -> void:
	for ix in range(-3, 4):
		for iz in range(-4, 2):
			_put(root, FLOOR, Vector3(ix * TILE, 0.0, iz * TILE))

## Замок справа, как на наброске. Отодвинут и укрупнён: в родном масштабе
## гекс-пака он был бы по пояс покупателю.
static func _castle(root: Node3D) -> void:
	_put(root, CASTLE, Vector3(26.0, 3.0, -40.0), -PI * 0.28, 9.0)
	_put(root, TOWER_A, Vector3(13.0, 1.5, -30.0), 0.4, 7.0)
	_put(root, TOWER_B, Vector3(40.0, 4.0, -54.0), -0.2, 8.0)
	_put(root, TOWER_A, Vector3(21.0, 5.0, -62.0), 0.9, 6.5)

## Три гряды, уходящие влево и вдаль — зелёная, синяя, фиолетовая с наброска.
## Дальние тонут в тумане сами, отдельного осветления не требуется.
static func _ridges(root: Node3D) -> void:
	# Ярусы поднимаются по Y: без этого гряды сливались в одну плоскую полосу,
	# а на наброске они отчётливо стоят друг над другом.
	for i in range(7):
		_put(root, HILLS_NEAR, Vector3(-10.0 - i * 12.0, 1.0, -26.0 - i * 2.0),
			i * 0.7, 7.0)
	for i in range(7):
		_put(root, HILLS_MID, Vector3(-2.0 - i * 15.0, 4.5, -48.0 - i * 3.0),
			i * 1.1, 10.0)
	for i in range(6):
		_put(root, MOUNTAIN_A if i % 2 == 0 else MOUNTAIN_C,
			Vector3(2.0 - i * 22.0, 9.0, -78.0 - i * 4.0), i * 0.5, 15.0)
	for i in range(3):
		_put(root, HILLS_FAR, Vector3(34.0 + i * 20.0, 3.0, -60.0 - i * 6.0),
			i * 0.8, 10.0)

## Лес среднего плана: кромка, из которой выходит нежить.
static func _forest(root: Node3D) -> void:
	# Лес разведён по краям: сплошной стеной он закрывал гряды,
	# и глубина, ради которой всё затевалось, пропадала.
	var spots := [
		Vector3(-17.0, 0.0, -13.0), Vector3(-23.0, 0.0, -18.0),
		Vector3(-29.0, 0.0, -13.0), Vector3(-21.0, 0.0, -24.0),
		Vector3(-33.0, 0.0, -21.0), Vector3(-15.0, 0.0, -20.0),
		Vector3(17.0, 0.0, -16.0), Vector3(23.0, 0.0, -22.0),
		Vector3(14.0, 0.0, -24.0),
	]
	var kinds := [TREES_A_LARGE, TREES_B_LARGE, TREES_A_MED, TREES_B_MED]
	for i in range(spots.size()):
		_put(root, kinds[i % kinds.size()], spots[i], float(i) * 0.9, 4.5)

	# одиночные деревья ближе, чтобы кромка не обрывалась стеной
	for spot in [Vector3(-9.5, 0.0, -10.5), Vector3(-13.5, 0.0, -8.0),
			Vector3(9.5, 0.0, -12.0)]:
		_put(root, TREE_A if spot.x < 0.0 else TREE_B, spot, spot.z, 3.2)

	for spot in [Vector3(-7.0, 0.0, -9.0), Vector3(7.5, 0.0, -9.5)]:
		_put(root, ROCK_A, spot, spot.x, 4.0)

static func _counter(root: Node3D) -> void:
	# Стол в паке длинный вдоль Z — разворачиваем поперёк взгляда.
	_put(root, TABLE, Vector3(0.0, 0.0, 0.0), PI * 0.5)
	_put(root, CHEST, Vector3(4.4, 0.0, 0.4), -PI * 0.35)
	_put(root, STOOL, Vector3(-3.4, 0.0, 1.0))

	# Факел у прилавка — единственный близкий источник света.
	_put(root, TORCH, Vector3(3.2, 2.2, 1.2), PI)
	var glow := OmniLight3D.new()
	glow.position = Vector3(3.2, 2.6, 1.4)
	glow.light_color = Color(1.0, 0.74, 0.4)
	glow.light_energy = 3.0
	glow.omni_range = 14.0
	root.add_child(glow)

static func _dressing(root: Node3D) -> void:
	_put(root, BARREL, Vector3(-4.6, 0.0, -0.6))
	_put(root, BARRELS_SMALL, Vector3(6.2, 0.0, -1.8), PI * 0.2)
	_put(root, CRATES, Vector3(-7.0, 0.0, -3.4), -PI * 0.15)
	_put(root, BARREL, Vector3(7.6, 0.0, -5.0))
	_put(root, CRATES, Vector3(-8.2, 0.0, -7.6), PI * 0.4)

static func _put(root: Node3D, scene: PackedScene, pos: Vector3,
		yaw: float = 0.0, scale_factor: float = 1.0) -> Node3D:
	var node: Node3D = scene.instantiate()
	node.position = pos
	node.rotation.y = yaw
	if not is_equal_approx(scale_factor, 1.0):
		node.scale = Vector3.ONE * scale_factor
	root.add_child(node)
	return node
