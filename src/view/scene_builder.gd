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

## Дорога идёт по x около нуля вглубь, к воротам замка. Лавка стоит
## на обочине, покупатели подходят по дороге.
const ROAD_HALF_WIDTH := 7.0
const ROAD_FAR := -52.0

static func build(root: Node3D) -> void:
	_ground(root)
	_road(root)
	_castle(root)
	_forest(root)
	_counter(root)
	_dressing(root)

static func _ground(root: Node3D) -> void:
	var plane := MeshInstance3D.new()
	var mesh := PlaneMesh.new()
	mesh.size = Vector2(260.0, 260.0)
	plane.mesh = mesh
	plane.position = Vector3(0.0, -0.05, -60.0)
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.20, 0.31, 0.15)
	mat.roughness = 1.0
	plane.material_override = mat
	root.add_child(plane)

static func _road(root: Node3D) -> void:
	for iz in range(-13, 3):
		for ix in range(-1, 2):
			_put(root, FLOOR, Vector3(ix * TILE, 0.0, iz * TILE))

## Замок справа, как на наброске. Отодвинут и укрупнён: в родном масштабе
## гекс-пака он был бы по пояс покупателю.
static func _castle(root: Node3D) -> void:
	# Замок стоит воротами в створе дороги и должен нависать, а не маячить
	# точкой на горизонте — как врата Штормграда в референсе.
	_put(root, CASTLE, Vector3(3.0, 0.0, ROAD_FAR - 8.0), 0.0, 13.0)
	_put(root, TOWER_B, Vector3(-13.0, 0.0, ROAD_FAR + 4.0), 0.0, 11.0)
	_put(root, TOWER_B, Vector3(17.0, 0.0, ROAD_FAR + 4.0), 0.0, 11.0)
	_put(root, TOWER_A, Vector3(-26.0, 0.0, ROAD_FAR - 6.0), 0.3, 9.0)
	_put(root, TOWER_A, Vector3(30.0, 0.0, ROAD_FAR - 6.0), -0.3, 9.0)

## Густой лес по обе стороны дороги. Гор нет намеренно: в референсе
## (Элвиннский лес) горизонт закрывает именно лес, а не хребты.
static func _forest(root: Node3D) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 20260725   # сцена должна быть одинаковой при каждом запуске

	var groves := [TREES_A_LARGE, TREES_B_LARGE, TREES_A_MED, TREES_B_MED]
	var singles := [TREE_A, TREE_B]

	for i in range(150):
		var z := rng.randf_range(6.0, ROAD_FAR - 26.0)
		var x: float = ROAD_HALF_WIDTH + rng.randf_range(1.0, 48.0)
		if rng.randf() < 0.5:
			x = -x

		var far := z < -26.0
		var scene: PackedScene = groves[rng.randi() % groves.size()] if far \
			else singles[rng.randi() % singles.size()]
		var factor := rng.randf_range(4.5, 7.0) if far else rng.randf_range(3.0, 4.4)
		_put(root, scene, Vector3(x, 0.0, z), rng.randf() * TAU, factor)

	# сплошная стена леса на горизонте, чтобы за замком не зияла пустота
	for i in range(46):
		_put(root, groves[i % groves.size()],
			Vector3(-112.0 + i * 5.0, 0.0, ROAD_FAR - 30.0 + rng.randf_range(-7.0, 7.0)),
			rng.randf() * TAU, rng.randf_range(6.0, 8.5))

	for spot in [Vector3(-8.6, 0.0, -9.0), Vector3(9.2, 0.0, -10.5),
			Vector3(-10.4, 0.0, -2.0)]:
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
