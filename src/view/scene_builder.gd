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

## Место i-го зелья на столешнице. Прилавок развёрнут вдоль дороги,
## поэтому места считаются по её направлению, а не по оси X.
static func counter_spot(i: int) -> Vector3:
	var offsets := [-2.6, -0.9, 0.8, 2.5]
	return ROAD_DIR * offsets[i % offsets.size()] + Vector3(0.0, COUNTER_TOP, 0.0)

## Куда встают покупатели. Очередь уходит вглубь и вправо — так три слота
## реально отличаются по дистанции, и падение уверенности трекера
## с расстоянием выглядит обоснованным, а не выдуманным.
## Очередь стоит на обочине у прилавка и вытягивается вдоль дороги
## в сторону замка — оттуда покупатели и приходят.
const SLOTS := [
	Vector3(-0.2, 0.0, -5.0),
	Vector3(1.75, 0.0, -7.8),
	Vector3(3.70, 0.0, -10.6),
]

## Нежить выходит из леса по ту сторону дороги.
const WOLF_LANE := Vector3(-6.5, 0.0, -9.0)
const WOLF_STEP := Vector3(1.1, 0.0, 1.6)

## Дорога проходит МИМО лавки по диагонали: справа-вдали ворота замка,
## влево-вперёд она уходит из кадра. Лавка стоит на обочине, а не на дороге.
## Герои идут по дороге из замка и в замок, нежить выходит из леса
## с противоположной стороны и дорогу пересекает.
const ROAD_DIR := Vector3(0.574, 0.0, -0.819)
const ROAD_CENTER := Vector3(0.0, 0.0, -7.0)
const ROAD_HALF_WIDTH := 5.0
const CASTLE_ALONG := 64.0

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

## Расстояние от точки до осевой линии дороги — нужно, чтобы лес
## не зарастал проезжую часть.
static func dist_to_road(p: Vector3) -> float:
	var v := p - ROAD_CENTER
	var along := v.dot(ROAD_DIR)
	return (v - ROAD_DIR * along).length()

static func _road(root: Node3D) -> void:
	var side := Vector3(ROAD_DIR.z, 0.0, -ROAD_DIR.x)
	var t := -70.0
	while t < CASTLE_ALONG + 6.0:
		for lane in [-1.0, 0.0, 1.0]:
			_put(root, FLOOR, ROAD_CENTER + ROAD_DIR * t + side * (lane * 3.4),
				atan2(ROAD_DIR.x, ROAD_DIR.z))
		t += 3.4

## Замок справа, как на наброске. Отодвинут и укрупнён: в родном масштабе
## гекс-пака он был бы по пояс покупателю.
static func _castle(root: Node3D) -> void:
	# Замок стоит воротами в створе дороги, справа-вдали. Отсюда приходят
	# и сюда уходят покупатели.
	var yaw := atan2(ROAD_DIR.x, ROAD_DIR.z)
	var side := Vector3(ROAD_DIR.z, 0.0, -ROAD_DIR.x)
	var gate := ROAD_CENTER + ROAD_DIR * CASTLE_ALONG

	_put(root, CASTLE, gate + ROAD_DIR * 10.0, yaw, 13.0)
	_put(root, TOWER_B, gate + side * 13.0, yaw, 11.0)
	_put(root, TOWER_B, gate - side * 13.0, yaw, 11.0)
	_put(root, TOWER_A, gate + ROAD_DIR * 6.0 + side * 27.0, yaw + 0.3, 9.0)
	_put(root, TOWER_A, gate + ROAD_DIR * 6.0 - side * 27.0, yaw - 0.3, 9.0)

## Густой лес по обе стороны дороги. Гор нет намеренно: в референсе
## (Элвиннский лес) горизонт закрывает именно лес, а не хребты.
static func _forest(root: Node3D) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 20260725   # сцена должна быть одинаковой при каждом запуске

	var groves := [TREES_A_LARGE, TREES_B_LARGE, TREES_A_MED, TREES_B_MED]
	var singles := [TREE_A, TREE_B]

	var planted := 0
	var tries := 0
	while planted < 190 and tries < 3000:
		tries += 1
		var p := Vector3(rng.randf_range(-95.0, 95.0), 0.0, rng.randf_range(-95.0, 14.0))

		# дорогу не зарастаем, и пятачок вокруг лавки тоже
		if dist_to_road(p) < ROAD_HALF_WIDTH + 2.5:
			continue
		if p.length() < 11.0:
			continue

		var far := p.z < -26.0
		var scene: PackedScene = groves[rng.randi() % groves.size()] if far \
			else singles[rng.randi() % singles.size()]
		var factor := rng.randf_range(4.5, 7.0) if far else rng.randf_range(3.0, 4.4)
		_put(root, scene, p, rng.randf() * TAU, factor)
		planted += 1

	# сплошная стена леса на горизонте, чтобы по краям не зияла пустота
	for i in range(56):
		var p := Vector3(-140.0 + i * 5.0, 0.0, -102.0 + rng.randf_range(-7.0, 7.0))
		if dist_to_road(p) < ROAD_HALF_WIDTH + 3.0:
			continue
		_put(root, groves[i % groves.size()], p, rng.randf() * TAU,
			rng.randf_range(6.0, 8.5))

	for spot in [Vector3(-9.0, 0.0, -2.5), Vector3(8.5, 0.0, -2.0),
			Vector3(-11.5, 0.0, 2.0)]:
		_put(root, ROCK_A, spot, spot.x, 4.0)

static func _counter(root: Node3D) -> void:
	# Прилавок ставится параллельно дороге: он стоит на обочине,
	# а не поперёк проезжей части.
	_put(root, TABLE, Vector3(0.0, 0.0, 0.0), atan2(ROAD_DIR.x, ROAD_DIR.z))
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
