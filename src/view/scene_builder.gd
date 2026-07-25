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

## Место i-го зелья на столешнице. Прилавок стоит почти фронтально,
## поэтому места идут просто по оси X.
static func counter_spot(i: int) -> Vector3:
	var offsets := [-2.4, -0.8, 0.8, 2.4]
	return Vector3(offsets[i % offsets.size()], COUNTER_TOP, -0.35)

## Куда встают покупатели. Очередь уходит вглубь и вправо — так три слота
## реально отличаются по дистанции, и падение уверенности трекера
## с расстоянием выглядит обоснованным, а не выдуманным.
## Очередь стоит на обочине у прилавка и вытягивается вдоль дороги
## в сторону замка — оттуда покупатели и приходят.
const SLOTS := [
	Vector3(-0.4, 0.0, -4.4),
	Vector3(3.0, 0.0, -5.4),
	Vector3(6.4, 0.0, -6.6),
]

## Нежить выходит из леса по ту сторону дороги.
const WOLF_LANE := Vector3(-7.0, 0.0, -10.5)
const WOLF_STEP := Vector3(1.2, 0.0, 1.7)

## Дорога проходит МИМО лавки по диагонали: справа-вдали ворота замка,
## влево-вперёд она уходит из кадра. Лавка стоит на обочине, а не на дороге.
## Герои идут по дороге из замка и в замок, нежить выходит из леса
## с противоположной стороны и дорогу пересекает.
const ROAD_Z := -7.5
const ROAD_HALF_WIDTH := 5.0
const CASTLE_ALONG := 96.0

## Рельеф. Возле лавки сглаживается в ноль, иначе прилавок и покупатели
## встали бы на склоне и повисли в воздухе.
static func ground_height(x: float, z: float) -> float:
	var h := sin(x * 0.085) * 1.1 + cos(z * 0.065) * 0.8 + sin((x + z) * 0.041) * 0.6
	var damp: float = clampf(Vector2(x, z + 3.0).length() / 20.0, 0.0, 1.0)
	return h * damp

## Осевая линия дороги. Перед лавкой идёт поперёк кадра, дальше загибается
## вправо-вглубь к воротам замка — так дорога и фронтальна, и ведёт куда надо.
static func road_point(s: float) -> Vector3:
	var bend: float = 0.017 if s > 0.0 else 0.004
	var z := ROAD_Z - s * s * bend
	return Vector3(s, ground_height(s, z), z)

static func road_yaw(s: float) -> float:
	var a := road_point(s - 0.5)
	var b := road_point(s + 0.5)
	var d := (b - a).normalized()
	return atan2(d.x, d.z)

## Расстояние от точки до дороги — чтобы лес не зарастал проезжую часть.
static func dist_to_road(p: Vector3) -> float:
	var best := 1e9
	var s := -110.0
	while s <= CASTLE_ALONG + 10.0:
		var c := road_point(s)
		var d := Vector2(p.x - c.x, p.z - c.z).length()
		if d < best:
			best = d
		s += 3.0
	return best

static func build(root: Node3D) -> void:
	_ground(root)
	_road(root)
	_castle(root)
	_forest(root)
	_counter(root)
	_dressing(root)

## Холмистая земля вместо плоскости: сетка, поднятая по ground_height().
static func _ground(root: Node3D) -> void:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)

	var step := 4.0
	var half := 130.0
	var n := int(half * 2.0 / step)
	for i in range(n):
		for j in range(n):
			var x0 := -half + i * step
			var z0 := -half + j * step
			var x1 := x0 + step
			var z1 := z0 + step
			var a := Vector3(x0, ground_height(x0, z0), z0)
			var b := Vector3(x1, ground_height(x1, z0), z0)
			var c := Vector3(x1, ground_height(x1, z1), z1)
			var d := Vector3(x0, ground_height(x0, z1), z1)
			for v in [a, c, b, a, d, c]:
				st.add_vertex(v)

	st.generate_normals()

	var node := MeshInstance3D.new()
	node.mesh = st.commit()
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(0.20, 0.31, 0.15)
	mat.roughness = 1.0
	# двусторонний, чтобы не гадать с порядком обхода треугольников
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	node.material_override = mat
	node.position.y = -0.08
	root.add_child(node)

static func _road(root: Node3D) -> void:
	var s := -100.0
	while s < CASTLE_ALONG + 8.0:
		var centre := road_point(s)
		var yaw := road_yaw(s)
		var side := Vector3(cos(yaw), 0.0, -sin(yaw))
		for lane: float in [-1.0, 0.0, 1.0]:
			var p: Vector3 = centre + side * (lane * 3.4)
			p.y = ground_height(p.x, p.z) + 0.02
			_put(root, FLOOR, p, yaw)
		s += 3.4

## Замок справа, как на наброске. Отодвинут и укрупнён: в родном масштабе
## гекс-пака он был бы по пояс покупателю.
static func _castle(root: Node3D) -> void:
	# Замок стоит воротами в створе дороги, справа-вдали. Отсюда приходят
	# и сюда уходят покупатели.
	var gate := road_point(CASTLE_ALONG)
	var yaw := road_yaw(CASTLE_ALONG)
	var side := Vector3(cos(yaw), 0.0, -sin(yaw))
	var fwd := Vector3(sin(yaw), 0.0, cos(yaw))

	_put(root, CASTLE, gate + fwd * 12.0, yaw, 15.0)
	_put(root, TOWER_B, gate + side * 14.0, yaw, 12.0)
	_put(root, TOWER_B, gate - side * 14.0, yaw, 12.0)
	_put(root, TOWER_A, gate + fwd * 6.0 + side * 30.0, yaw + 0.3, 10.0)
	_put(root, TOWER_A, gate + fwd * 6.0 - side * 30.0, yaw - 0.3, 10.0)

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
		p.y = ground_height(p.x, p.z) - 0.3
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
	# Прилавок почти фронтально к игроку, как на наброске: длинной
	# стороной поперёк взгляда.
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
	glow.shadow_enabled = true
	glow.shadow_bias = 0.05
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
