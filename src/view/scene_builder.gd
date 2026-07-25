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
const FENCE := preload("res://assets/props/barrier.glb")
const FENCE_POST := preload("res://assets/props/barrier_column.glb")
const STOOL := preload("res://assets/props/stool.glb")

const TREES_A_LARGE := preload("res://assets/hex/nature/trees_A_large.gltf")
const TREES_A_MED := preload("res://assets/hex/nature/trees_A_medium.gltf")
const TREES_B_LARGE := preload("res://assets/hex/nature/trees_B_large.gltf")
const TREES_B_MED := preload("res://assets/hex/nature/trees_B_medium.gltf")
const TREE_A := preload("res://assets/hex/nature/tree_single_A.gltf")
const TREE_B := preload("res://assets/hex/nature/tree_single_B.gltf")
const TREES_A_SMALL := preload("res://assets/hex/nature/trees_A_small.gltf")
const TREES_B_SMALL := preload("res://assets/hex/nature/trees_B_small.gltf")
const HILL_TREES_A := preload("res://assets/hex/nature/hills_A_trees.gltf")
const HILL_TREES_B := preload("res://assets/hex/nature/hills_B_trees.gltf")
const HILL_TREES_C := preload("res://assets/hex/nature/hills_C_trees.gltf")
const ROCK_A := preload("res://assets/hex/nature/rock_single_A.gltf")
const ROCK_C := preload("res://assets/hex/nature/rock_single_C.gltf")

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
## Откуда появляется — дальше самой дальней игровой дистанции, чтобы
## первый шаг тоже был шагом, а не возникновением из воздуха.
const WOLF_SPAWN_DISTANCE := 8

## Откуда покупатели приходят и куда уходят: со стороны замка (справа-вглубь)
## и прочь влево. Точки за краем кадра, чтобы вход и выход не возникали
## на глазах.
static func road_entry() -> Vector3:
	return road_point(34.0)

static func road_exit() -> Vector3:
	return road_point(-46.0)

## Дорога проходит МИМО лавки по диагонали: справа-вдали ворота замка,
## влево-вперёд она уходит из кадра. Лавка стоит на обочине, а не на дороге.
## Герои идут по дороге из замка и в замок, нежить выходит из леса
## с противоположной стороны и дорогу пересекает.
const ROAD_Z := -7.5
const ROAD_HALF_WIDTH := 4.6
const CASTLE_ALONG := 96.0

## Замок стоит правее и дальше, чем раньше: так он опускается в кадре
## и выходит из-под правой панели.
const CASTLE_POS := Vector3(58.0, 0.0, -52.0)
## Вокруг замка лес не сажаем — три ели закрывали его целиком.
const CASTLE_CLEAR := 34.0

## Рельеф. Возле лавки сглаживается в ноль, иначе прилавок и покупатели
## встали бы на склоне и повисли в воздухе.
static func ground_height(x: float, z: float) -> float:
	var h := sin(x * 0.085) * 1.1 + cos(z * 0.065) * 0.8 + sin((x + z) * 0.041) * 0.6
	var damp: float = clampf(Vector2(x, z + 3.0).length() / 20.0, 0.0, 1.0)
	return h * damp

## Осевая линия дороги — дуга окружности. У прилавка касательная идёт
## поперёк кадра (дорога фронтальна), дальше вправо она плавно
## разворачивается вглубь, и ворота замка попадают в кадр почти по центру.
## Влево радиус много больше, поэтому там дорога уходит почти прямо.
const ROAD_R_RIGHT := 42.0
const ROAD_R_LEFT := 260.0

static func road_point(s: float) -> Vector3:
	var r: float = ROAD_R_RIGHT if s > 0.0 else ROAD_R_LEFT
	var th := s / r
	var x := r * sin(th)
	var z := ROAD_Z - r + r * cos(th)
	return Vector3(x, ground_height(x, z), z)

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
	_fence(root)
	_castle(root)
	_forest(root)
	_counter(root)
	_dressing(root)
	_matte(root, {})

## Гасим зеркальный блик по всей статичной сцене.
##
## Шероховатости 1.0 мало: в StandardMaterial3D остаётся отдельный параметр
## metallic_specular со значением 0.5 по умолчанию, и он даёт пластиковый
## отсвет даже на матовой поверхности. Материалы у пака общие на атлас,
## поэтому обработанные запоминаем и не трогаем повторно.
static func _matte(node: Node, seen: Dictionary) -> void:
	if node is MeshInstance3D:
		var mi := node as MeshInstance3D
		var mats: Array = []
		if mi.material_override != null:
			mats.append(mi.material_override)
		if mi.mesh != null:
			for i in range(mi.mesh.get_surface_count()):
				mats.append(mi.mesh.surface_get_material(i))
		for m in mats:
			if m is BaseMaterial3D and not seen.has(m.get_instance_id()):
				seen[m.get_instance_id()] = true
				var b := m as BaseMaterial3D
				b.roughness = 1.0
				b.metallic = 0.0
				b.metallic_specular = 0.0
	for c in node.get_children():
		_matte(c, seen)

## Изгородь по дальней обочине — как в референсе, она держит перспективу
## и отделяет дорогу от поля.
static func _fence(root: Node3D) -> void:
	var s := -70.0
	while s < 46.0:
		var yaw := road_yaw(s)
		var side := Vector3(cos(yaw), 0.0, -sin(yaw))
		# ДАЛЬНЯЯ обочина: со знаком минус изгородь встала между камерой
		# и дорогой и загородила всю сцену.
		var p := road_point(s) + side * (ROAD_HALF_WIDTH + 2.2)
		p.y = ground_height(p.x, p.z) - 0.15
		# Длинная сторона у модели по оси X, поэтому к road_yaw добавляется
		# прямой угол — иначе секции встают поперёк дороги.
		# Модель в паке каменная, перекрашиваем в дерево.
		_tint(_put(root, FENCE, p, yaw + PI * 0.5, 1.0), Color(0.42, 0.28, 0.16))
		s += 4.0

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

## Дорога — не цельная плита, а утоптанная лента с россыпью булыжника.
static func _road(root: Node3D) -> void:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)

	# Поперёк лента дробится на полосы: по двум точкам на всю ширину она шла
	# плоско и на неровностях висела над землёй воздушной прослойкой.
	var cols := 6
	var prev: Array[Vector3] = []
	var s := -110.0
	while s <= CASTLE_ALONG + 14.0:
		var yaw := road_yaw(s)
		var side := Vector3(cos(yaw), 0.0, -sin(yaw))
		var centre := road_point(s)

		var row: Array[Vector3] = []
		for c in range(cols + 1):
			var f := -1.0 + 2.0 * float(c) / float(cols)
			var p: Vector3 = centre + side * (ROAD_HALF_WIDTH * f)
			p.y = ground_height(p.x, p.z) - 0.06   # чуть утоплена в грунт
			row.append(p)

		if not prev.is_empty():
			for c in range(cols):
				for v in [prev[c], prev[c + 1], row[c + 1],
						prev[c], row[c + 1], row[c]]:
					st.add_vertex(v)
		prev = row
		s += 2.5

	st.generate_normals()
	var band := MeshInstance3D.new()
	band.mesh = st.commit()
	var mat := StandardMaterial3D.new()
	# Светлее травы: тёмно-бурая лента ночью сливалась в чёрное пятно
	# и читалась как одна сплошная плита.
	mat.albedo_color = Color(0.46, 0.38, 0.28)
	mat.roughness = 1.0
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	band.material_override = mat
	root.add_child(band)

	# булыжник поверх ленты
	var rng := RandomNumberGenerator.new()
	rng.seed = 771
	var t := -108.0
	while t <= CASTLE_ALONG + 12.0:
		var yaw := road_yaw(t)
		var side := Vector3(cos(yaw), 0.0, -sin(yaw))
		# Камни кладём и за кромку ленты: иначе край выходит линейкой,
		# а не осыпью.
		for k in range(16):
			var off := rng.randf_range(-ROAD_HALF_WIDTH - 1.2, ROAD_HALF_WIDTH + 1.2)
			var p := road_point(t + rng.randf_range(-1.3, 1.3)) + side * off
			p.y = ground_height(p.x, p.z) - 0.09
			var stone := _put(root, ROCK_A if rng.randf() < 0.6 else ROCK_C, p,
				rng.randf() * TAU, rng.randf_range(0.9, 1.9))
			_tint(stone, Color(
				rng.randf_range(0.52, 0.66),
				rng.randf_range(0.50, 0.62),
				rng.randf_range(0.46, 0.58)))
		t += 2.4

## Замок справа, как на наброске. Отодвинут и укрупнён: в родном масштабе
## гекс-пака он был бы по пояс покупателю.
static func _castle(root: Node3D) -> void:
	# Замок стоит воротами в створе дороги, справа-вдали. Отсюда приходят
	# и сюда уходят покупатели.
	var gate := road_point(CASTLE_ALONG)
	var yaw := road_yaw(CASTLE_ALONG)
	var side := Vector3(cos(yaw), 0.0, -sin(yaw))
	var fwd := Vector3(sin(yaw), 0.0, cos(yaw))

	# Замок — в правом верхнем углу и по большей части ЗА кадром: в референсе
	# видно только воротную башню и кусок стены, всё остальное поле занимают
	# трава, дорога и лес. Координаты заданы явно, а не выведены из дуги
	# дороги: по дуге он выходил ровно в центр экрана.
	var keep := CASTLE_POS
	var wall_yaw := -0.55

	_put(root, CASTLE, keep, wall_yaw, 10.0)
	_put(root, TOWER_B, keep + Vector3(-16.0, 0.0, 8.0), wall_yaw, 8.5)
	for i in range(4):
		_put(root, TOWER_A, keep + Vector3(19.0 + i * 18.0, 0.0, -6.0 - i * 9.0),
			wall_yaw, 8.0)

	# Факелы на стенах: без них замок сливался с ночным небом чёрным силуэтом.
	# Тени у них выключены — это далёкий фон, а тень от каждого источника
	# стоит полного прохода по сцене.
	var torches := [
		keep + Vector3(-13.0, 9.0, 10.0), keep + Vector3(-4.0, 11.0, 6.0),
		keep + Vector3(7.0, 9.5, 3.0), keep + Vector3(18.0, 8.0, -4.0),
		keep + Vector3(34.0, 7.5, -12.0), keep + Vector3(52.0, 7.0, -21.0),
		keep + Vector3(-9.0, 2.5, 12.0), keep + Vector3(12.0, 2.5, 2.0),
	]
	for t in torches:
		_put(root, TORCH, t, wall_yaw + PI, 3.2)
		var fire := OmniLight3D.new()
		fire.position = t + Vector3(0.0, 1.0, 1.6)
		fire.light_color = Color(1.0, 0.66, 0.30)
		fire.light_energy = 9.0
		fire.omni_range = 26.0
		root.add_child(fire)

	# Общая подсветка стен, чтобы кладка читалась, а не тонула в чёрном.
	var wash := OmniLight3D.new()
	wash.position = keep + Vector3(6.0, 16.0, 16.0)
	wash.light_color = Color(0.72, 0.72, 0.95)
	wash.light_energy = 7.0
	wash.omni_range = 70.0
	root.add_child(wash)

## Можно ли сажать в этой точке: не на дороге, не на пятачке лавки,
## не перед замком и не на линии взгляда к нему.
static func _spot_free(p: Vector3) -> bool:
	var d := dist_to_road(p)
	# Обочина расчищена: раньше чаща подступала вплотную и заслоняла
	# и тропинку, и выходящую на неё нежить.
	if d < ROAD_HALF_WIDTH + 7.5:
		return false
	# Дальше лес загустевает постепенно, а не встаёт стенкой по линейке.
	if d < ROAD_HALF_WIDTH + 19.0:
		var t := (d - ROAD_HALF_WIDTH - 7.5) / 11.5
		var n := float(absi(hash(Vector2i(int(p.x * 4.0), int(p.z * 4.0)))) % 1000) / 1000.0
		if n > t:
			return false
	if Vector2(p.x, p.z).length() < 11.0:
		return false
	if Vector2(p.x - CASTLE_POS.x, p.z - CASTLE_POS.z).length() < CASTLE_CLEAR:
		return false
	if p.z < -14.0 and p.z > CASTLE_POS.z \
			and absf(p.x - CASTLE_POS.x * (p.z / CASTLE_POS.z)) < 11.0:
		return false
	# тропа нежити: куртина выросла прямо на ней и скрыла скелета,
	# рамка трекера висела пустой
	var a := Vector2(WOLF_LANE.x, WOLF_LANE.z)
	var b := a - Vector2(WOLF_STEP.x, WOLF_STEP.z) * 6.0
	var ab := b - a
	var t: float = clampf((Vector2(p.x, p.z) - a).dot(ab) / ab.length_squared(), 0.0, 1.0)
	if (Vector2(p.x, p.z) - (a + ab * t)).length() < 8.0:
		return false
	return true

## Густой лес по обе стороны дороги. Гор нет намеренно: в референсе
## (Элвиннский лес) горизонт закрывает именно лес, а не хребты.
static func _forest(root: Node3D) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 20260725   # сцена должна быть одинаковой при каждом запуске

	var groves := [TREES_A_LARGE, TREES_B_LARGE, TREES_A_MED, TREES_B_MED]
	var singles := [TREE_A, TREE_B]

	# Дальняя гряда: холмы с лесом задают ярусный силуэт — кроны на разной
	# высоте, а не ровная гребёнка одинаковых конусов.
	for i in range(26):
		var hx := rng.randf_range(-130.0, 130.0)
		var hz := rng.randf_range(-118.0, -58.0)
		if Vector2(hx - CASTLE_POS.x, hz - CASTLE_POS.z).length() < CASTLE_CLEAR + 8.0:
			continue
		var hill: PackedScene = [HILL_TREES_A, HILL_TREES_B, HILL_TREES_C][rng.randi() % 3]
		var f := rng.randf_range(11.0, 20.0)
		# Это шестиугольные плитки с толстым основанием. Поднятые над землёй,
		# они торчали из леса кубами — топим так, чтобы осталась только
		# вершина с деревьями. Высота плитки в родном масштабе ~1.33.
		var sink := -1.33 * f * 0.62
		_put(root, hill, Vector3(hx, sink, hz), rng.randf() * TAU, f)

	# Чаща куртинами, а не ровной россыпью: у каждой куртины свой ярус высоты,
	# оттого кроны стоят на разных уровнях и лес перестаёт быть гребёнкой
	# одинаковых конусов.
	var smalls := [TREES_A_SMALL, TREES_B_SMALL]
	for i in range(64):
		var cx := rng.randf_range(-105.0, 105.0)
		var cz := rng.randf_range(-105.0, 12.0)
		var spread := rng.randf_range(4.0, 13.0)

		# ярус куртины: чем дальше, тем выше могут быть деревья
		var depth: float = clampf((-cz) / 90.0, 0.0, 1.0)
		var tier := rng.randf()
		var lo := lerpf(3.4, 9.0, depth) * lerpf(0.7, 1.5, tier)
		var hi := lo * rng.randf_range(1.3, 1.9)

		for k in range(rng.randi_range(7, 15)):
			var ang := rng.randf() * TAU
			var rad := spread * sqrt(rng.randf())
			var p := Vector3(cx + cos(ang) * rad, 0.0, cz + sin(ang) * rad)
			if not _spot_free(p):
				continue

			var big := rng.randf() < 0.62
			var pool: Array = groves if big else singles
			var scene: PackedScene = pool[rng.randi() % pool.size()]
			p.y = ground_height(p.x, p.z) - 0.3
			_put(root, scene, p, rng.randf() * TAU,
				rng.randf_range(lo, hi) * (1.0 if big else 0.55))

		# подлесок под куртиной
		for k in range(rng.randi_range(4, 9)):
			var ang := rng.randf() * TAU
			var rad := spread * 1.3 * sqrt(rng.randf())
			var p := Vector3(cx + cos(ang) * rad, 0.0, cz + sin(ang) * rad)
			if not _spot_free(p):
				continue
			p.y = ground_height(p.x, p.z) - 0.25
			_put(root, smalls[rng.randi() % smalls.size()], p,
				rng.randf() * TAU, rng.randf_range(2.2, 4.6))

	# сплошная стена леса на горизонте, чтобы по краям не зияла пустота
	for i in range(56):
		var p := Vector3(-140.0 + i * 5.0, 0.0, -102.0 + rng.randf_range(-7.0, 7.0))
		if dist_to_road(p) < ROAD_HALF_WIDTH + 3.0:
			continue
		if Vector2(p.x - CASTLE_POS.x, p.z - CASTLE_POS.z).length() < CASTLE_CLEAR:
			continue
		_put(root, groves[i % groves.size()], p, rng.randf() * TAU,
			rng.randf_range(9.0, 13.0))

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

static func _tint(node: Node, c: Color) -> void:
	for child in node.get_children():
		_tint(child, c)
	if node is MeshInstance3D:
		var mat := StandardMaterial3D.new()
		mat.albedo_color = c
		mat.roughness = 1.0
		(node as MeshInstance3D).material_override = mat

static func _put(root: Node3D, scene: PackedScene, pos: Vector3,
		yaw: float = 0.0, scale_factor: float = 1.0) -> Node3D:
	var node: Node3D = scene.instantiate()
	node.position = pos
	node.rotation.y = yaw
	if not is_equal_approx(scale_factor, 1.0):
		node.scale = Vector3.ONE * scale_factor
	root.add_child(node)
	return node
