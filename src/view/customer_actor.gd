class_name CustomerActor
extends RefCounted

## Сборка покупателя из KayKit Adventurers Character Pack (CC0).
##
## Пять моделей с общим скелетом на 41 кость и одинаковым набором из 76 клипов —
## поэтому они полностью взаимозаменяемы. У каждой есть меш плаща, который
## красится в опознавательный цвет: по нему игрок уличает трекер в подмене ID.

const MODELS := [
	preload("res://assets/characters/Knight.glb"),
	preload("res://assets/characters/Barbarian.glb"),
	preload("res://assets/characters/Mage.glb"),
	preload("res://assets/characters/Rogue.glb"),
	preload("res://assets/characters/Rogue_Hooded.glb"),
]

const TARGET_HEIGHT := 1.75

## Оставляем только тело, плащ и головной убор. Всё остальное в паке —
## мечи, топоры, арбалеты, посохи, щиты, кружка — это оружие в руках,
## а к прилавку приходят покупать зелья, а не воевать.
const KEEP_SUFFIXES := [
	"_Cape", "_Body", "_Head", "_Head_Hooded",
	"_ArmLeft", "_ArmRight", "_LegLeft", "_LegRight",
	"_Helmet", "_Hat",
]

## Модель привязана к номеру покупателя, а не к случайности: сцена
## пересобирается каждый ход, и случайный выбор менял бы облик прямо в очереди.
static func spawn(customer_id: int, cloak_color: Color) -> Node3D:
	var node: Node3D = MODELS[customer_id % MODELS.size()].instantiate()

	for mi in _meshes(node):
		if not _is_kept(mi.name):
			mi.visible = false
		elif mi.name.ends_with("_Cape"):
			_tint(mi, cloak_color)

	# Рост нормируется по фактическим габаритам: если сменим пак,
	# масштаб не поедет.
	var box := _combined_aabb(node)
	if box.size.y > 0.001:
		node.scale = Vector3.ONE * (TARGET_HEIGHT / box.size.y)

	var anim: AnimationPlayer = node.find_child("AnimationPlayer", true, false)
	if anim != null and anim.has_animation("Idle"):
		anim.play("Idle")

	return node

static func _is_kept(mesh_name: String) -> bool:
	for suffix in KEEP_SUFFIXES:
		if mesh_name.ends_with(suffix):
			return true
	return false

static func _tint(mi: MeshInstance3D, c: Color) -> void:
	var mat := StandardMaterial3D.new()
	# Сплошной цвет без атласа: плащ должен читаться однозначно с любого
	# расстояния, текстура пака его бы замутила.
	mat.albedo_color = c
	mat.roughness = 0.85
	mi.material_override = mat

static func _combined_aabb(node: Node) -> AABB:
	var out := AABB()
	var first := true
	for mi in _meshes(node):
		if not mi.visible:
			continue
		var box := mi.get_aabb()
		if first:
			out = box
			first = false
		else:
			out = out.merge(box)
	return out

static func _meshes(node: Node) -> Array[MeshInstance3D]:
	var out: Array[MeshInstance3D] = []
	if node is MeshInstance3D:
		out.append(node)
	for c in node.get_children():
		out.append_array(_meshes(c))
	return out
