class_name Props
extends RefCounted

## Подвижный реквизит: зелья на прилавке, монеты, нежить из темноты.

## Форма бутылки отличает тип зелья даже когда цвет съеден темнотой —
## по спеке трекер слепнет, но сам прилавок игрок видит своими глазами.
const BOTTLES := [
	preload("res://assets/props/bottle_A_green.glb"),
	preload("res://assets/props/bottle_B_green.glb"),
	preload("res://assets/props/bottle_C_green.glb"),
]
const COIN := preload("res://assets/props/coin.glb")
const SKELETON := preload("res://assets/characters/Skeleton_Minion.glb")

const POTION_COLORS := [
	Color(0.30, 0.85, 0.42),   # лечения
	Color(0.86, 0.24, 0.24),   # силы
	Color(0.34, 0.52, 0.92),   # храбрости
]

static func bottle(potion: int) -> Node3D:
	var node: Node3D = BOTTLES[potion % BOTTLES.size()].instantiate()
	_tint_all(node, POTION_COLORS[potion % POTION_COLORS.size()], 0.35, 0.9)
	return node

static func coin() -> Node3D:
	return COIN.instantiate()

## Скелет вместо волка: нежить у ночного замка убедительнее зверя,
## а в паке анимаций у неё те же клипы, что у покупателей.
static func skeleton() -> Node3D:
	var node: Node3D = SKELETON.instantiate()
	for mesh_name in ["Skeleton_Blade", "Skeleton_Axe", "Skeleton_Crossbow",
			"Skeleton_Staff", "Skeleton_Quiver", "Skeleton_Arrow"]:
		var m := node.find_child(mesh_name, true, false)
		if m != null:
			m.visible = false

	# Через play_looping: клипы glTF приходят незациклёнными, и скелет
	# после одного прохода замирал в позе посреди шага.
	for clip in ["Walking_A", "Idle"]:
		var anim := node.find_child("AnimationPlayer", true, false) as AnimationPlayer
		if anim != null and anim.has_animation(clip):
			CustomerActor.play_looping(node, clip)
			break
	return node

static func _tint_all(node: Node, c: Color, roughness: float, metallic: float) -> void:
	for mi in _meshes(node):
		var mat := StandardMaterial3D.new()
		mat.albedo_color = c
		mat.roughness = roughness
		mat.metallic = metallic
		mi.material_override = mat

static func _meshes(node: Node) -> Array[MeshInstance3D]:
	var out: Array[MeshInstance3D] = []
	if node is MeshInstance3D:
		out.append(node)
	for c in node.get_children():
		out.append_array(_meshes(c))
	return out
