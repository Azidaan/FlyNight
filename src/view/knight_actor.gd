class_name KnightActor
extends RefCounted

## Сборка рыцаря из KayKit'овской модели (CC0).
##
## В паке рыцарь несёт сразу весь арсенал — все мечи и щиты видимы одновременно.
## Лишнее прячем. Плащ и щит красим в опознавательный цвет: это то, по чему
## игрок уличает трекер в подмене ID.

const MODEL := preload("res://assets/characters/Knight.glb")
const TARGET_HEIGHT := 1.75

## Всё, что рыцарь держать не должен. Щит Round_Shield оставляем.
const HIDE_MESHES := [
	"1H_Sword_Offhand", "1H_Sword", "2H_Sword",
	"Badge_Shield", "Rectangle_Shield", "Spike_Shield",
]
const TINT_MESHES := ["Knight_Cape", "Round_Shield"]

static func spawn(cloak_color: Color) -> Node3D:
	var node: Node3D = MODEL.instantiate()

	for mesh_name in HIDE_MESHES:
		var m := node.find_child(mesh_name, true, false)
		if m != null:
			m.visible = false

	for mesh_name in TINT_MESHES:
		_tint(node, mesh_name, cloak_color)

	# Нормируем рост по фактическим габаритам: так масштаб не привязан
	# к тому, в каких единицах пак экспортирован.
	var box := _combined_aabb(node)
	if box.size.y > 0.001:
		node.scale = Vector3.ONE * (TARGET_HEIGHT / box.size.y)

	var anim: AnimationPlayer = node.find_child("AnimationPlayer", true, false)
	if anim != null and anim.has_animation("Idle"):
		anim.play("Idle")

	return node

static func _tint(root: Node, mesh_name: String, c: Color) -> void:
	var mi := root.find_child(mesh_name, true, false) as MeshInstance3D
	if mi == null:
		return
	var mat := StandardMaterial3D.new()
	# Сплошной цвет без атласа: плащ должен читаться издалека однозначно,
	# а текстура пака его бы замутила.
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
