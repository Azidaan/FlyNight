extends SceneTree

## Разбор KayKit'овского рыцаря: какие есть клипы, есть ли скелет,
## как называются меши и материалы (нужно найти плащ, чтобы красить его
## в опознавательный цвет).

const MODEL := "res://assets/characters/Knight.glb"

func _initialize() -> void:
	var scene: PackedScene = load(MODEL)
	if scene == null:
		printerr("не загрузилось: ", MODEL)
		quit(1)
		return

	var root: Node = scene.instantiate()
	print("корень: ", root.name, "  (", root.get_class(), ")")

	var player: AnimationPlayer = root.find_child("AnimationPlayer", true, false)
	if player == null:
		printerr("AnimationPlayer не найден")
	else:
		var names := player.get_animation_list()
		print("\nанимаций: ", names.size())
		var interesting := []
		for n in names:
			var low := n.to_lower()
			if "idle" in low or "walk" in low or "run" in low or "death" in low or "interact" in low:
				interesting.append("%s (%.2f с)" % [n, player.get_animation(n).length])
		print("подходящие нам:")
		for i in interesting:
			print("  ", i)

	print("\nструктура:")
	_walk(root, 0)
	quit(0)

func _walk(node: Node, depth: int) -> void:
	var pad := "  ".repeat(depth)
	if node is Skeleton3D:
		print(pad, node.name, "  [Skeleton3D, костей ", node.get_bone_count(), "]")
	elif node is MeshInstance3D:
		var mi := node as MeshInstance3D
		var mats := []
		if mi.mesh != null:
			for i in range(mi.mesh.get_surface_count()):
				var m := mi.mesh.surface_get_material(i)
				mats.append(m.resource_name if m != null and m.resource_name != "" else "(без имени)")
		print(pad, node.name, "  [Mesh, поверхностей ", mats.size(), ": ", ", ".join(mats), "]")
	elif depth <= 2:
		print(pad, node.name, "  [", node.get_class(), "]")
	for c in node.get_children():
		_walk(c, depth + 1)
