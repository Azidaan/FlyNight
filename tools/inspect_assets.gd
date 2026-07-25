extends SceneTree

## Замер габаритов моделей. Без этого расстановка сцены превращается
## в подбор чисел вслепую.

func _initialize() -> void:
	var paths: Array[String] = []
	for dir_path in ["res://assets/props", "res://assets/characters"]:
		var d := DirAccess.open(dir_path)
		if d == null:
			continue
		for f in d.get_files():
			if f.ends_with(".glb"):
				paths.append(dir_path + "/" + f)
	paths.sort()

	print("%-26s %8s %8s %8s   %s" % ["модель", "ширина", "высота", "глубина", "низ по Y"])
	for p in paths:
		var scene: PackedScene = load(p)
		if scene == null:
			print("%-26s  НЕ ЗАГРУЗИЛОСЬ" % p.get_file())
			continue
		var root: Node = scene.instantiate()
		var box := _aabb(root)
		print("%-26s %8.2f %8.2f %8.2f   %6.2f" % [
			p.get_file(), box.size.x, box.size.y, box.size.z, box.position.y])
		root.free()
	quit(0)

func _aabb(node: Node) -> AABB:
	var out := AABB()
	var first := true
	for mi in _meshes(node):
		var box: AABB = mi.get_aabb()
		if mi is Node3D:
			box = (mi as Node3D).transform * box
		if first:
			out = box
			first = false
		else:
			out = out.merge(box)
	return out

func _meshes(node: Node) -> Array[MeshInstance3D]:
	var out: Array[MeshInstance3D] = []
	if node is MeshInstance3D:
		out.append(node)
	for c in node.get_children():
		out.append_array(_meshes(c))
	return out
