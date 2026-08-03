extends SceneTree

## Ищем кость кисти, чтобы прицепить к ней факел.

func _initialize() -> void:
	var scene: PackedScene = load("res://assets/characters/Knight.glb")
	var root: Node = scene.instantiate()
	var skel: Skeleton3D = root.find_child("Skeleton3D", true, false)
	if skel == null:
		printerr("скелета нет")
		quit(1)
		return

	print("костей: ", skel.get_bone_count())
	for i in range(skel.get_bone_count()):
		var n := skel.get_bone_name(i)
		var low := n.to_lower()
		if "hand" in low or "arm" in low or "slot" in low or "wrist" in low:
			print("  %2d  %s" % [i, n])
	quit(0)
