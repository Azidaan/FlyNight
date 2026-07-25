extends SceneTree

## Разбор всех персонажей пака: какие меши внутри. Нужно, чтобы понять,
## что прятать (лишнее оружие) и что красить в опознавательный цвет.

const MODELS := [
	"res://assets/characters/Knight.glb",
	"res://assets/characters/Barbarian.glb",
	"res://assets/characters/Mage.glb",
	"res://assets/characters/Rogue.glb",
	"res://assets/characters/Rogue_Hooded.glb",
]

func _initialize() -> void:
	for path in MODELS:
		var scene: PackedScene = load(path)
		if scene == null:
			printerr("не загрузилось: ", path)
			continue
		var root: Node = scene.instantiate()

		var names: Array[String] = []
		for mi in _meshes(root):
			names.append(mi.name)

		var anim: AnimationPlayer = root.find_child("AnimationPlayer", true, false)
		var clips := 0
		var has_idle := false
		var has_walk := false
		if anim != null:
			var list := anim.get_animation_list()
			clips = list.size()
			has_idle = anim.has_animation("Idle")
			for n in list:
				if "Walking_A" == n:
					has_walk = true

		var skel: Skeleton3D = root.find_child("Skeleton3D", true, false)
		var bones := skel.get_bone_count() if skel != null else 0

		print("\n=== ", path.get_file(), "  (клипов %d, костей %d, Idle:%s, Walking_A:%s)" % [
			clips, bones, "да" if has_idle else "НЕТ", "да" if has_walk else "НЕТ"])
		print("  ", ", ".join(names))
		root.free()
	quit(0)

func _meshes(node: Node) -> Array[MeshInstance3D]:
	var out: Array[MeshInstance3D] = []
	if node is MeshInstance3D:
		out.append(node)
	for c in node.get_children():
		out.append_array(_meshes(c))
	return out
