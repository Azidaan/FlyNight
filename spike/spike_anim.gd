extends Node3D

## Спайк на снятие главного технического риска: играет ли Godot скиновую
## анимацию из .glb. Модель — CesiumMan из набора Khronos, канонический
## тест на скининг. Скрипт сам себя проверяет и выходит с кодом 0 или 1.

const MODEL := "res://assets/spike/CesiumMan.glb"

func _ready() -> void:
	var scene: PackedScene = load(MODEL)
	if scene == null:
		printerr("СПАЙК ПРОВАЛЕН: файл не загрузился — ", MODEL)
		get_tree().quit(1)
		return

	var model: Node = scene.instantiate()
	add_child(model)

	var player: AnimationPlayer = model.find_child("AnimationPlayer", true, false)
	if player == null:
		printerr("СПАЙК ПРОВАЛЕН: AnimationPlayer не найден в импортированной сцене")
		get_tree().quit(1)
		return

	var names := player.get_animation_list()
	print("Анимаций найдено: ", names.size())
	for n in names:
		print("  - ", n, "  длительность ", player.get_animation(n).length)

	if names.is_empty():
		printerr("СПАЙК ПРОВАЛЕН: клипов нет")
		get_tree().quit(1)
		return

	var skeletons := _find_skeletons(model)
	print("Скелетов найдено: ", skeletons.size())
	for s in skeletons:
		print("  - ", s.name, "  костей ", s.get_bone_count())

	if skeletons.is_empty():
		printerr("СПАЙК ПРОВАЛЕН: Skeleton3D нет — это не скиновая анимация")
		get_tree().quit(1)
		return

	var skel: Skeleton3D = skeletons[0]
	var before: Transform3D = skel.get_bone_pose(0)

	player.play(names[0])
	print("Играет: ", names[0])

	await get_tree().create_timer(0.6).timeout

	var after: Transform3D = skel.get_bone_pose(0)
	if before.is_equal_approx(after):
		printerr("СПАЙК ПРОВАЛЕН: поза кости не изменилась за 0.6 с — анимация не двигает скелет")
		get_tree().quit(1)
		return

	print("СПАЙК ПРОЙДЕН: скиновая анимация glTF работает, кости двигаются")
	get_tree().quit(0)

func _find_skeletons(node: Node) -> Array[Skeleton3D]:
	var out: Array[Skeleton3D] = []
	if node is Skeleton3D:
		out.append(node)
	for c in node.get_children():
		out.append_array(_find_skeletons(c))
	return out
