extends Node3D

## Отрисовка иллюстраций для карт из моделей самой игры.
##
## Смысл: сгенерированная картинка всегда чуть-чуть не та — другой стиль,
## другой оттенок, другая толщина контура. Отрендеренный предмет совпадает
## с игрой идеально, потому что это и есть её предмет.
##
##     engine\Godot.exe --path . tools/render_card_art.tscn
##
## Работа идёт покадровым автоматом, а не цепочкой await: ожидания внутри
## _ready() вешали процесс молча, до единой строчки вывода.

const SIZE := 512
const OUT_DIR := "res://assets/ui/art/"
const SETTLE_FRAMES := 4   # кадров на прогрев после смены предмета

var _jobs := [
	{"scene": preload("res://assets/props/bottle_A_green.glb"),
		"name": "give", "pitch": 0.30, "yaw": 0.9, "zoom": 2.3, "reticle": false},
	{"scene": preload("res://assets/props/chest.glb"),
		"name": "restock", "pitch": 0.55, "yaw": 0.7, "zoom": 2.4, "reticle": false},
	{"scene": preload("res://assets/props/torch_lit.glb"),
		"name": "lantern", "pitch": 0.22, "yaw": 0.8, "zoom": 2.2, "reticle": false},
	{"scene": preload("res://assets/hex/nature/rock_single_C.gltf"),
		"name": "rock", "pitch": 0.55, "yaw": 0.9, "zoom": 2.3, "reticle": false},
	{"scene": preload("res://assets/characters/Rogue_Hooded.glb"),
		"name": "recapture", "pitch": 0.10, "yaw": 0.2, "zoom": 2.5, "reticle": true},
]

var _viewport: SubViewport
var _stage: Node3D
var _camera: Camera3D
var _index := -1
var _wait := 0

func _ready() -> void:
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(OUT_DIR))
	_build()
	print("рендерер готов, предметов: ", _jobs.size())
	_next()

func _build() -> void:
	_viewport = SubViewport.new()
	_viewport.size = Vector2i(SIZE, SIZE)
	_viewport.transparent_bg = true
	_viewport.own_world_3d = true
	_viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(_viewport)

	_stage = Node3D.new()
	_viewport.add_child(_stage)

	_camera = Camera3D.new()
	_viewport.add_child(_camera)

	# Тёплый ключевой сбоку-сверху и холодная заливка с другой стороны:
	# предмет читается объёмным и попадает в ночную палитру игры.
	var key := DirectionalLight3D.new()
	key.rotation = Vector3(-0.7, 0.6, 0.0)
	key.light_color = Color(1.0, 0.86, 0.66)
	key.light_energy = 2.6
	_viewport.add_child(key)

	var fill := DirectionalLight3D.new()
	fill.rotation = Vector3(-0.3, -2.2, 0.0)
	fill.light_color = Color(0.55, 0.66, 0.95)
	fill.light_energy = 1.2
	_viewport.add_child(fill)

	var env := WorldEnvironment.new()
	var e := Environment.new()
	e.background_mode = Environment.BG_CLEAR_COLOR
	e.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	e.ambient_light_color = Color(0.44, 0.48, 0.60)
	e.ambient_light_energy = 1.0
	env.environment = e
	_viewport.add_child(env)

func _next() -> void:
	_index += 1
	if _index >= _jobs.size():
		print("готово, файлы в ", OUT_DIR)
		get_tree().quit(0)
		return

	for c in _stage.get_children():
		_stage.remove_child(c)
		c.queue_free()

	var job: Dictionary = _jobs[_index]
	var node: Node3D = job["scene"].instantiate()
	_stage.add_child(node)

	var anim := node.find_child("AnimationPlayer", true, false) as AnimationPlayer
	if anim != null and anim.has_animation("Idle"):
		anim.play("Idle")

	_wait = SETTLE_FRAMES

func _process(_delta: float) -> void:
	if _index < 0 or _index >= _jobs.size():
		return
	if _wait > 0:
		_wait -= 1
		if _wait == 0:
			_aim()
		return
	_save()
	_next()

func _aim() -> void:
	var job: Dictionary = _jobs[_index]
	var box := _aabb(_stage)
	if box.size.length() < 0.0001:
		printerr("у ", job["name"], " не нашлось геометрии")
		return

	var centre := box.position + box.size * 0.5
	var radius: float = maxf(box.size.length() * 0.5, 0.001)
	var pitch: float = job["pitch"]
	var yaw: float = job["yaw"]
	var dir := Vector3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch))

	_camera.position = centre + dir.normalized() * (radius * float(job["zoom"]))
	_camera.look_at(centre, Vector3.UP)

func _save() -> void:
	var job: Dictionary = _jobs[_index]
	var tex := _viewport.get_texture()
	if tex == null:
		printerr("нет текстуры вьюпорта у ", job["name"])
		return
	var img := tex.get_image()
	if job["reticle"]:
		_draw_reticle(img)
	var path: String = OUT_DIR + str(job["name"]) + ".png"
	var err := img.save_png(path)
	print("%-12s %s" % [job["name"], "ok" if err == OK else "ОШИБКА %d" % err])

## Зелёные уголки прицела поверх фигуры — та самая рамка трекера, но на карте.
func _draw_reticle(img: Image) -> void:
	var green := Color(0.27, 0.87, 0.56)
	var m := int(SIZE * 0.15)
	var arm := int(SIZE * 0.15)
	var w := int(SIZE * 0.02)

	for corner in [Vector2i(m, m), Vector2i(SIZE - m, m),
			Vector2i(m, SIZE - m), Vector2i(SIZE - m, SIZE - m)]:
		var sx := -1 if corner.x > SIZE / 2 else 1
		var sy := -1 if corner.y > SIZE / 2 else 1
		for i in range(arm):
			for t in range(w):
				_dot(img, corner.x + sx * i, corner.y + sy * t, green)
				_dot(img, corner.x + sx * t, corner.y + sy * i, green)

func _dot(img: Image, x: int, y: int, c: Color) -> void:
	if x >= 0 and x < SIZE and y >= 0 and y < SIZE:
		img.set_pixel(x, y, c)

func _aabb(node: Node) -> AABB:
	var out := AABB()
	var first := true
	for mi in _meshes(node):
		if not mi.visible:
			continue
		var box: AABB = mi.global_transform * mi.get_aabb()
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
