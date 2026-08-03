extends Node3D

## Небо, свет, облака и дождь.
##
## Все три величины берутся из мира: `dark` и `rain` — те же числа, что роняют
## уверенность трекера, а доля ночи ведёт солнце по небу. Поэтому картинка
## не иллюстрирует погоду, а показывает ровно то, по чему считается механика.
##
## Мир меняет их скачком раз в ход, поэтому здесь они догоняются плавно:
## глазу нужен переход, а не щелчок.

const CLOUD_BIG := preload("res://assets/hex/nature/cloud_big.gltf")
const CLOUD_SMALL := preload("res://assets/hex/nature/cloud_small.gltf")

const LERP_SPEED := 0.7          # насколько быстро картинка догоняет мир
const CLOUD_COUNT := 26
const CLOUD_SPAN := 190.0        # облака ходят по кругу такой ширины
const CLOUD_DRIFT := 1.9         # метров в секунду

# Палитры неба: закат → глухая ночь. Между ними идёт интерполяция по `dark`.
const SKY_DUSK_TOP := Color(0.16, 0.20, 0.38)
const SKY_DUSK_HORIZON := Color(0.55, 0.38, 0.30)
const SKY_NIGHT_TOP := Color(0.020, 0.030, 0.070)
const SKY_NIGHT_HORIZON := Color(0.055, 0.075, 0.135)

var _dark := 0.5
var _rain := 0.2
var _phase := 0.0                # 0 — начало ночи, 1 — рассвет

var _target_dark := 0.5
var _target_rain := 0.2
var _target_phase := 0.0

var _clouds: Array[Node3D] = []
var _sky: ProceduralSkyMaterial
var _env: Environment
var _sun: DirectionalLight3D
var _rain_fx: GPUParticles3D
var _rain_material: ParticleProcessMaterial

func setup(env: Environment, sun: DirectionalLight3D) -> void:
	_env = env
	_sun = sun

	_sky = ProceduralSkyMaterial.new()
	_sky.sun_angle_max = 30.0
	_sky.sun_curve = 0.2
	var sky := Sky.new()
	sky.sky_material = _sky
	_env.sky = sky
	_env.background_mode = Environment.BG_SKY
	_env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY

	_spawn_clouds()
	_spawn_rain()
	_apply()

## Вызывается миром после каждого хода.
func set_weather(rain: float, dark: float, turn: int, total: int) -> void:
	_target_rain = clampf(rain, 0.0, 1.0)
	_target_dark = clampf(dark, 0.0, 1.0)
	_target_phase = clampf(float(turn) / float(maxi(total, 1)), 0.0, 1.0)

func _process(delta: float) -> void:
	var k: float = clampf(delta * LERP_SPEED, 0.0, 1.0)
	_dark = lerpf(_dark, _target_dark, k)
	_rain = lerpf(_rain, _target_rain, k)
	_phase = lerpf(_phase, _target_phase, k)
	_drift_clouds(delta)
	_apply()

# ── небо и свет ───────────────────────────────────────────────

func _apply() -> void:
	if _sky == null:
		return

	_sky.sky_top_color = SKY_DUSK_TOP.lerp(SKY_NIGHT_TOP, _dark)
	_sky.sky_horizon_color = SKY_DUSK_HORIZON.lerp(SKY_NIGHT_HORIZON, _dark)
	_sky.ground_bottom_color = Color(0.05, 0.06, 0.07)
	_sky.ground_horizon_color = _sky.sky_horizon_color
	# в ливень небо ещё и сереет
	var overcast := Color(0.20, 0.21, 0.24)
	_sky.sky_top_color = _sky.sky_top_color.lerp(overcast, _rain * 0.55)
	_sky.sky_horizon_color = _sky.sky_horizon_color.lerp(overcast, _rain * 0.45)

	_env.ambient_light_energy = lerpf(0.85, 0.22, _dark) * lerpf(1.0, 0.75, _rain)
	_env.fog_light_color = _sky.sky_horizon_color
	_env.fog_density = lerpf(0.010, 0.030, _rain) + _dark * 0.006

	if _sun != null:
		# Светило ползёт по небу за ночь: в начале низко на западе,
		# к рассвету низко на востоке.
		var arc := lerpf(-0.15, PI + 0.15, _phase)
		_sun.rotation = Vector3(-sin(arc) * 0.9 - 0.12, cos(arc) * 0.8 + 2.4, 0.0)
		_sun.light_color = Color(0.95, 0.78, 0.62).lerp(Color(0.52, 0.62, 0.92), _dark)
		_sun.light_energy = lerpf(1.5, 0.42, _dark) * lerpf(1.0, 0.55, _rain)

	if _rain_fx != null:
		_rain_fx.emitting = _rain_visible()
		_rain_fx.amount_ratio = clampf(_rain, 0.05, 1.0)

# ── облака ────────────────────────────────────────────────────

func _spawn_clouds() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = 90210
	for i in range(CLOUD_COUNT):
		var scene: PackedScene = CLOUD_BIG if rng.randf() < 0.55 else CLOUD_SMALL
		var node: Node3D = scene.instantiate()
		# Высоко: на малой высоте облака нависали потолком над самой лавкой.
		node.position = Vector3(
			rng.randf_range(-CLOUD_SPAN, CLOUD_SPAN),
			rng.randf_range(78.0, 118.0),
			rng.randf_range(-170.0, 30.0))
		node.scale = Vector3.ONE * rng.randf_range(13.0, 26.0)
		node.rotation.y = rng.randf() * TAU
		add_child(node)
		_clouds.append(node)

func _drift_clouds(delta: float) -> void:
	# В ливень облака идут быстрее и темнеют — это самый дешёвый способ
	# показать, что погода портится, ещё до того как усилится дождь.
	var speed := CLOUD_DRIFT * lerpf(1.0, 2.6, _rain)
	var tint := Color(0.72, 0.76, 0.86).lerp(Color(0.30, 0.31, 0.37), _rain)
	tint = tint.darkened(_dark * 0.35)

	for c in _clouds:
		c.position.x += speed * delta
		if c.position.x > CLOUD_SPAN:
			c.position.x = -CLOUD_SPAN
		_tint(c, tint)

func _tint(node: Node, c: Color) -> void:
	if node is MeshInstance3D:
		var mi := node as MeshInstance3D
		var mat := mi.material_override as StandardMaterial3D
		if mat == null:
			mat = StandardMaterial3D.new()
			mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
			mi.material_override = mat
		mat.albedo_color = c
	for child in node.get_children():
		_tint(child, c)

# ── дождь ─────────────────────────────────────────────────────

func _spawn_rain() -> void:
	_rain_fx = GPUParticles3D.new()
	_rain_fx.amount = 2400
	_rain_fx.lifetime = 1.4
	_rain_fx.preprocess = 1.4
	_rain_fx.local_coords = false
	_rain_fx.position = Vector3(0.0, 16.0, -8.0)

	var mesh := QuadMesh.new()
	mesh.size = Vector2(0.035, 0.62)
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.albedo_color = Color(0.70, 0.80, 0.95, 0.55)
	mat.billboard_mode = BaseMaterial3D.BILLBOARD_ENABLED
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	mesh.material = mat
	_rain_fx.draw_pass_1 = mesh

	_rain_material = ParticleProcessMaterial.new()
	_rain_material.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_BOX
	_rain_material.emission_box_extents = Vector3(34.0, 1.0, 26.0)
	_rain_material.direction = Vector3(0.16, -1.0, 0.0)
	_rain_material.spread = 2.0
	_rain_material.initial_velocity_min = 17.0
	_rain_material.initial_velocity_max = 22.0
	_rain_material.gravity = Vector3(0.0, -9.0, 0.0)
	_rain_fx.process_material = _rain_material

	add_child(_rain_fx)

func _rain_visible() -> bool:
	return _rain > 0.06
