extends Control

## Рамки трекинга: проекция объектов из 3D в экранные прямоугольники.
## Это визуальная суть игры — игрок видит сцену, но опознаёт объекты
## только по тому, что написано на рамке.

const COLOR_LOCKED := Color(0.27, 0.87, 0.56)
const COLOR_DRIFTING := Color(0.88, 0.69, 0.21)
const COLOR_LOST := Color(0.85, 0.32, 0.29)
const COLOR_SELECTED := Color(0.56, 0.83, 1.0)

var camera: Camera3D
var entries: Array = []   # [{pos, size, track}]
var selected_track_id: int = -1

func set_entries(cam: Camera3D, list: Array, sel: int) -> void:
	camera = cam
	entries = list
	selected_track_id = sel
	queue_redraw()

func _draw() -> void:
	if camera == null:
		return

	var font := ThemeDB.fallback_font
	for e in entries:
		var world: Vector3 = e["pos"]
		if camera.is_position_behind(world):
			continue

		var half: Vector3 = e["size"] * 0.5
		var top := camera.unproject_position(world + Vector3(0, half.y, 0))
		var bottom := camera.unproject_position(world - Vector3(0, half.y, 0))
		var side := camera.unproject_position(world + Vector3(half.x, 0, 0))
		var centre := camera.unproject_position(world)

		var h: float = absf(bottom.y - top.y)
		var w: float = maxf(absf(side.x - centre.x) * 2.0, 18.0)
		var rect := Rect2(centre.x - w * 0.5, top.y, w, h)

		var tr: Track = e["track"]
		var col := _colour_for(tr)
		var selected := tr.track_id == selected_track_id
		var width := 3.0 if selected else 1.5

		if tr.state == PN.TrackState.LOST:
			_draw_dashed(rect, col, width)
		else:
			draw_rect(rect, col, false, width)

		# уголки — так рамка читается как прицел, а не как окно
		var tick := minf(w, h) * 0.22
		for corner in [
			[rect.position, Vector2(tick, 0), Vector2(0, tick)],
			[rect.position + Vector2(rect.size.x, 0), Vector2(-tick, 0), Vector2(0, tick)],
			[rect.position + Vector2(0, rect.size.y), Vector2(tick, 0), Vector2(0, -tick)],
			[rect.end, Vector2(-tick, 0), Vector2(0, -tick)],
		]:
			draw_line(corner[0], corner[0] + corner[1], col, width + 1.0)
			draw_line(corner[0], corner[0] + corner[2], col, width + 1.0)

		if selected:
			draw_rect(rect.grow(4.0), COLOR_SELECTED, false, 1.0)

		var label := _label_for(tr)
		var text_pos := Vector2(rect.position.x, rect.position.y - 8.0)
		draw_string(font, text_pos + Vector2(1, 1), label,
			HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color(0, 0, 0, 0.85))
		draw_string(font, text_pos, label,
			HORIZONTAL_ALIGNMENT_LEFT, -1, 13, col)

func _colour_for(tr: Track) -> Color:
	match tr.state:
		PN.TrackState.LOCKED: return COLOR_LOCKED
		PN.TrackState.DRIFTING: return COLOR_DRIFTING
	return COLOR_LOST

func _label_for(tr: Track) -> String:
	if tr.kind == &"wolf":
		return "#%d  ??  %.2f" % [tr.track_id, tr.conf]
	if tr.kind == &"coin":
		return "#%d  монета  %.2f" % [tr.track_id, tr.conf]
	if tr.state == PN.TrackState.LOST:
		return "#%d  ПОТЕРЯН  %.2f" % [tr.track_id, tr.conf]
	return "#%d  %.2f  плащ %s  хочет %s" % [
		tr.track_id, tr.conf,
		PN.cloak_name(tr.remembered_cloak),
		PN.potion_name(tr.remembered_order)]

func _draw_dashed(rect: Rect2, col: Color, width: float) -> void:
	var step := 7.0
	for corner in [
		[rect.position, Vector2(rect.size.x, 0)],
		[rect.position + Vector2(rect.size.x, 0), Vector2(0, rect.size.y)],
		[rect.position + Vector2(0, rect.size.y), Vector2(rect.size.x, 0)],
		[rect.position, Vector2(0, rect.size.y)],
	]:
		var start: Vector2 = corner[0]
		var span: Vector2 = corner[1]
		var length := span.length()
		if length <= 0.0:
			continue
		var dir := span / length
		var t := 0.0
		while t < length:
			var a := start + dir * t
			var b := start + dir * minf(t + step * 0.5, length)
			draw_line(a, b, col, width)
			t += step
