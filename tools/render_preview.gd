extends SceneTree

## Собирает контактный лист из SVG-слоёв, чтобы посмотреть на арт глазами,
## не запуская игру. Плащ красится умножением — ровно как в игре через modulate.
##
##   engine\Godot_console.exe --headless --path . --script res://tools/render_preview.gd

const CLOAKS := [
	Color(0.78, 0.19, 0.19),
	Color(0.19, 0.34, 0.78),
	Color(0.19, 0.60, 0.31),
	Color(0.83, 0.72, 0.20),
	Color(0.50, 0.28, 0.70),
]
const SCALE := 1.2
const BG := Color(0.10, 0.11, 0.14)

func _initialize() -> void:
	var base := _load_svg("res://assets/art/knight_base.svg")
	var cloak := _load_svg("res://assets/art/knight_cloak.svg")
	if base == null or cloak == null:
		quit(1)
		return

	var w := base.get_width()
	var h := base.get_height()
	var pad := 16
	var sheet := Image.create(pad + (w + pad) * CLOAKS.size(), h + pad * 2, false, Image.FORMAT_RGBA8)
	sheet.fill(BG)

	for i in range(CLOAKS.size()):
		var frame := Image.create(w, h, false, Image.FORMAT_RGBA8)
		frame.fill(Color(0, 0, 0, 0))
		frame.blend_rect(base, Rect2i(0, 0, w, h), Vector2i.ZERO)

		var tinted := _tint(cloak, CLOAKS[i])
		frame.blend_rect(tinted, Rect2i(0, 0, w, h), Vector2i.ZERO)

		sheet.blend_rect(frame, Rect2i(0, 0, w, h), Vector2i(pad + (w + pad) * i, pad))

	var wolf := _load_svg("res://assets/art/wolf.svg")
	if wolf != null:
		sheet.blend_rect(wolf, Rect2i(0, 0, wolf.get_width(), wolf.get_height()),
			Vector2i(pad, h + pad - wolf.get_height()))

	var out := "res://tools/preview_knights.png"
	var err := sheet.save_png(out)
	if err != OK:
		printerr("не сохранилось: ", err)
		quit(1)
		return
	print("готово: %s  (%dx%d)" % [out, sheet.get_width(), sheet.get_height()])
	quit(0)

func _load_svg(path: String) -> Image:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		printerr("не открылся ", path)
		return null
	var img := Image.new()
	var err := img.load_svg_from_string(f.get_as_text(), SCALE)
	if err != OK:
		printerr("SVG не разобрался: ", path, " код ", err)
		return null
	img.convert(Image.FORMAT_RGBA8)
	return img

func _tint(src: Image, c: Color) -> Image:
	var out := Image.create(src.get_width(), src.get_height(), false, Image.FORMAT_RGBA8)
	for y in range(src.get_height()):
		for x in range(src.get_width()):
			var p := src.get_pixel(x, y)
			out.set_pixel(x, y, Color(p.r * c.r, p.g * c.g, p.b * c.b, p.a))
	return out
