extends SceneTree

## Точка входа headless-прогона. Находит все tests/test_*.gd, вызывает у каждого
## run(bag) и выходит с кодом 1, если хоть одна проверка провалилась.
##
##     engine\Godot_console.exe --headless --path . --script res://tests/run_all.gd

func _initialize() -> void:
	var bag := AssertBag.new()
	var dir := DirAccess.open("res://tests")
	if dir == null:
		printerr("Каталог res://tests не открывается")
		quit(1)
		return

	var names: Array[String] = []
	for f in dir.get_files():
		if f.begins_with("test_") and f.ends_with(".gd"):
			names.append(f)
	names.sort()

	if names.is_empty():
		printerr("Наборов тестов не найдено")
		quit(1)
		return

	for n in names:
		var script: GDScript = load("res://tests/" + n)
		if script == null:
			bag.failures.append("не загрузился набор " + n)
			continue
		var suite: RefCounted = script.new()
		suite.run(bag)

	if bag.failures.is_empty():
		print("OK: %d проверок в %d наборах" % [bag.checks, names.size()])
		quit(0)
	else:
		printerr("ПРОВАЛ: %d из %d" % [bag.failures.size(), bag.checks])
		for f in bag.failures:
			printerr("  - " + f)
		quit(1)
