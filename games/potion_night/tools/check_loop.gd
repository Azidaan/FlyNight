extends SceneTree

## Проверка зацикливания: клип ходьбы длится ~1.07 с, а идти персонаж может
## десяток секунд. Если loop_mode не выставлен, анимация замирает после
## первого прохода и фигура едет застывшей.

func _initialize() -> void:
	var holder := Node3D.new()
	root.add_child(holder)

	var actor := CustomerActor.spawn(1, Color.RED)
	holder.add_child(actor)
	CustomerActor.play_looping(actor, "Walking_A")

	var anim := actor.find_child("AnimationPlayer", true, false) as AnimationPlayer
	var track := anim.get_animation("Walking_A")
	print("длительность клипа: %.2f с" % track.length)
	print("loop_mode: ", track.loop_mode, "  (1 = LOOP_LINEAR)")

	await create_timer(track.length * 2.5).timeout

	if anim.is_playing():
		print("ПРОЙДЕНО: спустя %.1f с клип всё ещё играет" % (track.length * 2.5))
		quit(0)
	else:
		printerr("ПРОВАЛ: клип остановился, фигура будет ехать застывшей")
		quit(1)
