class_name TrackSource
extends RefCounted

## Шов, ради которого выстроена вся конструкция.
##
## FakeTracker считает деградацию по формулам спеки. YoloTracker из проекта 3
## будет принимать кадр вместо SceneView и гонять настоящую сеть. Игра работает
## только с Array[Track] и не знает, откуда он взялся.

func update(_view: SceneView) -> Array[Track]:
	return []
