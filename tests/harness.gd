class_name AssertBag
extends RefCounted

## Копилка проваленных проверок. Имя файла намеренно не начинается с "test_" —
## иначе раннер попытался бы выполнить его как набор тестов.

var failures: Array[String] = []
var checks: int = 0

func check(condition: bool, message: String) -> void:
	checks += 1
	if not condition:
		failures.append(message)

func eq(actual: Variant, expected: Variant, message: String) -> void:
	checks += 1
	if actual != expected:
		failures.append("%s — ожидалось %s, получено %s" % [message, str(expected), str(actual)])

func almost(actual: float, expected: float, message: String, eps: float = 0.001) -> void:
	checks += 1
	if absf(actual - expected) > eps:
		failures.append("%s — ожидалось ~%s, получено %s" % [message, str(expected), str(actual)])
