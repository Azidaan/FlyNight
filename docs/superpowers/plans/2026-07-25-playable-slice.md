# Ночная лавка зелий — план: играбельный срез

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Работающая пошаговая игра, проходимая от начала ночи до рассвета, с механикой трекинга — но с кубами вместо моделей и текстовым HUD, чтобы плейтест состоялся до вложений в графику.

**Architecture:** Три слоя. `src/world/` — чистая логика на `RefCounted`, ни одной ноды, тестируется headless. `src/tracking/` — источник треков за интерфейсом `TrackSource`, реализация `FakeTracker` считает деградацию по формулам из спеки. `src/view/` — Godot, подписан на события мира. Слои не знают друг о друге; шов `TrackSource` существует, чтобы позже подменить фейк на настоящий YOLO.

**Tech Stack:** Godot 4.7.1 stable (portable, в `engine/`), GDScript, самописный assert-раннер вместо GUT.

**Спека:** `docs/superpowers/specs/2026-07-25-potion-night-design.md`

---

## Ловушки этого окружения

Читать до начала работы:

1. **`.ps1` с кириллицей требуют UTF-8 с BOM.** PowerShell 5.1 читает файл без BOM как ANSI и падает с ParserError на русских буквах. Проверенное правило проекта. В этом плане скрипты намеренно написаны без кириллицы — не добавляй её без конвертации.
2. **`E:\Dissertation` не является git-репозиторием.** Задача 0 создаёт отдельный репозиторий только для игры.
3. **Диссертационный `.venv` не трогать.** Игра не использует Python вообще.
4. Всё держать внутри `E:\Dissertation\games\potion_night\` — требование переживаемости переустановки Windows.

---

## Структура файлов

| Файл | Ответственность |
|---|---|
| `tools/fetch_godot.ps1` | скачать и разложить portable-движок |
| `tests/harness.gd` | сбор проваленных проверок (`AssertBag`) |
| `tests/run_all.gd` | точка входа headless-прогона, находит `test_*.gd` |
| `src/world/types.gd` | все перечисления проекта (`PN`) |
| `src/world/weather.gd` | погода как функция номера хода |
| `src/world/knight.gd` | рыцарь: заказ, плащ, терпение, слот |
| `src/world/state.gd` | состояние ночи целиком |
| `src/world/events.gd` | событие для аниматора |
| `src/world/actions.gd` | применение карты игрока |
| `src/world/turn.gd` | ход мира |
| `src/tracking/track.gd` | один трек |
| `src/tracking/scene_view.gd` | что видит трекер |
| `src/tracking/track_source.gd` | интерфейс источника треков |
| `src/tracking/fake_tracker.gd` | реализация деградации |
| `src/view/main.tscn` + `main.gd` | уродливый играбельный срез |

---

## Task 0: Каркас проекта и движок

**Files:**
- Create: `E:\Dissertation\games\potion_night\.gitignore`
- Create: `E:\Dissertation\games\potion_night\project.godot`
- Create: `E:\Dissertation\games\potion_night\tools\fetch_godot.ps1`
- Create: `E:\Dissertation\games\potion_night\README.md`

- [ ] **Step 1: Создать репозиторий**

```powershell
cd E:\Dissertation\games\potion_night
git init
git config user.name "Egor"
git config user.email "owamskeizz@gmail.com"
```

Ожидается: `Initialized empty Git repository in E:/Dissertation/games/potion_night/.git/`

- [ ] **Step 2: Написать `.gitignore`**

```gitignore
engine/
.godot/
recordings/
*.tmp
```

`engine/` исключён намеренно: движок весит ~120 МБ и восстанавливается скриптом.

- [ ] **Step 3: Написать `project.godot`**

```ini
config_version=5

[application]
config/name="Ночная лавка зелий"
config/features=PackedStringArray("4.7", "Forward Plus")

[rendering]
renderer/rendering_method="forward_plus"
```

`run/main_scene` намеренно отсутствует — сцены ещё нет, а headless-тестам она не нужна. Добавится в Task 13.

- [ ] **Step 4: Написать `tools/fetch_godot.ps1`**

Без кириллицы — см. ловушку 1.

```powershell
$ErrorActionPreference = 'Stop'
$version = '4.7.1'
$root = Split-Path -Parent $PSScriptRoot
$engine = Join-Path $root 'engine'
$zip = Join-Path $env:TEMP "godot-$version.zip"
$url = "https://github.com/godotengine/godot/releases/download/$version-stable/Godot_v$version-stable_win64.exe.zip"

New-Item -ItemType Directory -Force $engine | Out-Null
Invoke-WebRequest -Uri $url -OutFile $zip
Expand-Archive -Path $zip -DestinationPath $engine -Force
Remove-Item $zip

$exe = Get-ChildItem $engine -Filter 'Godot_v*.exe' | Select-Object -First 1
Move-Item $exe.FullName (Join-Path $engine 'Godot.exe') -Force
& (Join-Path $engine 'Godot.exe') --version
```

Переименование в `Godot.exe` обязательно: иначе пути в тестах привязаны к версии.

- [ ] **Step 5: Запустить и проверить**

Run: `powershell -File tools\fetch_godot.ps1`
Expected: последняя строка вида `4.7.1.stable.official.<hash>`

- [ ] **Step 6: Написать `README.md`**

```markdown
# Ночная лавка зелий

Пошаговая игра про торговлю зельями ночью под дождём. Главная механика — трекинг:
ты видишь сцену, но кто есть кто знаешь только со слов трекера.

## Запуск

    powershell -File tools\fetch_godot.ps1     # один раз, качает движок
    engine\Godot.exe --path .                  # запустить игру
    engine\Godot.exe --headless --path . --script res://tests/run_all.gd

Движок: Godot 4.7.1 stable. Версия зафиксирована, не обновлять без проверки.

Дизайн: `docs/superpowers/specs/2026-07-25-potion-night-design.md`
```

- [ ] **Step 7: Коммит**

```bash
git add .gitignore project.godot tools/fetch_godot.ps1 README.md docs/
git commit -m "chore: каркас проекта, portable Godot 4.7.1"
```

---

## Task 1: Спайк — скиновая анимация .glb

**Это задача на снятие риска, а не на код.** Её результат — ответ «да» или «нет», и она идёт до всей остальной работы. Если анимация не заведётся, раздел 8 спеки пересматривается (процедурная анимация примитивов), но `world/` и `tracking/` от этого не меняются — план продолжает действовать.

**Files:**
- Create: `spike/spike_anim.tscn`, `spike/spike_anim.gd`
- Create: `assets/characters/` (одна модель)

- [ ] **Step 1: Скачать одну модель**

Kenney «Animated Characters», CC0: https://kenney.nl/assets/animated-characters-3
Положить один `.glb` в `assets/characters/knight.glb`.

- [ ] **Step 2: Написать `spike/spike_anim.gd`**

```gdscript
extends Node3D

func _ready() -> void:
	var scene: PackedScene = load("res://assets/characters/knight.glb")
	var model: Node = scene.instantiate()
	add_child(model)

	var player: AnimationPlayer = model.find_child("AnimationPlayer", true, false)
	if player == null:
		printerr("СПАЙК ПРОВАЛЕН: AnimationPlayer не найден")
		get_tree().quit(1)
		return

	var names: PackedStringArray = player.get_animation_list()
	print("Найдено анимаций: ", names.size())
	for n in names:
		print("  - ", n)

	var walk: String = ""
	for n in names:
		if "walk" in n.to_lower():
			walk = n
			break
	if walk == "":
		printerr("СПАЙК ПРОВАЛЕН: клип ходьбы не найден")
		get_tree().quit(1)
		return

	player.play(walk)
	print("СПАЙК ПРОЙДЕН: играет ", walk)
	await get_tree().create_timer(1.0).timeout
	get_tree().quit(0)
```

- [ ] **Step 3: Собрать сцену**

`spike/spike_anim.tscn` — корень `Node3D` со скриптом `spike_anim.gd`, дочерние `Camera3D` (позиция `(0, 1.5, 3)`, смотрит в начало координат) и `DirectionalLight3D`.

- [ ] **Step 4: Запустить**

Run: `engine\Godot.exe --path . spike/spike_anim.tscn`
Expected: `СПАЙК ПРОЙДЕН: играет <имя клипа>`, персонаж на экране шагает.

**Если провал** — остановиться, сообщить пользователю, не продолжать вслепую.

- [ ] **Step 5: Коммит**

```bash
git add spike/ assets/characters/
git commit -m "test: спайк скиновой анимации glTF — риск снят"
```

---

## Task 2: Харнесс тестов

**Files:**
- Create: `tests/harness.gd`
- Create: `tests/run_all.gd`
- Create: `tests/test_harness_selfcheck.gd`

- [ ] **Step 1: Написать провальный тест на сам харнесс**

`tests/test_harness_selfcheck.gd`:

```gdscript
extends RefCounted

func run(t: AssertBag) -> void:
	t.eq(2 + 2, 4, "арифметика жива")
	t.check(true, "check пропускает истину")
	t.almost(0.1 + 0.2, 0.3, "almost терпит плавающую точку")
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `engine\Godot.exe --headless --path . --script res://tests/run_all.gd`
Expected: FAIL — `run_all.gd` не существует.

- [ ] **Step 3: Написать `tests/harness.gd`**

Имя файла намеренно не начинается с `test_` — иначе раннер попытается выполнить его как набор тестов.

```gdscript
class_name AssertBag
extends RefCounted

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
```

- [ ] **Step 4: Написать `tests/run_all.gd`**

```gdscript
extends SceneTree

func _initialize() -> void:
	var bag := AssertBag.new()
	var dir := DirAccess.open("res://tests")
	var names: Array[String] = []
	for f in dir.get_files():
		if f.begins_with("test_") and f.ends_with(".gd"):
			names.append(f)
	names.sort()

	for n in names:
		var suite: RefCounted = load("res://tests/" + n).new()
		suite.run(bag)

	if bag.failures.is_empty():
		print("OK: %d проверок в %d наборах" % [bag.checks, names.size()])
		quit(0)
	else:
		printerr("ПРОВАЛ: %d из %d" % [bag.failures.size(), bag.checks])
		for f in bag.failures:
			printerr("  - " + f)
		quit(1)
```

- [ ] **Step 5: Запустить — убедиться, что проходит**

Run: `engine\Godot.exe --headless --path . --script res://tests/run_all.gd`
Expected: `OK: 3 проверок в 1 наборах`, код возврата 0.

- [ ] **Step 6: Проверить, что харнесс ловит провал**

Временно поменять в `test_harness_selfcheck.gd` строку на `t.eq(2 + 2, 5, "должно упасть")`, прогнать, увидеть `ПРОВАЛ: 1 из 3` и код возврата 1, вернуть обратно.

Это обязательный шаг: раннер, который не умеет падать, хуже отсутствующего.

- [ ] **Step 7: Коммит**

```bash
git add tests/
git commit -m "test: headless assert-раннер"
```

---

## Task 3: Перечисления

**Files:**
- Create: `src/world/types.gd`
- Create: `tests/test_types.gd`

- [ ] **Step 1: Написать провальный тест**

`tests/test_types.gd`:

```gdscript
extends RefCounted

func run(t: AssertBag) -> void:
	t.eq(PN.Potion.size(), 3, "три типа зелий")
	t.eq(PN.TrackState.size(), 3, "три состояния трека")
	t.eq(PN.ActionKind.size(), 5, "пять карт")
	t.eq(PN.potion_name(PN.Potion.HEALING), "лечения", "имя зелья лечения")
	t.eq(PN.cloak_name(PN.Cloak.RED), "красный", "имя красного плаща")
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `engine\Godot.exe --headless --path . --script res://tests/run_all.gd`
Expected: FAIL — `Identifier "PN" not declared`.

- [ ] **Step 3: Написать `src/world/types.gd`**

```gdscript
class_name PN
extends RefCounted

enum Potion { HEALING, STRENGTH, COURAGE }
enum Cloak { RED, BLUE, GREEN, YELLOW, PURPLE }
enum TrackState { LOCKED, DRIFTING, LOST }
enum ActionKind { GIVE, RESTOCK, LANTERN, ROCK, RECAPTURE }

const COUNTER_SLOTS := 4
const QUEUE_SLOTS := 3
const CHEST_PER_TYPE := 6
const NIGHT_TURNS := 30
const MAX_MISSES := 3

const CONF_LOCKED := 0.70
const CONF_DRIFTING := 0.35

static func potion_name(p: int) -> String:
	match p:
		Potion.HEALING: return "лечения"
		Potion.STRENGTH: return "силы"
		Potion.COURAGE: return "храбрости"
	return "?"

static func cloak_name(c: int) -> String:
	match c:
		Cloak.RED: return "красный"
		Cloak.BLUE: return "синий"
		Cloak.GREEN: return "зелёный"
		Cloak.YELLOW: return "жёлтый"
		Cloak.PURPLE: return "фиолетовый"
	return "?"

static func state_from_conf(conf: float) -> int:
	if conf >= CONF_LOCKED:
		return TrackState.LOCKED
	if conf >= CONF_DRIFTING:
		return TrackState.DRIFTING
	return TrackState.LOST
```

Все числа баланса живут здесь — раздел 13 спеки требует, чтобы они правились в одном месте.

- [ ] **Step 4: Запустить — убедиться, что проходит**

Expected: `OK: 8 проверок в 2 наборах`

- [ ] **Step 5: Коммит**

```bash
git add src/world/types.gd tests/test_types.gd
git commit -m "feat: перечисления и константы баланса"
```

---

## Task 4: Погода

**Files:**
- Create: `src/world/weather.gd`
- Create: `tests/test_weather.gd`

- [ ] **Step 1: Написать провальный тест**

`tests/test_weather.gd`:

```gdscript
extends RefCounted

func run(t: AssertBag) -> void:
	var w := Weather.new(0.0)

	t.almost(w.dark_at(0), 0.35, "в начале ночи темнота базовая")
	t.almost(w.dark_at(15), 0.90, "к середине ночи пик темноты")
	t.almost(w.dark_at(30), 0.35, "к рассвету темнота спадает")

	t.check(w.rain_at(15) > w.rain_at(0), "дождь усиливается к середине")
	t.check(w.rain_at(15) <= 1.0, "дождь не превышает единицы")
	t.check(w.rain_at(0) >= 0.0, "дождь не уходит в минус")

	var a := Weather.new(0.1)
	var b := Weather.new(-0.1)
	t.check(a.rain_at(10) != b.rain_at(10), "разный сдвиг даёт разные ночи")
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Expected: FAIL — `Identifier "Weather" not declared`.

- [ ] **Step 3: Написать `src/world/weather.gd`**

```gdscript
class_name Weather
extends RefCounted

var rain_offset: float

func _init(offset: float) -> void:
	rain_offset = offset

func dark_at(turn: int) -> float:
	var t := float(turn) / float(PN.NIGHT_TURNS)
	return clampf(0.35 + 0.55 * sin(PI * t), 0.0, 1.0)

func rain_at(turn: int) -> float:
	var t := float(turn) / float(PN.NIGHT_TURNS)
	var shifted := clampf(t - rain_offset, 0.0, 1.0)
	return clampf(0.15 + 0.75 * sin(PI * shifted), 0.0, 1.0)

static func offset_from_rng(rng: RandomNumberGenerator) -> float:
	return rng.randf_range(-0.15, 0.15)
```

- [ ] **Step 4: Запустить — убедиться, что проходит**

Expected: `OK: 15 проверок в 3 наборах`

- [ ] **Step 5: Коммит**

```bash
git add src/world/weather.gd tests/test_weather.gd
git commit -m "feat: погода как функция номера хода"
```

---

## Task 5: Рыцарь и состояние ночи

**Files:**
- Create: `src/world/knight.gd`, `src/world/state.gd`
- Create: `tests/test_state.gd`

- [ ] **Step 1: Написать провальный тест**

`tests/test_state.gd`:

```gdscript
extends RefCounted

func run(t: AssertBag) -> void:
	var k := Knight.new(7, PN.Potion.HEALING, PN.Cloak.RED, 8)
	t.eq(k.id, 7, "рыцарь помнит свой номер")
	t.eq(k.patience, 8, "терпение задано")
	t.eq(k.slot, -1, "новый рыцарь ещё не в очереди")

	var s := GameState.new(12345)
	t.eq(s.turn, 0, "ночь начинается с нулевого хода")
	t.eq(s.coins, 0, "монет нет")
	t.eq(s.misses, 0, "промахов нет")
	t.eq(s.counter.size(), 0, "прилавок пуст")
	t.eq(s.chest[PN.Potion.HEALING], PN.CHEST_PER_TYPE, "сундук полон")
	t.eq(s.wolf_distance, -1, "волка нет")
	t.eq(s.lantern_turns, 0, "фонарь погашен")
	t.check(not s.finished, "ночь не закончена")

	t.eq(s.free_queue_slot(), 0, "первый слот очереди свободен")
	s.knights.append(Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 8))
	s.knights[0].slot = 0
	t.eq(s.free_queue_slot(), 1, "занятый слот пропускается")

	t.eq(s.knight_in_slot(0).id, 1, "рыцарь находится по слоту")
	t.eq(s.knight_in_slot(2), null, "пустой слот даёт null")

	var s2 := GameState.new(12345)
	t.almost(s2.weather.rain_offset, s.weather.rain_offset, "одинаковый сид — одинаковая погода")
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `engine\Godot.exe --headless --path . --script res://tests/run_all.gd`
Expected: FAIL — `Identifier "Knight" not declared`.

- [ ] **Step 3: Написать `src/world/knight.gd`**

```gdscript
class_name Knight
extends RefCounted

var id: int
var order: int          # PN.Potion
var cloak: int          # PN.Cloak
var patience: int
var slot: int = -1      # -1 = вне очереди

func _init(p_id: int, p_order: int, p_cloak: int, p_patience: int) -> void:
	id = p_id
	order = p_order
	cloak = p_cloak
	patience = p_patience
```

- [ ] **Step 4: Написать `src/world/state.gd`**

```gdscript
class_name GameState
extends RefCounted

var seed_value: int
var rng := RandomNumberGenerator.new()
var weather: Weather

var turn: int = 0
var knights: Array[Knight] = []
var counter: Array[int] = []          # PN.Potion, не длиннее PN.COUNTER_SLOTS
var chest: Dictionary = {}            # PN.Potion -> int
var coins: int = 0
var misses: int = 0
var lantern_turns: int = 0
var wolf_distance: int = -1           # -1 = волка нет
var next_knight_id: int = 1

var finished: bool = false
var outcome: String = ""              # "" | "dawn" | "failed"

func _init(p_seed: int) -> void:
	seed_value = p_seed
	rng.seed = p_seed
	weather = Weather.new(Weather.offset_from_rng(rng))
	for p in [PN.Potion.HEALING, PN.Potion.STRENGTH, PN.Potion.COURAGE]:
		chest[p] = PN.CHEST_PER_TYPE

func rain() -> float:
	return weather.rain_at(turn)

func dark() -> float:
	return weather.dark_at(turn)

func lantern_on() -> bool:
	return lantern_turns > 0

func free_queue_slot() -> int:
	for i in range(PN.QUEUE_SLOTS):
		if knight_in_slot(i) == null:
			return i
	return -1

func knight_in_slot(slot: int) -> Knight:
	for k in knights:
		if k.slot == slot:
			return k
	return null

func knight_by_id(id: int) -> Knight:
	for k in knights:
		if k.id == id:
			return k
	return null

func available_potion_types() -> Array[int]:
	var out: Array[int] = []
	for p in [PN.Potion.HEALING, PN.Potion.STRENGTH, PN.Potion.COURAGE]:
		if chest[p] > 0 or counter.has(p):
			out.append(p)
	return out
```

`available_potion_types()` — то, что делает тупик невозможным по построению (раздел 6 спеки): заказы раздаются только среди этих типов.

- [ ] **Step 5: Запустить — убедиться, что проходит**

Expected: `OK: 29 проверок в 4 наборах`

- [ ] **Step 6: Коммит**

```bash
git add src/world/knight.gd src/world/state.gd tests/test_state.gd
git commit -m "feat: рыцарь и состояние ночи"
```

---

## Task 6: Типы трекинга

**Files:**
- Create: `src/tracking/track.gd`, `src/tracking/scene_object.gd`, `src/tracking/scene_view.gd`, `src/tracking/track_source.gd`
- Create: `tests/test_track_types.gd`

- [ ] **Step 1: Написать провальный тест**

`tests/test_track_types.gd`:

```gdscript
extends RefCounted

func run(t: AssertBag) -> void:
	var tr := Track.new(1, &"knight", 7)
	t.eq(tr.track_id, 1, "трек помнит номер")
	t.eq(tr.kind, &"knight", "трек помнит класс")
	t.eq(tr.object_id, 7, "трек привязан к объекту")
	t.check(not tr.switched, "новый трек не подменён")
	t.eq(tr.remembered_cloak, -1, "признак ещё не зафиксирован")

	tr.conf = 0.9
	tr.refresh_state()
	t.eq(tr.state, PN.TrackState.LOCKED, "0.9 это захват")
	tr.conf = 0.5
	tr.refresh_state()
	t.eq(tr.state, PN.TrackState.DRIFTING, "0.5 это дрейф")
	tr.conf = 0.2
	tr.refresh_state()
	t.eq(tr.state, PN.TrackState.LOST, "0.2 это потеря")

	var o := SceneObject.new(7, &"knight", 0.5, 0.0)
	o.cloak = PN.Cloak.BLUE
	var v := SceneView.new(0.3, 0.8, false)
	v.objects.append(o)
	t.eq(v.objects.size(), 1, "сцена содержит объект")
	t.almost(v.rain, 0.3, "сцена помнит дождь")
	t.check(not v.lantern_on, "фонарь погашен")

	var src := TrackSource.new()
	t.eq(src.update(v).size(), 0, "базовый источник ничего не выдаёт")
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Expected: FAIL — `Identifier "Track" not declared`.

- [ ] **Step 3: Написать `src/tracking/track.gd`**

```gdscript
class_name Track
extends RefCounted

var track_id: int
var kind: StringName            # &"knight" | &"wolf" | &"coin"
var object_id: int              # настоящий id объекта — то, чего игрок не видит
var conf: float = 1.0
var state: int = PN.TrackState.LOCKED
var switched: bool = false
var remembered_cloak: int = -1  # PN.Cloak, зафиксирован при захвате
var last_known_pos: Vector3 = Vector3.ZERO
var age: int = 0

func _init(p_track_id: int, p_kind: StringName, p_object_id: int) -> void:
	track_id = p_track_id
	kind = p_kind
	object_id = p_object_id

func refresh_state() -> void:
	state = PN.state_from_conf(conf)
```

`object_id` — настоящая привязка, `remembered_cloak` — то, что показывается игроку. Их расхождение и есть видимый признак подмены.

- [ ] **Step 4: Написать `src/tracking/scene_object.gd`**

```gdscript
class_name SceneObject
extends RefCounted

var object_id: int
var kind: StringName
var dist_norm: float
var occlusion: float
var cloak: int = -1             # PN.Cloak, только для рыцарей
var slot: int = -1              # позиция в очереди, только для рыцарей
var position: Vector3 = Vector3.ZERO

func _init(p_id: int, p_kind: StringName, p_dist: float, p_occl: float) -> void:
	object_id = p_id
	kind = p_kind
	dist_norm = p_dist
	occlusion = p_occl
```

- [ ] **Step 5: Написать `src/tracking/scene_view.gd`**

```gdscript
class_name SceneView
extends RefCounted

var objects: Array[SceneObject] = []
var rain: float
var dark: float
var lantern_on: bool
var turn: int = 0

func _init(p_rain: float, p_dark: float, p_lantern: bool) -> void:
	rain = p_rain
	dark = p_dark
	lantern_on = p_lantern

static func from_state(s: GameState) -> SceneView:
	var v := SceneView.new(s.rain(), s.dark(), s.lantern_on())
	v.turn = s.turn
	for k in s.knights:
		if k.slot < 0:
			continue
		var dist := float(k.slot) / float(PN.QUEUE_SLOTS - 1)
		var o := SceneObject.new(k.id, &"knight", dist, 0.0)
		o.cloak = k.cloak
		o.slot = k.slot
		v.objects.append(o)
	if s.wolf_distance >= 0:
		var w := SceneObject.new(-1, &"wolf", 1.0, 0.35)
		v.objects.append(w)
	return v
```

Дистанция слотов: слот 0 → 0.0, слот 1 → 0.5, слот 2 → 1.0 — как в таблице раздела 4 спеки.

- [ ] **Step 6: Написать `src/tracking/track_source.gd`**

```gdscript
class_name TrackSource
extends RefCounted

## Базовый источник треков. Возвращает пустой список.
## FakeTracker считает деградацию по формулам; YoloTracker (проект 3)
## будет принимать кадр вместо SceneView. Игра работает только с Array[Track].
func update(_view: SceneView) -> Array[Track]:
	return []
```

- [ ] **Step 7: Запустить — убедиться, что проходит**

Expected: `OK: 42 проверок в 5 наборах`

- [ ] **Step 8: Коммит**

```bash
git add src/tracking/ tests/test_track_types.gd
git commit -m "feat: типы трекинга и интерфейс источника"
```

---

## Task 7: FakeTracker — уверенность

**Files:**
- Create: `src/tracking/fake_tracker.gd`
- Create: `tests/test_fake_tracker.gd`

- [ ] **Step 1: Написать провальный тест**

`tests/test_fake_tracker.gd`:

```gdscript
extends RefCounted

func _clear_view() -> SceneView:
	var v := SceneView.new(0.0, 0.0, false)
	var o := SceneObject.new(1, &"knight", 0.0, 0.0)
	o.cloak = PN.Cloak.RED
	o.slot = 0
	v.objects.append(o)
	return v

func run(t: AssertBag) -> void:
	var ft := FakeTracker.new(999)
	ft.noise_amplitude = 0.0

	var tracks := ft.update(_clear_view())
	t.eq(tracks.size(), 1, "один объект — один трек")
	t.almost(tracks[0].conf, 1.0, "в ясную погоду уверенность полная")
	t.eq(tracks[0].state, PN.TrackState.LOCKED, "ясная погода — захват")
	t.eq(tracks[0].remembered_cloak, PN.Cloak.RED, "признак зафиксирован при захвате")

	var worst := SceneView.new(1.0, 1.0, false)
	var wo := SceneObject.new(1, &"knight", 1.0, 0.0)
	wo.cloak = PN.Cloak.RED
	wo.slot = 2
	worst.objects.append(wo)
	var ft2 := FakeTracker.new(999)
	ft2.noise_amplitude = 0.0
	var bad := ft2.update(worst)
	t.almost(bad[0].conf, 0.15, "худшие условия роняют уверенность до 0.15")
	t.eq(bad[0].state, PN.TrackState.LOST, "худшие условия — потеря")

	var lit := SceneView.new(0.0, 1.0, true)
	var lo := SceneObject.new(1, &"knight", 0.0, 0.0)
	lo.cloak = PN.Cloak.RED
	lo.slot = 0
	lit.objects.append(lo)
	var ft3 := FakeTracker.new(999)
	ft3.noise_amplitude = 0.0
	var okk := ft3.update(lit)
	t.almost(okk[0].conf, 0.9125, "фонарь гасит темноту вчетверо")

	var a := FakeTracker.new(42).update(_clear_view())
	var b := FakeTracker.new(42).update(_clear_view())
	t.almost(a[0].conf, b[0].conf, "одинаковый сид воспроизводим")

	var ids := FakeTracker.new(1)
	var first := ids.update(_clear_view())
	var second := ids.update(_clear_view())
	t.eq(first[0].track_id, second[0].track_id, "трек сохраняет номер между ходами")
	t.eq(second[0].age, 1, "возраст трека растёт")
```

Арифметика проверочных чисел: худшие условия `1.0 − 0.30·1 − 0.35·1 − 0.20·1 − 0.25·0 = 0.15`. Фонарь: `1.0 − 0 − 0.35·1·0.25 − 0 − 0 = 0.9125`.

- [ ] **Step 2: Запустить — убедиться, что падает**

Expected: FAIL — `Identifier "FakeTracker" not declared`.

- [ ] **Step 3: Написать `src/tracking/fake_tracker.gd`**

```gdscript
class_name FakeTracker
extends TrackSource

var noise_amplitude: float = 0.05
var _seed: int
var _next_track_id: int = 1
var _by_object: Dictionary = {}   # object_id -> Track

func _init(p_seed: int) -> void:
	_seed = p_seed

func update(view: SceneView) -> Array[Track]:
	var out: Array[Track] = []
	var seen: Dictionary = {}

	for o in view.objects:
		seen[o.object_id] = true
		var tr: Track = _by_object.get(o.object_id)
		if tr == null:
			tr = Track.new(_next_track_id, o.kind, o.object_id)
			tr.remembered_cloak = o.cloak
			_next_track_id += 1
			_by_object[o.object_id] = tr
		else:
			tr.age += 1

		tr.conf = confidence(view, o) + _noise(view.turn, tr.track_id)
		tr.conf = clampf(tr.conf, 0.0, 1.0)
		tr.refresh_state()
		if tr.state != PN.TrackState.LOST:
			tr.last_known_pos = o.position
		out.append(tr)

	for key in _by_object.keys():
		if not seen.has(key):
			_by_object.erase(key)

	return out

func confidence(view: SceneView, o: SceneObject) -> float:
	var dark_factor := 0.25 if view.lantern_on else 1.0
	var c := 1.0
	c -= 0.30 * view.rain
	c -= 0.35 * view.dark * dark_factor
	c -= 0.20 * o.dist_norm
	c -= 0.25 * o.occlusion
	return clampf(c, 0.0, 1.0)

func _noise(turn: int, track_id: int) -> float:
	if noise_amplitude <= 0.0:
		return 0.0
	var h := absi(hash("%d:%d:%d" % [_seed, turn, track_id]))
	return (float(h % 1000) / 1000.0) * (2.0 * noise_amplitude) - noise_amplitude
```

`noise_amplitude = 0.0` в тестах убирает шум и делает арифметику проверяемой точно — иначе формулу пришлось бы проверять неравенствами.

- [ ] **Step 4: Запустить — убедиться, что проходит**

Expected: `OK: 53 проверок в 6 наборах`

- [ ] **Step 5: Коммит**

```bash
git add src/tracking/fake_tracker.gd tests/test_fake_tracker.gd
git commit -m "feat: FakeTracker — уверенность и состояния треков"
```

---

## Task 8: FakeTracker — подмена ID

**Files:**
- Modify: `src/tracking/fake_tracker.gd`
- Modify: `tests/test_fake_tracker.gd`

- [ ] **Step 1: Дописать провальный тест**

Добавить в `tests/test_fake_tracker.gd` в конец `run()`:

```gdscript
	# чистая функция решения — проверяется без RNG
	t.check(not FakeTracker.should_switch(0.8, 0.5, 0.0), "уверенный трек не подменяется")
	t.check(not FakeTracker.should_switch(0.5, 0.7, 0.0), "хватает одного уверенного")
	t.check(not FakeTracker.should_switch(0.2, 0.5, 0.0), "потерянный трек не подменяется")
	t.check(not FakeTracker.should_switch(0.5, 0.2, 0.0), "потерянный второй тоже")
	t.check(FakeTracker.should_switch(0.4, 0.5, 0.0), "нулевой бросок всегда проходит")
	t.check(not FakeTracker.should_switch(0.4, 0.5, 0.99), "высокий бросок не проходит")

	# p = 0.25 * (1 - min(0.4, 0.5)) = 0.15
	t.check(FakeTracker.should_switch(0.4, 0.5, 0.14), "бросок ниже порога проходит")
	t.check(not FakeTracker.should_switch(0.4, 0.5, 0.16), "бросок выше порога не проходит")

	# подмена меняет привязку, но не признак
	var pair := SceneView.new(0.9, 0.9, false)
	for i in range(2):
		var o := SceneObject.new(i + 1, &"knight", 0.5, 0.0)
		o.cloak = PN.Cloak.RED if i == 0 else PN.Cloak.BLUE
		o.slot = i
		pair.objects.append(o)
	var ft := FakeTracker.new(7)
	ft.noise_amplitude = 0.0
	ft.update(pair)
	var before_cloaks := [ft.track_for(1).remembered_cloak, ft.track_for(2).remembered_cloak]
	ft.force_switch(1, 2)
	t.eq(ft.track_for(1).object_id, 2, "первый трек теперь указывает на второго рыцаря")
	t.eq(ft.track_for(2).object_id, 1, "и наоборот")
	t.eq(ft.track_for(1).remembered_cloak, before_cloaks[0], "признак остался прежним — в этом вся подлость")
	t.check(ft.track_for(1).switched, "флаг подмены выставлен")

	ft.recapture(pair)
	t.eq(ft.track_for(1).object_id, 1, "перезахват чинит привязку")
	t.check(not ft.track_for(1).switched, "перезахват снимает флаг")
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Expected: FAIL — `Invalid call. Nonexistent function 'should_switch'`.

- [ ] **Step 3: Дописать `src/tracking/fake_tracker.gd`**

Добавить в конец файла:

```gdscript
## Чистое решение о подмене. Вынесено отдельно, чтобы проверяться без RNG.
static func should_switch(conf_a: float, conf_b: float, roll: float) -> bool:
	if conf_a >= 0.60 or conf_b >= 0.60:
		return false
	if conf_a < PN.CONF_DRIFTING or conf_b < PN.CONF_DRIFTING:
		return false
	return roll < 0.25 * (1.0 - minf(conf_a, conf_b))

func track_for(object_id: int) -> Track:
	for tr in _by_object.values():
		if tr.object_id == object_id:
			return tr
	return null

## Меняет местами привязку двух треков, сохраняя запомненные признаки.
func force_switch(object_a: int, object_b: int) -> void:
	var ta := track_for(object_a)
	var tb := track_for(object_b)
	if ta == null or tb == null:
		return
	var tmp := ta.object_id
	ta.object_id = tb.object_id
	tb.object_id = tmp
	ta.switched = not ta.switched
	tb.switched = not tb.switched
	_by_object[ta.object_id] = ta
	_by_object[tb.object_id] = tb

## Пересопоставляет треки с объектами заново и снимает все подмены.
func recapture(view: SceneView) -> void:
	_by_object.clear()
	for o in view.objects:
		var tr := Track.new(_next_track_id, o.kind, o.object_id)
		tr.remembered_cloak = o.cloak
		_next_track_id += 1
		_by_object[o.object_id] = tr

## Прогоняет подмены между соседями по очереди. Вызывается ходом мира.
func roll_switches(view: SceneView, rng: RandomNumberGenerator) -> void:
	for slot in range(PN.QUEUE_SLOTS - 1):
		var a := _object_in_slot(view, slot)
		var b := _object_in_slot(view, slot + 1)
		if a < 0 or b < 0:
			continue
		var ta := track_for(a)
		var tb := track_for(b)
		if ta == null or tb == null:
			continue
		if should_switch(ta.conf, tb.conf, rng.randf()):
			force_switch(a, b)

func _object_in_slot(view: SceneView, slot: int) -> int:
	for o in view.objects:
		if o.kind == &"knight" and o.slot == slot:
			return o.object_id
	return -1
```

`roll_switches` перебирает только пары соседних слотов — инвариант «подмена только между соседями» держится структурой цикла, а не проверкой.

- [ ] **Step 4: Запустить — убедиться, что проходит**

Expected: `OK: 68 проверок в 6 наборах`

- [ ] **Step 5: Коммит**

```bash
git add src/tracking/fake_tracker.gd tests/test_fake_tracker.gd
git commit -m "feat: подмена ID между соседями по очереди"
```

---

## Task 9: События, результат действия и карта «Из сундука»

**Files:**
- Create: `src/world/events.gd`, `src/world/action_result.gd`, `src/world/actions.gd`
- Create: `tests/test_actions.gd`

- [ ] **Step 1: Написать провальный тест**

`tests/test_actions.gd`:

```gdscript
extends RefCounted

func run(t: AssertBag) -> void:
	var e := GameEvent.new(&"POTION_PLACED", {"potion": PN.Potion.HEALING})
	t.eq(e.kind, &"POTION_PLACED", "событие помнит вид")
	t.eq(e.data["potion"], PN.Potion.HEALING, "событие несёт данные")

	var bad := ActionResult.fail("нет места")
	t.check(not bad.ok, "провальный результат помечен")
	t.eq(bad.reason, "нет места", "причина сохранена")

	var s := GameState.new(1)
	var r := Actions.restock(s, PN.Potion.HEALING)
	t.check(r.ok, "пополнение с полным сундуком проходит")
	t.eq(s.counter.size(), 1, "зелье легло на прилавок")
	t.eq(s.chest[PN.Potion.HEALING], PN.CHEST_PER_TYPE - 1, "сундук уменьшился")
	t.eq(r.events.size(), 1, "породилось одно событие")
	t.eq(r.events[0].kind, &"POTION_PLACED", "событие про выкладку")

	for i in range(3):
		Actions.restock(s, PN.Potion.HEALING)
	t.eq(s.counter.size(), PN.COUNTER_SLOTS, "прилавок заполнен")
	var full := Actions.restock(s, PN.Potion.HEALING)
	t.check(not full.ok, "на полный прилавок не положить")
	t.eq(s.counter.size(), PN.COUNTER_SLOTS, "состояние не изменилось при отказе")

	var s2 := GameState.new(2)
	s2.chest[PN.Potion.STRENGTH] = 0
	var empty := Actions.restock(s2, PN.Potion.STRENGTH)
	t.check(not empty.ok, "из пустого сундука не достать")
	t.eq(s2.counter.size(), 0, "прилавок не тронут")
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Expected: FAIL — `Identifier "GameEvent" not declared`.

- [ ] **Step 3: Написать `src/world/events.gd`**

```gdscript
class_name GameEvent
extends RefCounted

var kind: StringName
var data: Dictionary

func _init(p_kind: StringName, p_data: Dictionary = {}) -> void:
	kind = p_kind
	data = p_data
```

Событие намеренно нетипизировано по содержимому: `view` разбирает его по `kind`, а мир не должен знать, как это рисуется.

- [ ] **Step 4: Написать `src/world/action_result.gd`**

```gdscript
class_name ActionResult
extends RefCounted

var ok: bool = true
var reason: String = ""
var events: Array[GameEvent] = []

static func fail(p_reason: String) -> ActionResult:
	var r := ActionResult.new()
	r.ok = false
	r.reason = p_reason
	return r

func emit(kind: StringName, data: Dictionary = {}) -> void:
	events.append(GameEvent.new(kind, data))
```

- [ ] **Step 5: Написать `src/world/actions.gd`**

```gdscript
class_name Actions
extends RefCounted

static func restock(s: GameState, potion: int) -> ActionResult:
	if s.counter.size() >= PN.COUNTER_SLOTS:
		return ActionResult.fail("прилавок полон")
	if s.chest.get(potion, 0) <= 0:
		return ActionResult.fail("в сундуке пусто")

	s.chest[potion] -= 1
	s.counter.append(potion)

	var r := ActionResult.new()
	r.emit(&"POTION_PLACED", {"potion": potion, "slot": s.counter.size() - 1})
	return r
```

Отказ не меняет состояние — это проверяется тестом отдельно, потому что частично применённое действие сломало бы детерминизм.

- [ ] **Step 6: Запустить — убедиться, что проходит**

Expected: `OK: 82 проверок в 7 наборах`

- [ ] **Step 7: Коммит**

```bash
git add src/world/events.gd src/world/action_result.gd src/world/actions.gd tests/test_actions.gd
git commit -m "feat: события, результат действия, карта пополнения"
```

---

## Task 10: Карта «Отдать»

Здесь механика подмены впервые кусает игрока: цель выбирается **по номеру трека**, а трек может указывать не на того рыцаря.

**Files:**
- Modify: `src/world/actions.gd`
- Modify: `tests/test_actions.gd`

- [ ] **Step 1: Дописать провальный тест**

Добавить в конец `run()` в `tests/test_actions.gd`:

```gdscript
	# Сцена: два рыцаря, оба хотят разное
	var g := GameState.new(3)
	var k1 := Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 8)
	k1.slot = 0
	var k2 := Knight.new(2, PN.Potion.STRENGTH, PN.Cloak.BLUE, 8)
	k2.slot = 1
	g.knights.append_array([k1, k2])
	g.counter.append_array([PN.Potion.HEALING, PN.Potion.STRENGTH])

	var ft := FakeTracker.new(5)
	ft.noise_amplitude = 0.0
	var tracks := ft.update(SceneView.from_state(g))
	var t1: int = ft.track_for(1).track_id

	# верное зелье верному рыцарю
	var good := Actions.give(g, tracks, t1, 0)
	t.check(good.ok, "выдача проходит")
	t.eq(g.knights.size(), 1, "довольный рыцарь ушёл")
	t.eq(g.counter.size(), 1, "зелье списано с прилавка")
	t.check(_has(good.events, &"COIN_THROWN"), "довольный бросает монету")

	# не то зелье
	var g2 := GameState.new(4)
	var k3 := Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 8)
	k3.slot = 0
	g2.knights.append(k3)
	g2.counter.append(PN.Potion.COURAGE)
	var ft2 := FakeTracker.new(5)
	ft2.noise_amplitude = 0.0
	var tr2 := ft2.update(SceneView.from_state(g2))
	var wrong := Actions.give(g2, tr2, tr2[0].track_id, 0)
	t.check(wrong.ok, "выдача не того зелья всё равно тратит ход")
	t.eq(g2.knights[0].patience, 5, "терпение упало на три")
	t.eq(g2.counter.size(), 0, "зелье потрачено впустую")
	t.check(_has(wrong.events, &"POTION_REFUSED"), "породилось событие отказа")

	# подмена уводит зелье не тому
	var g3 := GameState.new(5)
	var a := Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 8)
	a.slot = 0
	var b := Knight.new(2, PN.Potion.STRENGTH, PN.Cloak.BLUE, 8)
	b.slot = 1
	g3.knights.append_array([a, b])
	g3.counter.append(PN.Potion.HEALING)
	var ft3 := FakeTracker.new(5)
	ft3.noise_amplitude = 0.0
	var tr3 := ft3.update(SceneView.from_state(g3))
	var id_a: int = ft3.track_for(1).track_id
	ft3.force_switch(1, 2)
	var swapped := Actions.give(g3, tr3, id_a, 0)
	t.check(swapped.ok, "выдача по подменённому треку проходит")
	t.eq(g3.knight_by_id(1).patience, 8, "первый рыцарь ничего не получил")
	t.eq(g3.knight_by_id(2).patience, 5, "зелье ушло второму и не подошло ему")

	# несуществующий трек
	var nores := Actions.give(g3, tr3, 9999, 0)
	t.check(not nores.ok, "по чужому номеру трека выдать нельзя")

func _has(events: Array, kind: StringName) -> bool:
	for e in events:
		if e.kind == kind:
			return true
	return false
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Expected: FAIL — `Nonexistent function 'give'`.

- [ ] **Step 3: Дописать `src/world/actions.gd`**

```gdscript
static func give(s: GameState, tracks: Array[Track], track_id: int, counter_index: int) -> ActionResult:
	if counter_index < 0 or counter_index >= s.counter.size():
		return ActionResult.fail("на прилавке нет такого зелья")

	var target: Track = null
	for tr in tracks:
		if tr.track_id == track_id:
			target = tr
			break
	if target == null or target.kind != &"knight":
		return ActionResult.fail("нет такого трека")

	# Вот он, укус подмены: трек ведёт к object_id, а не к тому,
	# кого игрок видел под этой рамкой.
	var knight := s.knight_by_id(target.object_id)
	if knight == null:
		return ActionResult.fail("трек ведёт в пустоту")

	var potion: int = s.counter[counter_index]
	s.counter.remove_at(counter_index)

	var r := ActionResult.new()
	if potion == knight.order:
		s.knights.erase(knight)
		r.emit(&"POTION_HANDED", {"knight": knight.id, "potion": potion})
		r.emit(&"KNIGHT_LEAVES_HAPPY", {"knight": knight.id, "slot": knight.slot})
		r.emit(&"COIN_THROWN", {"knight": knight.id})
	else:
		knight.patience = maxi(0, knight.patience - 3)
		r.emit(&"POTION_REFUSED", {"knight": knight.id, "potion": potion})
	return r
```

Монета не начисляется здесь: она становится объектом со своим треком и подбирается ходом мира — раздел 5 спеки. Начислять сразу значило бы, что трекинг не влияет на доход.

- [ ] **Step 4: Запустить — убедиться, что проходит**

Expected: `OK: 94 проверок в 7 наборах`

- [ ] **Step 5: Коммит**

```bash
git add src/world/actions.gd tests/test_actions.gd
git commit -m "feat: карта выдачи зелья, цель по номеру трека"
```

---

## Task 11: Карты «Фонарь», «Камень», «Перезахват»

**Files:**
- Modify: `src/world/actions.gd`
- Modify: `tests/test_actions.gd`

- [ ] **Step 1: Дописать провальный тест**

Добавить в конец `run()` (перед `_has`):

```gdscript
	var g4 := GameState.new(6)
	var lr := Actions.lantern(g4)
	t.check(lr.ok, "фонарь зажигается")
	t.eq(g4.lantern_turns, 3, "горит три хода")
	t.check(g4.lantern_on(), "фонарь считается горящим")
	t.check(_has(lr.events, &"LANTERN_LIT"), "событие поджига")
	t.check(not Actions.lantern(g4).ok, "горящий фонарь не поджечь снова")

	# камень: волка нет
	var g5 := GameState.new(7)
	t.check(not Actions.rock(g5, [] as Array[Track]).ok, "без волка камень не бросить")

	# камень: волк есть, трек держится
	g5.wolf_distance = 3
	var wt := Track.new(50, &"wolf", -1)
	wt.conf = 0.5
	wt.refresh_state()
	var hit := Actions.rock(g5, [wt] as Array[Track])
	t.check(hit.ok, "по дрейфующему треку попадание засчитано")
	t.eq(g5.wolf_distance, -1, "волк убежал")
	t.check(_has(hit.events, &"WOLF_FLEES"), "событие бегства")

	# камень: трек потерян — промах, ход потрачен
	var g6 := GameState.new(8)
	g6.wolf_distance = 2
	var lost := Track.new(51, &"wolf", -1)
	lost.conf = 0.1
	lost.refresh_state()
	var miss := Actions.rock(g6, [lost] as Array[Track])
	t.check(miss.ok, "промах — тоже потраченный ход, а не отказ")
	t.eq(g6.wolf_distance, 2, "волк на месте")
	t.check(_has(miss.events, &"ROCK_MISSED"), "событие промаха")

	# перезахват
	var g7 := GameState.new(9)
	var k7 := Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 8)
	k7.slot = 0
	g7.knights.append(k7)
	var ft7 := FakeTracker.new(11)
	ft7.noise_amplitude = 0.0
	ft7.update(SceneView.from_state(g7))
	ft7.force_switch(1, 1)
	var rec := Actions.recapture(g7, ft7)
	t.check(rec.ok, "перезахват проходит")
	t.check(not ft7.track_for(1).switched, "подмены сняты")
	t.check(_has(rec.events, &"TRACKER_RESET"), "событие сброса")
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Expected: FAIL — `Nonexistent function 'lantern'`.

- [ ] **Step 3: Дописать `src/world/actions.gd`**

```gdscript
const LANTERN_TURNS := 3

static func lantern(s: GameState) -> ActionResult:
	if s.lantern_on():
		return ActionResult.fail("фонарь уже горит")
	s.lantern_turns = LANTERN_TURNS
	var r := ActionResult.new()
	r.emit(&"LANTERN_LIT", {"turns": LANTERN_TURNS})
	return r

static func rock(s: GameState, tracks: Array[Track]) -> ActionResult:
	if s.wolf_distance < 0:
		return ActionResult.fail("волка не видно")

	var wolf: Track = null
	for tr in tracks:
		if tr.kind == &"wolf":
			wolf = tr
			break

	var r := ActionResult.new()
	r.emit(&"ROCK_THROWN", {})
	if wolf == null or wolf.state == PN.TrackState.LOST:
		r.emit(&"ROCK_MISSED", {})
		return r

	s.wolf_distance = -1
	r.emit(&"WOLF_FLEES", {})
	return r

static func recapture(s: GameState, tracker: FakeTracker) -> ActionResult:
	tracker.recapture(SceneView.from_state(s))
	var r := ActionResult.new()
	r.emit(&"TRACKER_RESET", {})
	return r
```

Промах возвращает `ok = true`: ход потрачен, это и есть наказание. Отказ (`ok = false`) означал бы «ход не засчитан» и позволял бы кидать камень бесплатно, пока не попадёшь.

- [ ] **Step 4: Запустить — убедиться, что проходит**

Expected: `OK: 111 проверок в 7 наборах`

- [ ] **Step 5: Коммит**

```bash
git add src/world/actions.gd tests/test_actions.gd
git commit -m "feat: карты фонаря, камня и перезахвата"
```

---

## Task 12: Монеты в полёте

Монета не начисляется в момент выдачи: она живёт объектом со своим треком один ход. Если трек потерян — монета укатилась. Так трекинг влияет на доход напрямую.

**Files:**
- Modify: `src/world/state.gd`, `src/tracking/scene_view.gd`, `src/world/actions.gd`
- Create: `tests/test_coins.gd`

- [ ] **Step 1: Написать провальный тест**

`tests/test_coins.gd`:

```gdscript
extends RefCounted

func run(t: AssertBag) -> void:
	var s := GameState.new(21)
	s.coins_in_flight.append(101)
	var v := SceneView.from_state(s)
	var found := false
	for o in v.objects:
		if o.kind == &"coin":
			found = true
			t.eq(o.object_id, 101, "монета попала в сцену со своим номером")
			t.almost(o.occlusion, 0.30, "монета частично перекрыта прилавком")
			t.almost(o.dist_norm, 0.0, "монета падает у самого прилавка")
	t.check(found, "монета видна трекеру")

	# монета в ясную погоду подбирается
	var clear := GameState.new(22)
	clear.coins_in_flight.append(7)
	var ft := FakeTracker.new(1)
	ft.noise_amplitude = 0.0
	var tracks := ft.update(SceneView.from_state(clear))
	var r := ActionResult.new()
	Turn.resolve_coins(clear, tracks, r)
	t.eq(clear.coins, 1, "монета зачтена")
	t.eq(clear.coins_in_flight.size(), 0, "список очищен")

	# потерянный трек — монета укатилась
	var dark_state := GameState.new(23)
	dark_state.coins_in_flight.append(8)
	var lost := Track.new(1, &"coin", 8)
	lost.conf = 0.1
	lost.refresh_state()
	var r2 := ActionResult.new()
	Turn.resolve_coins(dark_state, [lost] as Array[Track], r2)
	t.eq(dark_state.coins, 0, "монеты нет")
	t.eq(dark_state.coins_in_flight.size(), 0, "но список всё равно очищен")
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Expected: FAIL — `Invalid access to property 'coins_in_flight'`.

- [ ] **Step 3: Добавить поле в `src/world/state.gd`**

После `var next_knight_id: int = 1` добавить:

```gdscript
var coins_in_flight: Array[int] = []
var next_coin_id: int = 100
```

- [ ] **Step 4: Учесть монеты в `src/tracking/scene_view.gd`**

В `from_state()` перед `return v` добавить:

```gdscript
	for coin_id in s.coins_in_flight:
		v.objects.append(SceneObject.new(coin_id, &"coin", 0.0, 0.30))
```

- [ ] **Step 5: Породить монету в `src/world/actions.gd`**

В `give()` заменить строку `r.emit(&"COIN_THROWN", {"knight": knight.id})` на:

```gdscript
		var coin_id := s.next_coin_id
		s.next_coin_id += 1
		s.coins_in_flight.append(coin_id)
		r.emit(&"COIN_THROWN", {"knight": knight.id, "coin": coin_id})
```

- [ ] **Step 6: Написать `src/world/turn.gd` с одной функцией**

```gdscript
class_name Turn
extends RefCounted

static func resolve_coins(s: GameState, tracks: Array[Track], r: ActionResult) -> void:
	for coin_id in s.coins_in_flight:
		var found: Track = null
		for tr in tracks:
			if tr.kind == &"coin" and tr.object_id == coin_id:
				found = tr
				break
		if found != null and found.state != PN.TrackState.LOST:
			s.coins += 1
			r.emit(&"COIN_COLLECTED", {"coin": coin_id})
		else:
			r.emit(&"COIN_LOST", {"coin": coin_id})
	s.coins_in_flight.clear()
```

- [ ] **Step 7: Запустить — убедиться, что проходит**

Expected: `OK: 121 проверок в 8 наборах`

- [ ] **Step 8: Коммит**

```bash
git add src/world/state.gd src/world/turn.gd src/world/actions.gd src/tracking/scene_view.gd tests/test_coins.gd
git commit -m "feat: монеты подбираются только при живом треке"
```

---

## Task 13: Ход мира и конец ночи

**Files:**
- Modify: `src/world/turn.gd`
- Create: `tests/test_turn.gd`

- [ ] **Step 1: Написать провальный тест**

`tests/test_turn.gd`:

```gdscript
extends RefCounted

func _tracker() -> FakeTracker:
	var ft := FakeTracker.new(77)
	ft.noise_amplitude = 0.0
	return ft

func run(t: AssertBag) -> void:
	var s := GameState.new(31)
	var k := Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 2)
	k.slot = 0
	s.knights.append(k)

	Turn.world_turn(s, _tracker())
	t.eq(s.turn, 1, "номер хода вырос")
	t.eq(s.knights[0].patience, 1, "терпение упало")

	var r := Turn.world_turn(s, _tracker())
	t.eq(s.knights.size(), 0, "терпение кончилось, рыцарь ушёл")
	t.eq(s.misses, 1, "засчитан промах")
	t.check(_has(r.events, &"KNIGHT_LEAVES_ANGRY"), "событие ухода")

	# фонарь догорает
	var s2 := GameState.new(32)
	s2.lantern_turns = 1
	Turn.world_turn(s2, _tracker())
	t.eq(s2.lantern_turns, 0, "фонарь погас")

	# волк подходит, вдвое быстрее при фонаре
	var s3 := GameState.new(33)
	s3.wolf_distance = 4
	Turn.world_turn(s3, _tracker())
	t.eq(s3.wolf_distance, 3, "волк сделал шаг")
	s3.lantern_turns = 2
	Turn.world_turn(s3, _tracker())
	t.eq(s3.wolf_distance, 1, "при фонаре волк шагает вдвое")

	# волк дошёл и крадёт
	var s4 := GameState.new(34)
	s4.wolf_distance = 1
	s4.counter.append(PN.Potion.HEALING)
	var r4 := Turn.world_turn(s4, _tracker())
	t.eq(s4.counter.size(), 0, "зелье украдено")
	t.eq(s4.wolf_distance, -1, "волк убежал с добычей")
	t.check(_has(r4.events, &"WOLF_STEALS"), "событие кражи")

	# провал по трём промахам
	var s5 := GameState.new(35)
	s5.misses = 2
	var k5 := Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 1)
	k5.slot = 0
	s5.knights.append(k5)
	var r5 := Turn.world_turn(s5, _tracker())
	t.eq(s5.misses, 3, "третий промах")
	t.check(s5.finished, "ночь закончена")
	t.eq(s5.outcome, "failed", "исход — провал")
	t.check(_has(r5.events, &"NIGHT_FAILED"), "событие провала")

	# рассвет
	var s6 := GameState.new(36)
	s6.turn = PN.NIGHT_TURNS - 1
	var r6 := Turn.world_turn(s6, _tracker())
	t.check(s6.finished, "ночь закончена по времени")
	t.eq(s6.outcome, "dawn", "исход — рассвет")
	t.check(_has(r6.events, &"DAWN"), "событие рассвета")

	# законченная ночь не тикает дальше
	var before := s6.turn
	Turn.world_turn(s6, _tracker())
	t.eq(s6.turn, before, "после конца ход не идёт")

	# очередь подтягивается к прилавку
	var s7 := GameState.new(37)
	var far := Knight.new(1, PN.Potion.HEALING, PN.Cloak.RED, 9)
	far.slot = 2
	s7.knights.append(far)
	Turn.world_turn(s7, _tracker())
	t.eq(s7.knights[0].slot, 0, "рыцарь подошёл к прилавку")

	# заказы только из доступного
	var s8 := GameState.new(38)
	s8.chest[PN.Potion.STRENGTH] = 0
	s8.chest[PN.Potion.COURAGE] = 0
	for i in range(20):
		Turn.world_turn(s8, _tracker())
	for kk in s8.knights:
		t.eq(kk.order, PN.Potion.HEALING, "заказ только из того, что есть")

	# детерминизм
	var a := GameState.new(4242)
	var b := GameState.new(4242)
	for i in range(10):
		Turn.world_turn(a, FakeTracker.new(1))
		Turn.world_turn(b, FakeTracker.new(1))
	t.eq(a.knights.size(), b.knights.size(), "одинаковый сид — одинаковая ночь")
	t.eq(a.wolf_distance, b.wolf_distance, "волк ведёт себя одинаково")

func _has(events: Array, kind: StringName) -> bool:
	for e in events:
		if e.kind == kind:
			return true
	return false
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Expected: FAIL — `Nonexistent function 'world_turn'`.

- [ ] **Step 3: Дописать `src/world/turn.gd`**

```gdscript
const ARRIVAL_CHANCE := 0.45
const WOLF_BASE_CHANCE := 0.10
const WOLF_DARK_FACTOR := 0.25

static func world_turn(s: GameState, tracker: FakeTracker) -> ActionResult:
	var r := ActionResult.new()
	if s.finished:
		return r

	resolve_coins(s, tracker.update(SceneView.from_state(s)), r)
	_tick_patience(s, r)
	_tick_lantern(s, r)
	_step_wolf(s, r)
	_advance_queue(s, r)
	_maybe_arrive(s, r)
	_maybe_wolf_appears(s, r)

	s.turn += 1

	var view := SceneView.from_state(s)
	tracker.update(view)
	tracker.roll_switches(view, s.rng)

	_check_end(s, r)
	return r

static func _tick_patience(s: GameState, r: ActionResult) -> void:
	for k in s.knights.duplicate():
		k.patience -= 1
		if k.patience <= 0:
			s.knights.erase(k)
			s.misses += 1
			r.emit(&"KNIGHT_LEAVES_ANGRY", {"knight": k.id, "slot": k.slot})

static func _tick_lantern(s: GameState, r: ActionResult) -> void:
	if s.lantern_turns > 0:
		s.lantern_turns -= 1
		if s.lantern_turns == 0:
			r.emit(&"LANTERN_DIED", {})

static func _step_wolf(s: GameState, r: ActionResult) -> void:
	if s.wolf_distance < 0:
		return
	s.wolf_distance -= 2 if s.lantern_on() else 1
	if s.wolf_distance > 0:
		r.emit(&"WOLF_STEPS", {"distance": s.wolf_distance})
		return
	if not s.counter.is_empty():
		var idx := s.rng.randi_range(0, s.counter.size() - 1)
		var stolen: int = s.counter[idx]
		s.counter.remove_at(idx)
		r.emit(&"WOLF_STEALS", {"potion": stolen})
	else:
		r.emit(&"WOLF_STEALS", {"potion": -1})
	s.wolf_distance = -1
	r.emit(&"WOLF_FLEES", {})

static func _advance_queue(s: GameState, r: ActionResult) -> void:
	var occupied := s.knights.duplicate()
	occupied.sort_custom(func(a, b): return a.slot < b.slot)
	for i in range(occupied.size()):
		if occupied[i].slot != i:
			occupied[i].slot = i
			r.emit(&"KNIGHT_ADVANCES", {"knight": occupied[i].id, "slot": i})

static func _maybe_arrive(s: GameState, r: ActionResult) -> void:
	var slot := s.free_queue_slot()
	if slot < 0:
		return
	var kinds := s.available_potion_types()
	if kinds.is_empty():
		return
	if s.rng.randf() >= ARRIVAL_CHANCE:
		return
	var k := Knight.new(
		s.next_knight_id,
		kinds[s.rng.randi_range(0, kinds.size() - 1)],
		s.rng.randi_range(0, PN.Cloak.size() - 1),
		s.rng.randi_range(6, 9))
	k.slot = slot
	s.next_knight_id += 1
	s.knights.append(k)
	r.emit(&"KNIGHT_ARRIVES", {"knight": k.id, "slot": slot, "cloak": k.cloak})

static func _maybe_wolf_appears(s: GameState, r: ActionResult) -> void:
	if s.wolf_distance >= 0:
		return
	if s.rng.randf() < WOLF_BASE_CHANCE + WOLF_DARK_FACTOR * s.dark():
		s.wolf_distance = 6
		r.emit(&"WOLF_APPEARS", {"distance": 6})

static func _check_end(s: GameState, r: ActionResult) -> void:
	if s.misses >= PN.MAX_MISSES:
		s.finished = true
		s.outcome = "failed"
		r.emit(&"NIGHT_FAILED", {"coins": s.coins})
	elif s.turn >= PN.NIGHT_TURNS:
		s.finished = true
		s.outcome = "dawn"
		r.emit(&"DAWN", {"coins": s.coins})
```

`_tick_patience` итерирует по `duplicate()`: удаление из массива во время обхода пропускает элементы. Это самая частая ошибка в таком коде.

- [ ] **Step 4: Запустить — убедиться, что проходит**

Expected: `OK: 145 проверок в 9 наборах`

- [ ] **Step 5: Коммит**

```bash
git add src/world/turn.gd tests/test_turn.gd
git commit -m "feat: ход мира и условия конца ночи"
```

---

## Task 14: Смоук — ночь всегда заканчивается

Пишется **до** играбельного среза: он ловит зависший цикл и мёртвую ночь дешевле, чем это будет видно глазами.

**Files:**
- Create: `tests/test_smoke_nights.gd`

- [ ] **Step 1: Написать провальный тест**

```gdscript
extends RefCounted

func run(t: AssertBag) -> void:
	var stuck := 0
	var dawns := 0
	var fails := 0

	for s in range(50):
		var g := GameState.new(1000 + s)
		var ft := FakeTracker.new(2000 + s)
		var guard := 0
		while not g.finished and guard < PN.NIGHT_TURNS + 5:
			Turn.world_turn(g, ft)
			guard += 1
		if not g.finished:
			stuck += 1
		elif g.outcome == "dawn":
			dawns += 1
		else:
			fails += 1

	t.eq(stuck, 0, "ни одна ночь не зависла")
	t.eq(dawns + fails, 50, "у каждой ночи есть исход")
	t.check(dawns > 0, "хотя бы одна ночь доживает до рассвета")
```

Прогон без единого действия игрока — все рыцари уходят злыми. То, что часть ночей всё же доживает до рассвета, означает, что рыцари не приходят слишком плотно: иначе три промаха набирались бы всегда и игрок не успевал бы ничего.

- [ ] **Step 2: Запустить**

Run: `engine\Godot.exe --headless --path . --script res://tests/run_all.gd`

Если `dawns == 0` — приход рыцарей слишком частый. Это находка баланса, а не поломка: снизить `ARRIVAL_CHANCE` в `turn.gd` и записать наблюдение в раздел 13 спеки.

- [ ] **Step 3: Коммит**

```bash
git add tests/test_smoke_nights.gd
git commit -m "test: пятьдесят ночей заканчиваются и не виснут"
```

---

## Task 15: Уродливый играбельный срез

Кубы вместо моделей, текст вместо панелей. Смотреть не на что — **играть можно**. Это точка первого плейтеста.

**Files:**
- Create: `src/view/main.tscn`, `src/view/main.gd`
- Modify: `project.godot`

- [ ] **Step 1: Прописать главную сцену**

В `project.godot` в секцию `[application]` добавить:

```ini
run/main_scene="res://src/view/main.tscn"
```

- [ ] **Step 2: Написать `src/view/main.gd`**

```gdscript
extends Node3D

const SLOT_X := [0.0, 1.6, 3.2]

var state: GameState
var tracker: FakeTracker
var tracks: Array[Track] = []
var selected: int = 0
var log_lines: Array[String] = []

@onready var hud: Label = $HUD/Text
@onready var slots: Node3D = $Slots

func _ready() -> void:
	_new_night(randi())

func _new_night(s: int) -> void:
	state = GameState.new(s)
	tracker = FakeTracker.new(s + 1)
	tracks = tracker.update(SceneView.from_state(state))
	log_lines = ["Ночь началась. Сид %d" % s]
	_redraw()

func _unhandled_key_input(event: InputEvent) -> void:
	if not event.is_pressed() or event.is_echo():
		return
	if state.finished:
		if event.keycode == KEY_SPACE:
			_new_night(randi())
		return

	var r: ActionResult = null
	match event.keycode:
		KEY_TAB: _cycle(); return
		KEY_Q: r = Actions.restock(state, PN.Potion.HEALING)
		KEY_W: r = Actions.restock(state, PN.Potion.STRENGTH)
		KEY_E: r = Actions.restock(state, PN.Potion.COURAGE)
		KEY_L: r = Actions.lantern(state)
		KEY_R: r = Actions.rock(state, tracks)
		KEY_T: r = Actions.recapture(state, tracker)
		KEY_1, KEY_2, KEY_3, KEY_4:
			var idx: int = event.keycode - KEY_1
			r = Actions.give(state, tracks, _selected_track_id(), idx)
		_: return

	if r == null:
		return
	if not r.ok:
		_log("отказ: " + r.reason)
		_redraw()
		return

	_log_events(r.events)
	_log_events(Turn.world_turn(state, tracker).events)
	tracks = tracker.update(SceneView.from_state(state))
	_redraw()

func _cycle() -> void:
	var knight_tracks := tracks.filter(func(t): return t.kind == &"knight")
	if knight_tracks.is_empty():
		return
	selected = (selected + 1) % knight_tracks.size()
	_redraw()

func _selected_track_id() -> int:
	var knight_tracks := tracks.filter(func(t): return t.kind == &"knight")
	if knight_tracks.is_empty() or selected >= knight_tracks.size():
		return -1
	return knight_tracks[selected].track_id

func _log(line: String) -> void:
	log_lines.append(line)
	if log_lines.size() > 8:
		log_lines.pop_front()

func _log_events(events: Array[GameEvent]) -> void:
	for e in events:
		_log(String(e.kind) + " " + str(e.data))

func _redraw() -> void:
	for c in slots.get_children():
		c.queue_free()
	for k in state.knights:
		var box := MeshInstance3D.new()
		box.mesh = BoxMesh.new()
		box.position = Vector3(SLOT_X[k.slot], 0.5, -2.0 - k.slot * 1.5)
		slots.add_child(box)

	var lines: Array[String] = []
	lines.append("ход %d/%d   монеты %d   промахи %d/%d   дождь %.2f   тьма %.2f   фонарь %s" % [
		state.turn, PN.NIGHT_TURNS, state.coins, state.misses, PN.MAX_MISSES,
		state.rain(), state.dark(), "горит" if state.lantern_on() else "нет"])
	lines.append("прилавок: " + str(state.counter.map(func(p): return PN.potion_name(p))))
	lines.append("сундук: л%d с%d х%d" % [
		state.chest[PN.Potion.HEALING], state.chest[PN.Potion.STRENGTH], state.chest[PN.Potion.COURAGE]])
	lines.append("волк: " + ("нет" if state.wolf_distance < 0 else "дистанция %d" % state.wolf_distance))
	lines.append("")
	lines.append("--- ТРЕКИНГ ---")

	var sel := _selected_track_id()
	for tr in tracks:
		var mark := ">" if tr.track_id == sel else " "
		var st := ["ЗАХВАТ", "ДРЕЙФ", "ПОТЕРЯ"][tr.state]
		if tr.kind == &"knight":
			lines.append("%s #%d %s %.2f  плащ %s" % [
				mark, tr.track_id, st, tr.conf, PN.cloak_name(tr.remembered_cloak)])
		else:
			lines.append("%s #%d %s %.2f  [%s]" % [mark, tr.track_id, st, tr.conf, tr.kind])

	lines.append("")
	lines.append("--- ЗАКАЗЫ (правда, для отладки) ---")
	for k in state.knights:
		lines.append("  рыцарь %d слот %d плащ %s хочет %s терпение %d" % [
			k.id, k.slot, PN.cloak_name(k.cloak), PN.potion_name(k.order), k.patience])

	lines.append("")
	lines.append("TAB выбрать · 1-4 отдать · Q/W/E достать · L фонарь · R камень · T перезахват")
	if state.finished:
		lines.append("")
		lines.append("=== %s === монет %d · ПРОБЕЛ — новая ночь" % [
			"РАССВЕТ" if state.outcome == "dawn" else "ПРОВАЛ", state.coins])
	lines.append("")
	lines.append_array(log_lines)

	hud.text = "\n".join(lines)
```

Блок «ЗАКАЗЫ (правда)» показывает настоящее состояние мира рядом с показаниями трекера. В финальной версии его не будет — но на плейтесте он нужен, иначе не понять, честно ли врёт трекер.

- [ ] **Step 3: Собрать `src/view/main.tscn`**

Дерево нод:

```
Main (Node3D, скрипт main.gd)
├── Camera3D          position (1.6, 1.6, 1.5), rotation.x -10°
├── DirectionalLight3D  rotation.x -50°, energy 0.3
├── Slots (Node3D)
└── HUD (CanvasLayer)
    └── Text (Label)   anchors full rect, offset 16px,
                       theme_override_font_sizes/font_size = 14
```

- [ ] **Step 4: Запустить и сыграть ночь целиком**

Run: `engine\Godot.exe --path .`

Проверить руками:
- ночь доходит до рассвета или до провала;
- при плохой погоде уверенность падает и рамки уходят в `ПОТЕРЯ`;
- фонарь поднимает уверенность и ускоряет волка;
- **подмена ID случается и видна** — в блоке трекинга плащ не совпадает с блоком правды.

- [ ] **Step 5: Коммит**

```bash
git add src/view/ project.godot
git commit -m "feat: играбельный срез на кубах с текстовым HUD"
```

- [ ] **Step 6: Плейтест с пользователем**

Показать, сыграть несколько ночей, задать три вопроса:

1. Подмена ID — интересно или бесит?
2. Тридцать ходов — мало, много, в самый раз?
3. Хватает ли информации, чтобы решать осмысленно?

**Ответы записать в раздел 13 спеки и только после этого начинать графику.** Это точка, ради которой срез стоит третьим, а не после красоты.

---

## Самопроверка плана

**Покрытие спеки.**

| Раздел спеки | Задача |
|---|---|
| 4 — цикл ночи, слоты, карты | 9, 10, 11, 13 |
| 5 — уверенность, состояния, подмена, монеты, волк | 7, 8, 11, 12, 13 |
| 6 — зелья, прилавок, сундук, терпение, невозможность тупика | 3, 5, 9, 13 |
| 7 — анимация | **вне объёма среза** — заложены события, проигрывание в плане одевания |
| 8 — ассеты, свет, дождь | **вне объёма среза** — снят только риск (задача 1) |
| 9 — три слоя, `TrackSource`, детерминизм | 5, 6, 13 |
| 10 — структура папок | 0 |
| 11 — тесты headless | 2, 14 |
| 12 — порядок работ | весь план |

Разделы 7 и 8 сознательно не покрыты: это план одевания, который пишется после плейтеста.

**Запись ground-truth (раздел 9 спеки) в срез не вошла** — она нужна проекту 3, а не плейтесту. Добавляется в план одевания.

**Согласованность имён.** `PN.state_from_conf` (3) → `Track.refresh_state` (6) → пороги `PN.CONF_LOCKED/CONF_DRIFTING` (3, 8). `SceneView.from_state` (6) → используется в 11, 12, 13, 15. `FakeTracker.track_for` (8) → используется в тестах 10, 11. `Actions.give` принимает `Array[Track]` во всех вызовах. `Turn.resolve_coins` (12) вызывается из `Turn.world_turn` (13).

**Известное расхождение со спекой.** Спека говорит, что перезахват может выдать новые номера треков; `FakeTracker.recapture` так и делает (`_next_track_id` продолжает расти). Тест в задаче 8 проверяет только починку привязки и снятие флага, номера не фиксирует — это намеренно, иначе тест сломается от любой правки нумерации.

**Инвариант, который держим во всех оставшихся задачах:** `src/world/` и `src/tracking/` не наследуют `Node` и ничего не знают о сценах. Проверяется тем, что все их тесты гоняются headless без единой сцены.
