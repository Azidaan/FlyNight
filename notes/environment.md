# Окружение проекта (Этап 1)

Дата фиксации: 2026-07-01. Железо: NVIDIA RTX 5070 (12 ГБ), драйвер 581.29, Windows 11 Pro.

## Принцип: полная локальность

Всё окружение живёт внутри `E:\Dissertation` и переживает переустановку Windows (диска C:).
Обычный venv от системного Python так не умеет — его `pyvenv.cfg` ссылается на
интерпретатор на C:. Поэтому базовый интерпретатор тоже лежит в проекте.

| Компонент | Расположение |
|-----------|--------------|
| Портативный CPython 3.10.11 | `python310\tools\python.exe` (NuGet-пакет) |
| Архив пакета (на случай пересоздания) | `installers\python-3.10.11-nuget.zip` |
| Виртуальное окружение | `.venv\` (home = `E:\Dissertation\python310\tools`) |
| Склад wheel-пакетов (офлайн-установка) | `wheelhouse\` |
| Кэши pip / torch / ultralytics / matplotlib | `.cache\{pip,torch,ultralytics,matplotlib}` |
| Предобученные веса YOLO | `weights\pretrained\` |
| Сторонние репозитории (eval tools) | `third_party\` |

Кэши перенаправляются автоматически: `.venv\Lib\site-packages\sitecustomize.py`
(мастер-копия — `scripts\setup\sitecustomize.py`) выставляет `PIP_CACHE_DIR`,
`TORCH_HOME`, `YOLO_CONFIG_DIR`, `MPLCONFIGDIR` при каждом старте интерпретатора.
Дублирующие экспорты дописаны в `.venv\Scripts\Activate.ps1` и `activate.bat`.

## Версии (зафиксированная конфигурация, Часть 2.7 плана)

- Python 3.10.11 (портативный, NuGet)
- torch 2.10.0+cu130, torchvision 0.25.0+cu130 (CUDA 13.0)
- Полный список с точными версиями — `requirements.txt` (pip freeze)

Правило: суффикс версии torch обязан быть `+cu130`, никогда не `+cpu`.
Проверяется скриптом `scripts\smoke\test_cuda.py`.

## Восстановление после переустановки Windows

1. Поставить NVIDIA-драйвер (581.29 или новее).
2. Подключить/сохранить диск E: — окружение работает сразу, без установки чего-либо:
   `E:\Dissertation\.venv\Scripts\python.exe scripts\smoke\test_cuda.py`.
3. Если `.venv` всё же сломан — пересоздать полностью офлайн:
   ```
   E:\Dissertation\python310\tools\python.exe -m venv E:\Dissertation\.venv --clear
   copy scripts\setup\sitecustomize.py .venv\Lib\site-packages\
   .venv\Scripts\python.exe -m pip install --no-index --find-links wheelhouse -U pip setuptools wheel
   .venv\Scripts\python.exe -m pip install --no-index --find-links wheelhouse -c constraints.txt -r requirements.in
   ```
4. Если сломан и `python310\` — распаковать `installers\python-3.10.11-nuget.zip`
   в `python310\` и повторить шаг 3.

Интернет при восстановлении не нужен: все wheel-пакеты лежат в `wheelhouse\`.

## Датасет

BDD100K скачан полностью и проверен по MD5 19.06.2026 (`data\bdd100k\md5_verify.log`,
все архивы OK). Распакован: `images/track/{train,val,test}` — 1400/200 видео
(train/val), `labels/box_track_20/{train,val}`, per-image Det-лейблы
`labels/bdd100k_labels_images_{train,val}.json` (источник тэгов weather/timeofday
для стресс-поднаборов), `info/100k`. Структура соответствует Этапу 2 плана.

## Инструмент оценки трекинга

**План А подтверждён (2026-07-02):** официальный пакет `bdd100k` 1.0.1 (+ `scalabel` 0.3.1)
установлен с PyPI, `python -m bdd100k.eval.run --help` работает. Нативный формат BDD100K,
конвертация MOT-результатов не требуется. TrackEval (План Б) не понадобился.

## Выбор версии YOLO

**YOLO26 (2026-07-02):** все три версии (YOLO26/v11/v8, масштаб s, ultralytics 8.4.84)
прошли 4 smoke-проверки; выбрана YOLO26 как первая по приоритету Части 2.8 плана.
FPS на батче 16: YOLO26 — 237, YOLOv11 — 340, YOLOv8 — 297 (цель ≥ 50).
Детали: `notes\yolo_version_choice.md`, `experiments\results\smoke_test_yolo.csv`.

## SkillOpt (эксперименты рядом с диссертацией)

Установлен 2026-07-02 в отдельном окружении (не трогает основной стек):
- `tools\skillopt\.venv` — venv от локального `python310\`, пакет `skillopt[claude]` 0.1.0
  (CLI: `skillopt-train`, `skillopt-eval`; wheels в общем `wheelhouse\`).
- `tools\skillopt\SkillOpt\` — клон репозитория (движок `skillopt_sleep`, конфиги, доки,
  плагин). Не удалять: на него ссылается marketplace Claude Code.
- Плагин **skillopt-sleep** установлен в Claude Code (scope user):
  `claude plugin marketplace add tools\skillopt\SkillOpt\plugins\claude-code`,
  затем `claude plugin install skillopt-sleep@skillopt-sleep --scope user`.
  После переустановки Windows повторить эти две команды (клон на E: сохранится).

## Smoke-тесты

Прогон всех проверок Этапа 1: `powershell -File scripts\smoke\run_all_smoke.ps1`.
Результаты YOLO-тестов: `experiments\results\smoke_test_yolo.csv`,
решение по версии — `notes\yolo_version_choice.md`.
