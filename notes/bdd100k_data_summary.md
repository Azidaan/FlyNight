# Сводка по подготовленным данным BDD100K MOT

Дата подготовки: 2026-06-15.

## Источник

Скачано через `openxlab dataset download` из репозитория `OpenDataLab/BDD100K`. Оригинальный архив: `/raw/BDD100K.tar.gz` (63.02 ГБ). MD5 всех 10 zip-шардов MOT 2020 Images верифицированы и совпали с эталонами из парных `.md5`-файлов (см. `data/bdd100k/md5_verify.log`).

Каноническое зеркало `dl.cv.ethz.ch` на момент подготовки данных **физически снято с DNS** (NXDOMAIN); Berkeley-страница `bdd-data.berkeley.edu` не отдаёт MOT labels из-за поломанной кнопки. Это документировано открытым [issue #369](https://github.com/bdd100k/bdd100k/issues/369). OpenDataLab — единственный действующий канал на дату подготовки.

## Структура

Корень `E:\Dissertation\data\bdd100k\` (отличается от `data\bdd100k_mot\` в Части 5.1 плана — единственное расхождение, которое нужно поправить при будущей правке СВОДНЫЙ_ПЛАН1.md).

```
data\bdd100k\
├── images\
│   └── track\
│       ├── train\<video_id>\<video_id>-<frame>.jpg     # 1400 видео, ~280K кадров
│       ├── val\<video_id>\<video_id>-<frame>.jpg       #  200 видео, ~40K кадров
│       └── test\<video_id>\<video_id>-<frame>.jpg      #  400 видео, ~80K кадров
├── labels\
│   ├── box_track_20\
│   │   ├── train\<video_id>.json                       # MOT Scalabel: frames[].labels[] с track id
│   │   └── val\<video_id>.json                         # 200
│   ├── bdd100k_labels_images_train.json                # 69863 per-image записей с attributes (weather/timeofday/scene)
│   └── bdd100k_labels_images_val.json                  # 10000 per-image записей
└── info\
    └── 100k\
        ├── train\<video_id>.json                       # 70000 файлов: GPS/accel/gyro/timestamp/rideID
        └── val\<video_id>.json                         # 10000
```

Помимо этого, в `data\bdd100k\extracted\BDD100K\` остаются **как бонус** не-MOT подсеты в виде zip-архивов: `100K Images`, `10K Images`, `Detection 2020 Labels`, `Drivable Area`, `Lane Marking`, `Pose Estimation`, `Segmentation`, `Instance Segmentation`, `Panoptic Segmentation`, `MOTS 2020 Images`, `MOTS 2020 Labels`, `Videos`. Могут пригодиться для опционального раздела 4.4 плана и для smoke-тестов.

## Формат MOT label JSON (для конвертера BDD JSON → YOLO txt из Этапа 2.3 плана)

Каждый файл `labels\box_track_20\<split>\<video_id>.json` — это **список** объектов-кадров (frames). Пример первого кадра:

```json
{
  "videoName": "0000f77c-6257be58",
  "name":      "0000f77c-6257be58-0000001.jpg",
  "frameIndex": 0,
  "labels": [
    {
      "id":       "00002828",                    // ← track ID (стабильный по всему видео)
      "category": "car",                         // ← одна из 10 классов плана
      "attributes": {"crowd": false, "occluded": false, "truncated": true},
      "box2d":    {"x1": 0, "x2": 202.74, "y1": 266.36, "y2": 493.48}
    },
    ...
  ]
}
```

**Важно**: тэги weather/timeofday/scene **отсутствуют** в MOT label JSON. Они живут в `bdd100k_labels_images_<split>.json` per-image и присоединяются к MOT-видео по совпадению ключа: `video_id == os.path.splitext(image_name)[0]`. Это покрывает **100%** MOT-видео (1400/1400 train, 200/200 val).

Это означает, что `subset_filter.py` (Этап 2.3 плана) должен загружать **два** источника:
- список train/val MOT-видео из `images\track\<split>\` (имена директорий)
- атрибуты из `labels\bdd100k_labels_images_<split>.json` (поле `attributes.weather` / `timeofday` / `scene`)

## Размеры стресс-поднаборов

### Train (1400 видео)

| Подмножество | Определение | Видео |
|---|---|---|
| Normal | timeofday=daytime AND weather=clear | 274 |
| Night | timeofday=night | 564 |
| Rain | weather=rainy | 85 |
| Fog (узкое) | weather=foggy | **1** ⚠ |

Полное распределение:
- weather: clear 773, overcast 196, undefined 150, partly cloudy 114, rainy 85, snowy 81, foggy 1
- timeofday: daytime 757, night 564, dawn/dusk 78, undefined 1
- scene: city street 854, highway 370, residential 155, parking lot 10, undefined 8, tunnel 3

### Val (200 видео)

| Подмножество | Видео |
|---|---|
| Normal | 36 |
| Night | 71 |
| Rain | 16 |
| Fog | 0 ⚠ |

## Известная проблема: Fog почти пустой

В BDD100K только 1 train-видео и 0 val-видео с тэгом weather=foggy — статистически невозможно. Это известный недостаток самого датасета (BDD собирался преимущественно в California, туман редкий).

Этап 2 плана уже описывает план Б на этот случай:
> «количество видео в одном из поднаборов слишком мало (например, меньше 50 видео в Fog). План Б: пересмотреть определение поднабора в сторону более широкого критерия (например, weather in ['foggy', 'overcast'] для Fog).»

Если применить расширение weather ∈ {foggy, overcast}:
- Fog train: 1+196 = **197 видео** ✓
- Fog val: 0+(overcast в val) ≈ 30-40 (точное число посчитаем при первом запуске subset_filter.py)

**Рекомендация**: использовать расширенное определение Fog с самого начала, явно сослаться в разделе 2.3 диссертации на ограничение BDD100K и обосновать выбор. Это снимает риск, что субсет окажется статистически невалидным.

## Логи и скрипты этой подготовки

В `data\bdd100k\` лежат служебные файлы, которые можно безопасно удалить после ревизии:
- `download.log`, `extract.log`, `extract.err` — логи скачки и распаковки tar.gz
- `md5_verify.log` — MD5-верификация шардов
- `unzip_images.log` — лог параллельной распаковки MOT-images
- `peek_zips.py`, `unzip_images.py`, `sanity_check.py`, `verify_tags.py` — одноразовые скрипты подготовки
- `watch_progress.ps1`, `watch_extract.ps1` — watcher-ы для прогресс-баров (тоже одноразовые)
