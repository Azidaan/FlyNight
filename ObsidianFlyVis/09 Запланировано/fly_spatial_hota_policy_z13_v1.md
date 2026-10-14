---
aliases:
- 'З13 v2: HOTA при одной политике верхнего 1% уверенности'
tags:
- эксперимент
- исход/ready
- направление/flyvis
исход: готов к запуску
направление: FlyVis
начат: '2026-10-14'
---

# ⚪ З13 v2: HOTA при одной политике верхнего 1% уверенности

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** готов к запуску · `fly_spatial_hota_policy_z13_v1`

## Вопрос
Даёт ли готовый пространственный сигнал FlyVis пользу HOTA при одной новой политике вмешательства?

## Гипотеза
Провал инженерного 90%/50 допуска не определяет HOTA; одна фиксированная политика позволит проверить фактический эффект.

## Ожидалось (до прогона)
Одна рабочая точка B/C/D: верхний 1% всех естественных предсказаний на train26, без меток и HOTA в выборе; все равные оценки вместе. Полный трекер на dev35 с прежней MASA, стенд и B/C/D, без нового обучения. Это новый опыт, старый итог [fly_spatial_quality_z13_v2↗](obsidian://open?vault=ObsidianFlyVis&file=01%20%D0%9C%D1%83%D1%85%D0%B0%2F%D0%AD%D0%BA%D1%81%D0%BF%D0%B5%D1%80%D0%B8%D0%BC%D0%B5%D0%BD%D1%82%D1%8B%2Ffly_spatial_quality_z13_v2) сохраняется. Подготовка и короткая проба разрешены; полный бюджет после замера.

## Нельзя заявлять
Проба не подтверждает HOTA. Верхний 1% не гарантирует безопасность или такую же частоту вмешательств на dev. Dev — разработка; независимые видео не открываются. Исходный отрицательный критерий не переписывается.

## Критерии приёмки (записаны до прогона)
- ✅ No-intervention output exactly matches baseline; stored-episode spatial probabilities agree; only lost tracks and centers changed; bounded batches of four; all arms load.  
  *runs/z13_spatial_hota_v1_20261014/technical/report.json: all models loaded, streaming parity <=1e-5, no-intervention outputs exactly equal. CPU contract tests cover center-only/lost-only/chunking/memory.*
- ⏳ D−стенд: общая HOTA минимум +0,10 и нижняя граница парного ДИ выше нуля
- ⏳ Вклад FlyVis: нижняя граница парного ДИ D−B выше нуля
- ⏳ Польза обучения ядра отдельно: нижняя граница парного ДИ D−C выше нуля

## Доказательства
- Политика и критерии до HOTA: `runs/z13_spatial_hota_v1_20261014/protocol.json` · 
- Исходное завершённое обучение и провал прежнего допуска: `runs/z13_spatial_v2_20261013/final_verification.json` · 
- Technical probe, no HOTA or annotation reading: `runs/z13_spatial_hota_v1_20261014/technical/report.json` · 
- One-video MASA inference price: `runs/z13_spatial_hota_v1_20261014/technical/masa_price/report.json` · 
- Arithmetic workload estimate, not guaranteed bound: `runs/z13_spatial_hota_v1_20261014/price.json` · 
- Neuron timing matches training; environment kept unchanged: `runs/z13_spatial_hota_v1_20261014/timing_contract.json` · 

