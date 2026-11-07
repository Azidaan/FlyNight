---
aliases:
- Равные признаки прежних областей D/C/R1 для временного считывания
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/fly-problem-P8
- муха
- проверка-гипотезы/завершена
исход: неопределённый
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 Равные признаки прежних областей D/C/R1 для временного считывания

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_temporal_region_features_v1`

Сохранены434блока; исходная подготовка остановлена после зависания передачиCUDA. Научного результата нет.

## Вопрос
Можно ли получить равные причинные признаки прежних областей без изменения нынешнего ответа FlyVis?

## Гипотеза
Проверенная доступность прошлых наблюдений допускает отдельное извлечение признаков; способность различать личность пока не проверена.

## Ожидалось (до прогона)
Prepare previous-region68activity and observable geometry for one future learned cross-frame FlyVis readout. Existing current features/target/action labels stay immutable and are referenced byUID, not replaced. D/C/R1 have identical selected observations/regions.

## Что сделано
2026-10-25

## Получено
runs/z13_temporal_identity_20261025/legacy_preservation.json:434готовыхблока с провереннымихешами; runs/z13_temporal_region_features_20261025/stalled_transfer_audit_20261025.json:двастекаCUDAH2D,прогресс434с22:40;controller_result.json exit4294967295/all_owned_ended после остановки проверенного дерева.

## Интерпретация
Подтверждена техническая остановка, причина драйвера не установлена. ОставшиесяC/R1 не рассчитаны; метрики эффективности не получены.

## Решение
Готовые434блока сохранить. Новая выборка прошлых опорных наблюдений имеет отдельный протокол и бюджет; прежний срок не продлевается и старые источники не переписываются.

## Нельзя заявлять
Earlier same-class regions may be distractors or too stale, and FlyVis may not encode enough identity. Availability/oracle passed, but learned benefit remains unknown. Spatial-set module stays deferred; this is a distinct temporal input, not resuming its75models.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-25 (e511d16db083ef576538765cc23acefd0a11b7dc) — Verified technical CUDA transfer stall; preserve partial work and original error.
  > Подготовка временных признаков; обучение ещё не запускалось.

## Критерии приёмки (записаны до прогона)
- ⏳ For every eligible current query, old primary peak center<.5px and score<.001. Same weights, input sequence, normalization and target coefficients. No tolerance weakening. Stored previous arrays finite and save/reload exact; all22090UIDs and previous region indices same D/C/R1. Replay from saved data before any future learner.
- ⏳ OneGPUmutex;16episode inference packing,not learning batch. First2Dgroups:2*measured seconds per weighted-frame*remainingD+Cweightedframes+600 must fit remaining7200seconds. If not, preserve and stop. R1 remains deterministic retinal feature computation onCPU, not CPU learning. Independent controller hiddenIdle; no coretraining/dev/HOTA/heldout75/download.

## Доказательства
- Правила причинных входов, воспроизведение и7200сбюджет доGPUизвлечения: `runs/z13_temporal_region_features_20261025/protocol.json` · 

