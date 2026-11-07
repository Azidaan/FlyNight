---
aliases:
- 'Б3 — память объектов со связями мухи: проверка обучения и перестройки связей'
tags:
- эксперимент
- исход/ready
- направление/flyvis
- метка/infrastructure
исход: готов к запуску
направление: FlyVis
начат: '2026-10-27'
---

# ⚪ Б3 — память объектов со связями мухи: проверка обучения и перестройки связей

↑ [[09 Запланировано/FlyVis — планы|FlyVis]] · **Исход:** готов к запуску · `b3_object_memory_gradient_v1`

Готовая ограниченная проверка реализации; в локальной очереди после Б1. Научная гипотеза не проверена.

## Вопрос
Проходит ли обучающая ошибка по рабочему пути выбранного узла с реальными входами и согласованными контролями?

## Ожидалось (до прогона)
Learn appearance257 -> receptor721 adapter, recurrent memory, same candidate comparison head; additive identity correction to frozen appearance base. Three scratch Adam steps, known real-region identity BCE; labels separate from observed inputs. No checkpoints promoted.

## Нельзя заявлять
One real clip and three scratch steps, not memory accuracy, topology superiority, HOTA or scientific B3 acceptance. Adapter has substantially more parameters than the core; ordinary control budget differs and is reported.

## Критерии приёмки (записаны до прогона)
- ⏳ Zero output preserves base probabilities exactly
- ⏳ Finite loss and nonzero adapter gradient after head update; D/W finite nonzero core gradient
- ⏳ Report actual rewiring invariants and budgets; real inputs use only past observed assignments, not true identity

## Доказательства
- Условия и критерии до расчёта: `runs/b123_20261027/b3_gradient/protocol.json` · 
- Автор разрешил Б1–Б3 до 07:30 МСК: `runs/b123_20261027/authorization.json` · 

## Код
- `tracking/b3_object_memory.py`
- `scripts/b3_gradient_profile_20261027.py`

