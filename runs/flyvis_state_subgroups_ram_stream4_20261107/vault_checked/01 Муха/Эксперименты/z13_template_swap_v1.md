---
aliases:
- 'З13: изолированная подмена образца при прежнем состоянии поиска'
tags:
- эксперимент
- исход/completed_positive
- направление/flyvis
- метка/fly-problem-P8
- метка/engineering-future
- муха
- проверка-гипотезы/завершена
исход: положительный
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 З13: изолированная подмена образца при прежнем состоянии поиска

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** положительный · `z13_template_swap_v1`

Изолированная подмена полного фильтра проверена для D/C/R1; это диагностика зависимости, не новое улучшение.

## Вопрос
Использует ли действующий различающий фильтр правильный образец сверх той информации, которая уже содержится в неизменном состоянии мухи?

## Гипотеза
Уверенный отклик может недостаточно зависеть от личности образца; это гипотеза, а не установленная причина. Подмена полного фильтра отделяет его вклад от изменения состояния.

## Ожидалось (до прогона)
Frozen query activity and recurrent history computed once per arm in original sorted groups of8. Replace all68 filter coefficients with3 distinct fit104 donor filters; exact same observed721-column mask and known different observed identity/video. Full721-column maps and peaks recomputed. No fixed peaks/scores from correct template supplied to wrong condition. Normalization and per-arm lambda unchanged; no fitting or selection on26 outcomes. До3600с, одинGPU; C/D/R1, полная проверка8326строк, одинаковые доноры.

## Что сделано
2026-10-25

## Получено
runs/z13_template_swap_20261025/recovery1/report.json → arms: {'D': {'paired': {'mean_per_video': 0.16577610613956928, 'ci95': [0.11596939668820588, 0.2363515249359722], 'better': 26, 'n': 26}, 'eligible': 4844, 'interpretation': 'own_template_advantage_detected'}, 'C': {'paired': {'mean_per_video': 0.17868367855068618, 'ci95': [0.11680008697668476, 0.26428624168329073], 'better': 26, 'n': 26}, 'eligible': 4844, 'interpretation': 'own_template_advantage_detected'}, 'R1': {'paired': {'mean_per_video': 0.01001659629159891, 'ci95': [-0.02038076160825022, 0.03495597707890163], 'better': 16, 'n': 26}, 'eligible': 4844, 'interpretation': 'inconclusive'}}. Все8326 исходных строк каждого варианта воспроизведены; первичные карты и интервалы проверены независимо.

## Интерпретация
В этом считывателе правильный образец даёт положительный вклад в полезные центры сверх трёх чужих моделей при общей истории поиска. Общее утверждение, что текущий фильтр игнорирует образец, не подтверждается.

## Решение
Не переносить вспомогательное обучение из статьи только по сходству симптомов. Сопоставить диагностические ответы с ошибками отказа и доступностью правильного кандидата перед выбором одной функциональной правки. Дорогой этап автоматически не запускается.

## Нельзя заявлять
Correct observations still enter the frozen recurrent state. Lack of additional filter benefit cannot establish total template neglect or an intrinsic FlyVis defect. Other-video donors are easier than in-scene distractors; geometry matching does not equal class/appearance matching. Gaussian labels and discriminative ridge fitting already condition the model on the observed region. Reused train26, no claim of independent generalization.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-25 (58e88472581222d0dc29bc4f8676cfaf2b2e91b7) — Проверен полный либо частичный результат изолированной подмены.
  > Протокол изолированной проверки и строгие пары зарегистрированы до вычисления исходов.
- **Суть**, 2026-10-25 (58e88472581222d0dc29bc4f8676cfaf2b2e91b7) — Служебный сбой сохранён; начато восстановление в прежнем окне.
  > Сохранена незавершённая изолированная проверка образца.
- **Суть**, 2026-10-25 (60ca8b5b42f6ebcdc7463e31b01925b71375026e) — Проверен полный либо частичный результат изолированной подмены.
  > Идёт recovery1: тот же изолированный опыт после создания выходных каталогов; прежняя ошибка сохранена.
- **Получено**, 2026-10-25 (60ca8b5b42f6ebcdc7463e31b01925b71375026e) — Проверен полный либо частичный результат изолированной подмены.
  > runs/z13_template_swap_20261025/controller_result.json: {'exit_code': 1, 'hard_timeout': False, 'worker_ended': True, 'all_owned_ended': True, 'remaining': {}, 'owned': {'20212': 134373567588969631, '28436': 134373567589088023, '27248': 134373567589288230}, 'ended': '2026-10-25T02:06:54.707370+03:00'}. Сохранённых групп:0
- **Интерпретация**, 2026-10-25 (60ca8b5b42f6ebcdc7463e31b01925b71375026e) — Проверен полный либо частичный результат изолированной подмены.
  > Неполный проход не даёт вывода о зависимости от образца или эффективности.
- **Решение**, 2026-10-25 (60ca8b5b42f6ebcdc7463e31b01925b71375026e) — Проверен полный либо частичный результат изолированной подмены.
  > Сохранить готовое; проверить служебную причину и остаток исходного бюджета. Не повторять готовые группы и не продлевать бюджет.

## Критерии приёмки (записаны до прогона)
- ✅ All8326 own-template rows/arm; centers<0.5pixel and scores<0.001 vs frozen single_rows, all source hashes fixed; same original groups of8. Query arrays unchanged by projection. Foreign model IDs and exact observed mask checked for every eligible episode. All26 videos contain eligible positive rows.  
  *runs/z13_template_swap_20261025/recovery1/independent_audit.json*
- ✅ On the exact-mask eligible kind1 rows, each video contributes its own-template useful-center fraction minus the mean of the3 foreign-template fractions. Paired-video95%CI: lower>0 detects useful template dependence. Upper<0.02 establishes only an effect below2pp in this diagnostic. Otherwise inconclusive. This is not a criterion for model usefulness or training approval.  
  *runs/z13_template_swap_20261025/recovery1/report.json*

## Доказательства
- Постановка, три донорских фильтра и правила вывода до результата: `runs/z13_template_swap_20261025/protocol.json` · 
- Только создание выходных каталогов; исходный срок и код неизменны: `runs/z13_template_swap_20261025/recovery1/protocol.json` · 
- Свой и три чужих фильтра, одинаковое состояние и геометрия: `runs/z13_template_swap_20261025/recovery1/report.json` · 

## Код
- `tracking/z13_template_swap.py`
- `tests/test_z13_template_swap.py`
- `scripts/z13_template_swap_20261025.py`
- `scripts/z13_finish_template_swap_20261025.py`
- `scripts/z13_template_swap_console.ps1`

