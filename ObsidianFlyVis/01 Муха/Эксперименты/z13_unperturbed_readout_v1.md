---
aliases:
- 'З13: обучение считывателя на эпизодах без искусственного сдвига окна'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/fly-problem-P8
- муха
исход: неопределённый
направление: FlyVis
начат: '2026-10-24'
---

# 🟡 З13: обучение считывателя на эпизодах без искусственного сдвига окна

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_unperturbed_readout_v1`

Постановка ошибочна:кэш уже исключил сдвинутые эпизоды;остановлено без научного сравнения.

## Вопрос
Поможет ли согласование режима обучающих вырезок с применением при равном размере и бюджете?

## Гипотеза
Удаление искусственных сдвигов может улучшить выбор полезных ответов;польза и причинность не установлены.

## Ожидалось (до прогона)
Change only fitting-sample distribution inside the learned FlyVis candidate readout:unperturbed episodes versus equally sized mixed episodes. No new features,loss,core weights,candidates or architecture. Same BCE candidate useful labels,original saved positive weight for each arm/mode/seed,AdamW lr0.001 weight_decay0.01,clip10,50referenceepochs,nref2919,batch128,1150updates/145950row presentations,initial seeds0/1/2. Three-seed mean logits,never pick best seed. Old18full-data models independently replayed as secondary reference. Same current/past information;unperturbed flag used only for sampling/evaluation grouping,not as network input.

## Что сделано
2026-10-24

## Получено
runs/z13_unperturbed_readout_20261024/eligibility_audit.json:исходных эпизодов19798,со сдвигом9888;в выбранных обучающих/оценочных строках со сдвигом0/0. Обе обучающие выборки содержат одни и те же20778известных строк. Сохранено12готовыхмоделей;controller_result exit1/all_owned_ended после stop.request.

## Интерпретация
Агент ошибочно перенёс описание исходного архива на уже отфильтрованный кэш. Два режима не различаются составом строк,поэтому обучение не проверяет гипотезу. Из различия порядка пакетов выводов не делаем. Это ошибка постановки,не отрицательный результат FlyVis.

## Решение
Не продолжать эту постановку и не принимать её модели;веса/логи/источники/exit1 сохранить. До любого нового обучения проверять на конкретных входных строках наличие заявленного различия факторов.

## Нельзя заявлять
Artificial crop shifts may improve robustness;removing them can reduce diversity and overfit. Current fit samples are naturally different episodes,not counterfactual copies;even matched video/kind/count leaves object/age/size differences. Unperturbed baseline episodes are only closer to runtime,not actual corrected closed-loop state. A confirmed-yield gain can reflect unknown-label composition,not real precision;bounds reported. This exploratory repeatedly used26train pilot cannot establish cause,HOTA or seed independence of the core.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-24 (c5df9b85b63b36990cb81adfa53713b9cc47b62c) — Обнаружено отсутствие заявленного различия обучающих строк;явная поправка ошибки агента.
  > Отдельная замена обучающей выборки,критерии до исходов.

## Критерии приёмки (записаны до прогона)
- ⏳ Evaluate only originally unperturbed episodes of all26reused train videos. Select ceil(0.05*N) from ALL eligible rows per video before reading outcomes. Confirmed yield=known-useful/ALL selected;unknown cases separately give an upper bound (useful+unknown)/selected,not observed errors. Dclean-Dmixed confirmed-yield mean>=0.05 and paired95%lower>0. Dclean must beat cleanDgeometry and cleanR1activity with both lower95>0. Useful-center mean on kind1 must not fall below mixedD. All26 must have eligible and positive rows. All required;otherwise no expensive transfer.  
  *runs/z13_unperturbed_readout_20261024/eligibility_audit.json;comparison not identifiable*
- ❌ 36моделей,3сида,полное воспроизведение18старых моделей,старой32шаговойтраектории и неизменность источников.  
  *runs/z13_unperturbed_readout_20261024/eligibility_audit.json;stopped before36models*

## Доказательства
- Новый режим выборки и все контроли до обучения: `runs/z13_unperturbed_readout_20261024/protocol.json` · 

## Код
- `scripts/z13_unperturbed_readout_20261024.py`

