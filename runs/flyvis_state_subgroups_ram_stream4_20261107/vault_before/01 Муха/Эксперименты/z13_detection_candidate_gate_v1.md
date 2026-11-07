---
aliases:
- 'З13: запас локализации при чтении трёх детекций вместо пиков'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/fly-problem-P8
- муха
- проверка-гипотезы/завершена
исход: неопределённый
направление: FlyVis
начат: '2026-10-24'
---

# 🟢 З13: запас локализации при чтении трёх детекций вместо пиков

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_detection_candidate_gate_v1`

Проверка доступности кандидатов завершена;inconclusive;реализация нового считывания ещё не принята.

## Вопрос
Есть ли дополнительный запас полезных центров для смены пространства кандидатов считывателя?

## Гипотеза
Реальные рамки могут содержать полезный центр,отсутствующий среди3пиков;самой мухе этот запас может быть недоступен.

## Ожидалось (до прогона)
Replace the output candidates of a future FlyVis readout:three actual detection centers,not three arbitrary response peaks. First measure saved-data conditional localization reserve;no model is trained in this gate. Raw unchanged YOLO26 boxes confidence>=0.25 (existing first-association threshold),positive area,center inside same saved square crop;nearest3 to current saved Kalman prediction,stable original order for ties. No class filtering or GT in generation. At most3 candidates,as in old peak readout. No cap variants or threshold search.

## Что сделано
2026-10-24

## Получено
runs/z13_detection_candidate_gate_20261024/report.json → mean/paired_vs_D_peak3_oracle: {"mean": {"detection3_oracle": 0.4744743791566531, "all_detection_oracle": 0.4872443261974299, "nearest_detection": 0.4112729139962888, "D_peak3_oracle": 0.45497892184525557, "C_peak3_oracle": 0.4357160847062459, "R1_peak3_oracle": 0.22894914273146919, "D_peak1": 0.25488581030332724}, "paired": {"detection3_oracle": {"mean_per_video": 0.019495457311397398, "ci95": [-0.07148105864317593, 0.09944142920737432], "better": 15, "n": 26}, "all_detection_oracle": {"mean_per_video": 0.03226540435217425, "ci95": [-0.06083055692598325, 0.11699576374033069], "better": 15, "n": 26}, "nearest_detection": {"mean_per_video": -0.0437060078489668, "ci95": [-0.12056731071227109, 0.023377944867076214], "better": 11, "n": 26}}}

## Интерпретация
Измерен только условный оракульный запас предложения детекций при известных центрах. Это не работа обученного узла,не вклад FlyVis и не польза стенду.

## Решение
Не извлекать признаки этой версии;сохранить полный отрицательный/неубедительный исход и выбрать другой механизм.

## Нельзя заявлять
Correct detection may already be selected by geometry;all gain may come from YOLO proposals,notFlyVis. Oracle ignores multi-track competition;class-unfiltered square-crop support is not identical to the hexagonal feature field. Projection/descriptor extraction and missing-candidate refusal remain untested. Sparse labels,perturbed training crops,and repeatedly used26train do not establish closed-loop transfer.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-24 (e8d47b72ffb4b05af27773aa181be95e22661d3b) — Сохранена исходная постановка перед итоговым summary;начальное значение изprepare.
  > Диагностическая CPU-проверка до600с зарегистрирована до расчёта.

## Критерии приёмки (записаны до прогона)
- ❌ All26 repeated-train videos and all8326 saved labels align exactly. Detection3 oracle useful fraction minus oldD peak3 oracle:mean>=0.02 and paired95%lower>0. If upper<0.02 stop this candidate set;otherwise inconclusive. This is only an oracle availability gate,not an attainable gain or HOTA.  
  *runs/z13_detection_candidate_gate_20261024/report.json*

## Доказательства
- Кандидаты,критерий и ограничения до исхода: `runs/z13_detection_candidate_gate_20261024/protocol.json` · 

## Код
- `scripts/z13_detection_candidate_gate_20261024.py`

