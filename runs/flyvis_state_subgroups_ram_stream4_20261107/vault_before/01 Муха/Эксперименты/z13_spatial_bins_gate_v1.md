---
aliases:
- 'Пространственное считывание: доступность четырёх частей реальной рамки'
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

# 🟢 Пространственное считывание: доступность четырёх частей реальной рамки

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_spatial_bins_gate_v1`

Геометрическая мишень: inconclusive

## Вопрос
Do enough known queries with a reachable beneficial action also have a correct and an incorrect real detector candidate, each with nonempty retinal support in all4ordered quadrants?

## Гипотеза
Рабочая гипотеза: пространственный рисунок внутри области может содержать сведения, отброшенные одним средним. Геометрическая доступность этих сведений и их полезность проверяются отдельно.

## Ожидалось (до прогона)
Candidate internal readout: replace one mean-pooled68-channel vector per detector region by four ordered2x2 spatial-bin vectors. Keep original unit-clock core, weights and target model; equal-parameter control repeats the original mean into all4bins. This gate measures only geometric feasibility and oracle exposure, not prediction or performance.

## Что сделано
2026-10-25

## Получено
runs/z13_spatial_bins_gate_20261025/report.json → spatial_oracle_exposure: {'mean_per_video': 0.022802605365315858, 'ci95': [0.01636847685657445, 0.02932090951337184], 'better': 22, 'n': 26};37296запросов,26600известных, прежние реальные области совпадают.

## Интерпретация
Это только доля случаев с разрешимыми четырьмя частями рамки и оракульно достижимым полезным действием среди неверных кандидатов. Не измерены различимость признаков, точность или HOTA.

## Решение
Pass -> separately preregister and budget four-bin feature extraction, old-response parity, equal-capacity readout controls and assignment gate. No automatic HOTA or training admission from geometric coverage. Fail/inconclusive -> preserve this exact criterion; do not relax bin occupancy or choose a favorable subset. Gate: inconclusive

## Нельзя заявлять
The lattice may resolve4bins yet contain no identity evidence. Neighboring/overlapping regions may share their content. Extra dimensions can overfit and repeat geometry; any full prototype needs equal-parameter repeated-mean, stockC, R1 and geometry controls,3seeds and actual assignment effects. Known outcomes are used only for this prerequisite, never as runtime inputs. Reusedtrain26 is exploratory.

## Критерии приёмки (записаны до прогона)
- ❌ All130video/37296query/26600known bindings, exact previous region choices and source hashes. Admission requires lower paired-video95%CI of the fraction of known evaluate26 queries with a beneficial action and a noncorrect competing region, both four-bin resolved, strictly>0.02. Upper<0.02 stops this specific spatial-bin prototype for insufficient exposure; otherwise inconclusive. This is an oracle availability ceiling, not achieved2ppgain.  
  *runs/z13_spatial_bins_gate_20261025/report.json*

## Доказательства
- Мишень, равный контроль, критерий и бюджет до измерения: `runs/z13_spatial_bins_gate_20261025/protocol.json` · 

