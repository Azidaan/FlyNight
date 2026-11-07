---
aliases:
- 'З13: односторонняя ошибка фона в модели цели'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P8
- муха
- проверка-гипотезы/завершена
исход: отрицательный
направление: FlyVis
начат: '2026-10-22'
---

# 🟢 З13: односторонняя ошибка фона в модели цели

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_hinge_target_v1`

Общие критерии не выполнены; эта односторонняя ошибка не принята. Частные сравнения сохраняются отдельно; по этому исходу не закрывается линия FlyVis.

## Вопрос
Поможет ли прекращение штрафа уже подавленного фона улучшить D сверх согласованной балансировки, включая отбор полезных ответов?

## Гипотеза
Освобождение от регрессии точных низких значений фона может улучшить различающую модель; исправление отказа не установлено.

## Ожидалось (до прогона)
D/C/R1,полные26train,одна функция ошибки; прежние гауссова цель,веса0,5/0,5,68признаков,история,коэффициент регуляризации. CPUдо7200с,сначала стоимость/воспроизведение.

## Что сделано
2026-10-23

## Получено
runs/z13_hinge_target_20261023/report.json → paired={'D_hinge-D_balanced': {'mean_per_video': 0.0020485259100313886, 'ci95': [-0.012061937999979027, 0.016012496173129245], 'better': 12, 'n': 26}, 'C_hinge-C_balanced': {'mean_per_video': 0.002128685086037144, 'ci95': [-0.020024462058896613, 0.024375551762804513], 'better': 11, 'n': 26}, 'R1_hinge-R1_balanced': {'mean_per_video': 0.000826900280452192, 'ci95': [-0.002760819427486095, 0.004610002424210071], 'better': 3, 'n': 26}, 'D_hinge-R1_hinge': {'mean_per_video': 0.20483198490067345, 'ci95': [0.1461247022723927, 0.28864451543479513], 'better': 25, 'n': 26}, 'C_hinge-R1_hinge': {'mean_per_video': 0.15773169160468387, 'ci95': [0.12081953112002014, 0.19974649208895573], 'better': 26, 'n': 26}, 'D_hinge-C_hinge': {'mean_per_video': 0.04710029329598963, 'ci95': [0.0006180256312590165, 0.10138786067179091], 'better': 16, 'n': 26}}; selected_paired={'D_hinge-D_balanced': {'mean_per_video': -0.0016965385182439448, 'ci95': [-0.004815041636747064, 0.00047483380816714147], 'better': 1, 'n': 26}, 'C_hinge-C_balanced': {'mean_per_video': 0.0009586932995178518, 'ci95': [-0.001681760223955282, 0.0032705796686092275], 'better': 5, 'n': 26}, 'R1_hinge-R1_balanced': {'mean_per_video': 0.0, 'ci95': [0.0, 0.0], 'better': 0, 'n': 26}}; summary={'D_balanced': {'rows': 8326, 'useful_mean_video': 0.28985455084403927, 'coverage': {'0.01': {'selected': 84, 'known': 61, 'unknown': 23, 'useful': 2, 'harmful': 59, 'useful_share_known': 0.03278688524590164}, '0.02': {'selected': 167, 'known': 131, 'unknown': 36, 'useful': 3, 'harmful': 128, 'useful_share_known': 0.022900763358778626}, '0.05': {'selected': 417, 'known': 332, 'unknown': 85, 'useful': 7, 'harmful': 325, 'useful_share_known': 0.02108433734939759}}, 'selected': 84, 'known': 61, 'unknown': 23, 'selected_useful': 2, 'selected_absent': 58}, 'D_hinge': {'rows': 8326, 'useful_mean_video': 0.2919030767540707, 'coverage': {'0.01': {'selected': 84, 'known': 67, 'unknown': 17, 'useful': 1, 'harmful': 66, 'useful_share_known': 0.014925373134328358}, '0.02': {'selected': 167, 'known': 142, 'unknown': 25, 'useful': 1, 'harmful': 141, 'useful_share_known': 0.007042253521126761}, '0.05': {'selected': 417, 'known': 339, 'unknown': 78, 'useful': 3, 'harmful': 336, 'useful_share_known': 0.008849557522123894}}, 'selected': 84, 'known': 67, 'unknown': 17, 'selected_useful': 1, 'selected_absent': 66}, 'C_balanced': {'rows': 8326, 'useful_mean_video': 0.24267409837204393, 'coverage': {'0.01': {'selected': 84, 'known': 65, 'unknown': 19, 'useful': 2, 'harmful': 63, 'useful_share_known': 0.03076923076923077}, '0.02': {'selected': 167, 'known': 128, 'unknown': 39, 'useful': 6, 'harmful': 122, 'useful_share_known': 0.046875}, '0.05': {'selected': 417, 'known': 330, 'unknown': 87, 'useful': 14, 'harmful': 316, 'useful_share_known': 0.04242424242424243}}, 'selected': 84, 'known': 65, 'unknown': 19, 'selected_useful': 2, 'selected_absent': 58}, 'C_hinge': {'rows': 8326, 'useful_mean_video': 0.24480278345808107, 'coverage': {'0.01': {'selected': 84, 'known': 64, 'unknown': 20, 'useful': 7, 'harmful': 57, 'useful_share_known': 0.109375}, '0.02': {'selected': 167, 'known': 127, 'unknown': 40, 'useful': 8, 'harmful': 119, 'useful_share_known': 0.06299212598425197}, '0.05': {'selected': 417, 'known': 313, 'unknown': 104, 'useful': 11, 'harmful': 302, 'useful_share_known': 0.03514376996805112}}, 'selected': 84, 'known': 64, 'unknown': 20, 'selected_useful': 7, 'selected_absent': 54}, 'R1_balanced': {'rows': 8326, 'useful_mean_video': 0.08624419157294504, 'coverage': {'0.01': {'selected': 84, 'known': 60, 'unknown': 24, 'useful': 2, 'harmful': 58, 'useful_share_known': 0.03333333333333333}, '0.02': {'selected': 167, 'known': 124, 'unknown': 43, 'useful': 3, 'harmful': 121, 'useful_share_known': 0.024193548387096774}, '0.05': {'selected': 417, 'known': 300, 'unknown': 117, 'useful': 4, 'harmful': 296, 'useful_share_known': 0.013333333333333334}}, 'selected': 84, 'known': 60, 'unknown': 24, 'selected_useful': 2, 'selected_absent': 43}, 'R1_hinge': {'rows': 8326, 'useful_mean_video': 0.08707109185339722, 'coverage': {'0.01': {'selected': 84, 'known': 59, 'unknown': 25, 'useful': 2, 'harmful': 57, 'useful_share_known': 0.03389830508474576}, '0.02': {'selected': 167, 'known': 122, 'unknown': 45, 'useful': 3, 'harmful': 119, 'useful_share_known': 0.02459016393442623}, '0.05': {'selected': 417, 'known': 299, 'unknown': 118, 'useful': 4, 'harmful': 295, 'useful_share_known': 0.013377926421404682}}, 'selected': 84, 'known': 59, 'unknown': 25, 'selected_useful': 2, 'selected_absent': 42}}. Полные D/C/R1,8326строк каждый; все источники/сходимость/исходный контроль проверены.

## Интерпретация
Общие критерии не выполнены; эта односторонняя ошибка не принята. Частные сравнения сохраняются отдельно; по этому исходу не закрывается линия FlyVis.

## Решение
Only if all main gates pass proceed to a separate fixed transfer/stand protocol. Otherwise stop this one-sided residual implementation; retain accepted prior version and record localization versus refusal separately. No endless scalar selector search and no core training,dev,HOTA,protected75 here. Old90%/50criterion unchanged.

## Нельзя заявлять
Не полная DiMP и не её точная формула; не HOTA,не калиброванный отказ,не обучение734параметров. train26 используется повторно,старые90%/50не заменены.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-23 (3017939b11cd385af683cd502132c6565395104f) — Проверенный полный либо частичный CPU-исход.
  > До расчёта зарегистрирована отдельная замена остаточной ошибки фона.

## Критерии приёмки (записаны до прогона)
- ✅ Reproduce previous balanced control for all8326rows/arm: center<0.5px,score<0.001,same lambda. Every observed solver converges at frozen tolerance; hashes unchanged.  
  *runs/z13_hinge_target_20261023/report.json*
- ❌ D hinge-D balanced mean paired-video useful-center fraction>=0.02 and95%lower>0 on all present train26 rows.  
  *runs/z13_hinge_target_20261023/report.json*
- ❌ D hinge-D balanced95%lower>0 for selected_success_fraction_positive at equal1%quota84.  
  *runs/z13_hinge_target_20261023/report.json*
- ✅ D hinge-R1 hinge95%lower>0 on all present train26 rows, identical history and objective.  
  *runs/z13_hinge_target_20261023/report.json*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [useful_center_fraction↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%D0%94%D0%BE%D0%BB%D1%8F%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D1%8B%D1%85%20%D1%86%D0%B5%D0%BD%D1%82%D1%80%D0%BE%D0%B2%2C%20%D1%81%D1%80%D0%B5%D0%B4%D0%BD%D0%B5%D0%B5%20%D0%BF%D0%BE%20%D0%B2%D0%B8%D0%B4%D0%B5%D0%BE) | 0.2919030767540707 | 0.28985455084403927 | train26;D hinge vs balanced | `runs/z13_hinge_target_20261023/report.json` · summary.D_hinge.useful_mean_video |

## Доказательства
- Одна замена с критериями до исхода: `runs/z13_hinge_target_20261023/protocol.json` · 
- Обязательный согласованный контроль и непройденный отказ: `runs/z13_balanced_ridge_20261022/report.json` · 
- Фактическое окончание своего дерева: `runs/z13_hinge_target_20261023/controller_result.json` · 
- Полный первичный исход и парные контроли: `runs/z13_hinge_target_20261023/report.json` · 

## Код
- `tracking/z13_hinge_target.py`
- `tests/test_z13_hinge_target.py`
- `scripts/z13_hinge_target_20261023.py`

## Коммиты
- `3017939b11cd385af683cd502132c6565395104f`

