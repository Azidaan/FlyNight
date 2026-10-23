---
aliases:
- 'З13: обучаемые соответствия пиков с отсутствующей парой'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P8
- метка/KeepTrack
- метка/engineering-future
- муха
исход: отрицательный
направление: FlyVis
начат: '2026-10-23'
---

# 🔴 З13: обучаемые соответствия пиков с отсутствующей парой

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_learned_association_v1`

Эта версия обучаемых соответствий не прошла все заранее заданные критерии. Это исход конкретного обучения и применения, не доказательство отсутствия информации об идентичности в FlyVis.

## Вопрос
Улучшит ли явное обучение соответствий причинное ведение кандидата цели сверх фиксированного сопоставления, карты и входа глаза?

## Гипотеза
Частичные истинные пары цели и искусственные пары кандидатов могут обучить различению соседей и отсутствующей пары; обычный временной контекст этого не проверял.

## Ожидалось (до прогона)
18 CPU моделей на104train,3сида;18 сетей,1150шагов,64реальных+64искусственныхпары в пакете. Оценка всех26train. До7200с+20с завершения, своё дерево до07:40; без ядра/GPU/dev/HOTA/75.

## Что сделано
2026-10-23

## Получено
runs/z13_learned_association_20261023/report.json, paired: {"D_activity-D_fixed": {"mean_per_video": -0.05488882860314511, "ci95": [-0.08749986089745915, -0.016291758388861228], "better": 2, "n": 26}, "D_activity-D_geometry": {"mean_per_video": -0.025032171997771892, "ci95": [-0.04785040378762505, 0.0015345465254136752], "better": 6, "n": 26}, "D_activity-D_old_activity": {"mean_per_video": -0.05242470177183708, "ci95": [-0.09425541199479817, 0.005358701182476033], "better": 4, "n": 26}, "D_activity-D_old_geometry": {"mean_per_video": -0.1009976940075792, "ci95": [-0.1341912327043433, -0.06376263389794061], "better": 3, "n": 26}, "D_activity-D_top1": {"mean_per_video": -0.053199974070833815, "ci95": [-0.09558929852852846, 0.0042140885773715425], "better": 4, "n": 26}, "D_activity-D_kalman": {"mean_per_video": -0.09497449354227624, "ci95": [-0.12109020236244097, -0.0671097862928177], "better": 1, "n": 26}, "C_activity-C_fixed": {"mean_per_video": -0.06547369758122346, "ci95": [-0.09902625831287545, -0.0324170726712572], "better": 4, "n": 26}, "C_activity-C_geometry": {"mean_per_video": -0.003694615850458785, "ci95": [-0.02175457053652508, 0.016655627055436287], "better": 8, "n": 26}, "C_activity-C_old_activity": {"mean_per_video": -0.12058676929560486, "ci95": [-0.14803321182209944, -0.09350733898566774], "better": 1, "n": 26}, "C_activity-C_old_geometry": {"mean_per_video": -0.1371265072952653, "ci95": [-0.16621744445597758, -0.10683517005544868], "better": 1, "n": 26}, "C_activity-C_top1": {"mean_per_video": -0.09838392255369957, "ci95": [-0.1250172311614474, -0.07317815057691596], "better": 0, "n": 26}, "C_activity-C_kalman": {"mean_per_video": -0.11639046209215503, "ci95": [-0.14454976676676345, -0.08801186947966366], "better": 0, "n": 26}, "R1_activity-R1_fixed": {"mean_per_video": -0.06336618191102357, "ci95": [-0.08629915282292148, -0.04112496030296784], "better": 3, "n": 26}, "R1_activity-R1_geometry": {"mean_per_video": -0.02365081555051722, "ci95": [-0.0400682831791365, -0.0064727856665080345], "better": 5, "n": 26}, "R1_activity-R1_old_activity": {"mean_per_video": -0.09659216413005892, "ci95": [-0.12049383202784242, -0.07376935986544511], "better": 0, "n": 26}, "R1_activity-R1_old_geometry": {"mean_per_video": -0.10718249629754034, "ci95": [-0.13090379923208237, -0.0844232616398174], "better": 0, "n": 26}, "R1_activity-R1_top1": {"mean_per_video": -0.03498848497341034, "ci95": [-0.06974598163471801, 0.017813128754163438], "better": 3, "n": 26}, "R1_activity-R1_kalman": {"mean_per_video": -0.11127064357364545, "ci95": [-0.136770647237028, -0.08723132240860577], "better": 0, "n": 26}, "D_activity-R1_activity": {"mean_per_video": 0.14797050747377755, "ci95": [0.09913354311781528, 0.2070400410146046], "better": 23, "n": 26}, "C_activity-R1_activity": {"mean_per_video": 0.10575863699653279, "ci95": [0.06492153602900354, 0.15348717667810963], "better": 22, "n": 26}, "D_activity-C_activity": {"mean_per_video": 0.04221187047724477, "ci95": [0.016780086560881587, 0.06925319555702937], "better": 19, "n": 26}}. pair_audit.json: {"D": {"real_pairs": 2875, "synthetic_source_frames": 159026, "counts": {"gap6_matched": 1315, "gap6_missing": 1054, "gap6_appeared": 462, "gap5_missing": 5, "gap5_matched": 16, "gap2_matched": 3, "gap2_missing": 2, "gap3_missing": 1, "gap4_matched": 4, "gap4_missing": 1, "gap1_matched": 4, "gap1_missing": 2, "gap1_appeared": 2, "gap5_appeared": 4}}, "C": {"real_pairs": 2825, "synthetic_source_frames": 159026, "counts": {"gap6_appeared": 520, "gap6_matched": 1219, "gap6_missing": 1047, "gap5_matched": 13, "gap5_missing": 6, "gap2_missing": 2, "gap3_missing": 1, "gap4_matched": 4, "gap1_matched": 4, "gap1_appeared": 2, "gap1_missing": 2, "gap5_appeared": 4, "gap2_matched": 1}}, "R1": {"real_pairs": 1694, "synthetic_source_frames": 159026, "counts": {"gap6_matched": 524, "gap6_missing": 726, "gap6_appeared": 414, "gap5_matched": 5, "gap5_appeared": 6, "gap5_missing": 7, "gap2_appeared": 1, "gap4_matched": 4, "gap1_matched": 3, "gap1_missing": 1, "gap1_appeared": 1, "gap2_matched": 2}}}. Все18моделей,1150шагов,73600реальных и73600искусственныхпар на модель; воспроизведение старого соответствия на8326строках каждого ядра точное.

## Интерпретация
Эта версия обучаемых соответствий не прошла все заранее заданные критерии. Это исход конкретного обучения и применения, не доказательство отсутствия информации об идентичности в FlyVis.

## Решение
Не заменять рабочий считыватель этой реализацией. Не подбирать порог или лучший сид по train26. Следующее действие должно отличаться механизмом и иметь проверяемую мишень.

## Нельзя заявлять
Не полная KeepTrack; соседние пики могут принадлежать одному объекту. Синтетическое удаление не доказывает работу при реальной окклюзии; пары с разрывом до6кадров отличаются от30Гц применения. Train26 поисковый, отказ не калиброван, HOTA не измерена.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-23 (eeafc17bae2c8d81b2a5fa80f2c4c9747d7cc734) — Полный проверенный исход явного соответствия.
  > Протокол до результатов: явные частичные соответствия и пустая пара, без выдуманных идентичностей соседей.

## Критерии приёмки (записаны до прогона)
- ✅ Тесты причинности/пустой пары/масок, воспроизведение фиксированного контроля, состав пар,18моделей и хеши проверены.  
  *runs/z13_learned_association_20261023/report.json; reproduction.json; pair_audit.json; controller_result.json*
- ❌ D learned−D fixed >=2п.п. и нижняя95%повидеоДИ>0.  
  *runs/z13_learned_association_20261023/report.json; reproduction.json; pair_audit.json; controller_result.json*
- ❌ D learned лучше D без активности, R1 learned и прежней карты104D: все нижние95%повидеоДИ>0.  
  *runs/z13_learned_association_20261023/report.json; reproduction.json; pair_audit.json; controller_result.json*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [useful_center_fraction↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%D0%94%D0%BE%D0%BB%D1%8F%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D1%8B%D1%85%20%D1%86%D0%B5%D0%BD%D1%82%D1%80%D0%BE%D0%B2%2C%20%D1%81%D1%80%D0%B5%D0%B4%D0%BD%D0%B5%D0%B5%20%D0%BF%D0%BE%20%D0%B2%D0%B8%D0%B4%D0%B5%D0%BE) | 0.2016858362324934 | 0.2565746648356385 | train26 D learned association | `runs/z13_learned_association_20261023/report.json` · summary.D_activity.useful_mean_video |

## Доказательства
- Протокол, контроли и критерии до прогона: `runs/z13_learned_association_20261023/protocol.json` · 
- Предыдущий мягкий контекст не улучшил равный контроль: `runs/z13_temporal_candidates_20261023/report.json` · 
- Все парные исходы: `runs/z13_learned_association_20261023/report.json` · 
- Точное воспроизведение фиксированного контроля: `runs/z13_learned_association_20261023/reproduction.json` · 
- Настоящие размеченные пары и пробелы: `runs/z13_learned_association_20261023/pair_audit.json` · 
- Своё дерево завершено: `runs/z13_learned_association_20261023/controller_result.json` · 

## Код
- `tracking/z13_learned_association.py`
- `scripts/z13_learned_association_20261023.py`
- `tests/test_z13_learned_association.py`

