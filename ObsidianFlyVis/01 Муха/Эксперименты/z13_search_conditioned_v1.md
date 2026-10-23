---
aliases:
- 'З13: модель цели с учётом текущего поиска'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P8
- метка/ToMP
- метка/engineering-future
- муха
исход: отрицательный
направление: FlyVis
начат: '2026-10-23'
---

# 🔴 З13: модель цели с учётом текущего поиска

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_search_conditioned_v1`

Эта адаптация предсказания модели цели не прошла все заданные критерии. Это не отрицание механизма ToMP или всей линии FlyVis; наш средний образец и3пика существенно беднее исходной модели.

## Вопрос
Поможет ли текущий поиск при предсказании весов модели цели сверх равного контроля без такого доступа?

## Гипотеза
Модель цели может учитывать текущих конкурентов через предсказываемые веса; проверяется доступ поиска к весам при одинаковой архитектуре.

## Ожидалось (до прогона)
36CPUмоделей,104train,26оценки,3сида;5400с+20с завершения,до07:40. Без ядра/GPU/dev/HOTA/75.

## Что сделано
2026-10-23

## Получено
runs/z13_search_conditioned_20261023/report.json, paired: {"D_search_activity-D_template_activity": {"mean_per_video": -0.018743147529366712, "ci95": [-0.03778956437627713, -0.0028644461985393487], "better": 3, "n": 26}, "D_search_geometry-D_template_geometry": {"mean_per_video": -0.004981317598724001, "ci95": [-0.012400514154813334, 0.002519828441919836], "better": 7, "n": 26}, "D_search_activity-D_old_activity": {"mean_per_video": -0.001871592399210761, "ci95": [-0.024034530157217085, 0.016081631143938523], "better": 11, "n": 26}, "D_search_activity-D_old_geometry": {"mean_per_video": -0.050444584634952896, "ci95": [-0.0964525836067944, -0.01563636576872383], "better": 4, "n": 26}, "D_search_activity-D_search_geometry": {"mean_per_video": -0.03388488102832433, "ci95": [-0.05727396262505543, -0.011470120331469746], "better": 4, "n": 26}, "D_search_activity-D_top1": {"mean_per_video": -0.0026468646982074928, "ci95": [-0.0279846274372211, 0.018638469735263726], "better": 12, "n": 26}, "D_search_activity-D_kalman": {"mean_per_video": -0.04442138416964991, "ci95": [-0.09337905455168898, -0.005572705475002118], "better": 9, "n": 26}, "C_search_activity-C_template_activity": {"mean_per_video": 0.013708671927309986, "ci95": [-0.013619106875825838, 0.05847881258289832], "better": 7, "n": 26}, "C_search_geometry-C_template_geometry": {"mean_per_video": -0.006281016211890169, "ci95": [-0.014032004850427344, 0.002487975956975934], "better": 5, "n": 26}, "C_search_activity-C_old_activity": {"mean_per_video": 0.0037344121640624514, "ci95": [-0.015128194006023608, 0.020626754554299583], "better": 11, "n": 26}, "C_search_activity-C_old_geometry": {"mean_per_video": -0.012805325835597998, "ci95": [-0.03499629272286068, 0.01010711776782421], "better": 8, "n": 26}, "C_search_activity-C_search_geometry": {"mean_per_video": -0.0022380191646588785, "ci95": [-0.03470618912319035, 0.04594173079216651], "better": 5, "n": 26}, "C_search_activity-C_top1": {"mean_per_video": 0.0259372589059677, "ci95": [0.0075027877753365, 0.0423418526494519], "better": 18, "n": 26}, "C_search_activity-C_kalman": {"mean_per_video": 0.007930719367512258, "ci95": [-0.016704308573899387, 0.03201845701747899], "better": 15, "n": 26}, "R1_search_activity-R1_template_activity": {"mean_per_video": -0.03265742309399419, "ci95": [-0.07427427786858165, -0.008530843663488366], "better": 3, "n": 26}, "R1_search_geometry-R1_template_geometry": {"mean_per_video": 0.00011560051082816332, "ci95": [-0.004434975364722935, 0.004624823242180112], "better": 4, "n": 26}, "R1_search_activity-R1_old_activity": {"mean_per_video": -0.01950844948576008, "ci95": [-0.0640473135911562, 0.0077360693424451425], "better": 8, "n": 26}, "R1_search_activity-R1_old_geometry": {"mean_per_video": -0.030098781653241483, "ci95": [-0.07340138948620151, -0.003908002862375084], "better": 5, "n": 26}, "R1_search_activity-R1_search_geometry": {"mean_per_video": -0.02694337603320117, "ci95": [-0.0706397090281761, 0.0004583116720656666], "better": 4, "n": 26}, "R1_search_activity-R1_top1": {"mean_per_video": 0.04209522967088852, "ci95": [0.026345740675709432, 0.05865340898547876], "better": 19, "n": 26}, "R1_search_activity-R1_kalman": {"mean_per_video": -0.03418692892934661, "ci95": [-0.0798504409527372, -0.004990330158709697], "better": 4, "n": 26}, "D_search_activity-R1_search_activity": {"mean_per_video": 0.12143990220210502, "ci95": [0.08435491306041362, 0.16538096512906617], "better": 23, "n": 26}, "D_search_activity-C_search_activity": {"mean_per_video": -0.031556201609796206, "ci95": [-0.07932758571911042, 0.003892326296668395], "better": 7, "n": 26}, "C_search_activity-R1_search_activity": {"mean_per_video": 0.15299610381190124, "ci95": [0.09428472873861557, 0.2378617968493633], "better": 23, "n": 26}}. Все36моделей завершены,параметров на модель 7273;1150шагов и145950предъявлений;18старыхмоделей и32старыхшага воспроизведены.

## Интерпретация
Эта адаптация предсказания модели цели не прошла все заданные критерии. Это не отрицание механизма ToMP или всей линии FlyVis; наш средний образец и3пика существенно беднее исходной модели.

## Решение
Не заменять рабочий считыватель этой версией; не выбирать лучший сид или подбирать по train26. Следующая правка должна иметь иной механизм и измеримую мишень.

## Нельзя заявлять
Не полная ToMP: только средний образец и3пика, без плотного поля/размеров/регрессии рамки. Train26 повторно использован, нет HOTA и калиброванного отказа; большая ёмкость сама по себе не вклад мухи.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-23 (b32afc8e3f6e230311983d6ed807fbfcb8f5fad7) — Полный проверенный исход предсказателя весов.
  > Критерии до результатов; два одинаковых предсказателя весов различаются только доступом поиска.

## Критерии приёмки (записаны до прогона)
- ✅ Тесты доступа к весам/перестановки/масок,18контролей,32старыхшага,36моделей и все хеши проверены.  
  *runs/z13_search_conditioned_20261023/report.json; reproduction.json; timing.json; controller_result.json*
- ❌ Dsearch-Dtemplate >=2п.п. и нижняя95%повидеоДИ>0.  
  *runs/z13_search_conditioned_20261023/report.json; reproduction.json; timing.json; controller_result.json*
- ❌ Dsearch лучше старой Dс активностью,старой картыD,R1search и Dsearchбезактивности: все нижние95%ДИ>0.  
  *runs/z13_search_conditioned_20261023/report.json; reproduction.json; timing.json; controller_result.json*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [useful_center_fraction↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%D0%94%D0%BE%D0%BB%D1%8F%20%D0%BF%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D1%8B%D1%85%20%D1%86%D0%B5%D0%BD%D1%82%D1%80%D0%BE%D0%B2%2C%20%D1%81%D1%80%D0%B5%D0%B4%D0%BD%D0%B5%D0%B5%20%D0%BF%D0%BE%20%D0%B2%D0%B8%D0%B4%D0%B5%D0%BE) | 0.2522389456051197 | 0.27098209313448646 | train26 search-conditioned D | `runs/z13_search_conditioned_20261023/report.json` · summary.D_search_activity.useful_mean_video |

## Доказательства
- Один фактор внутри основного сравнения, критерии и хеши: `runs/z13_search_conditioned_20261023/protocol.json` · 
- Все парные исходы: `runs/z13_search_conditioned_20261023/report.json` · 
- Воспроизведение старого чтения: `runs/z13_search_conditioned_20261023/reproduction.json` · 
- Своё дерево завершено: `runs/z13_search_conditioned_20261023/controller_result.json` · 

## Код
- `tracking/z13_search_conditioned.py`
- `tests/test_z13_search_conditioned.py`
- `scripts/z13_search_conditioned_20261023.py`

