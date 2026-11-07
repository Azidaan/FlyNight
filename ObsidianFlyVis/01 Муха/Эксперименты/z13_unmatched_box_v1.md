---
aliases:
- 'Применение полного ответа FlyVis: неназначенный трек и свободная детекция'
tags:
- эксперимент
- исход/completed_negative
- направление/flyvis
- метка/fly-problem-P9
- муха
- проверка-гипотезы/завершена
исход: отрицательный
направление: FlyVis
начат: '2026-10-25'
---

# 🟢 Применение полного ответа FlyVis: неназначенный трек и свободная детекция

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** отрицательный · `z13_unmatched_box_v1`

Защиты применения полной рамки: основной критерий не пройден.

## Вопрос
Защищает ли наблюдаемое исходное назначение от вреда полной поправки при пользе сверх равных контролей?

## Гипотеза
Рабочая гипотеза: часть вреда создаёт полная поправка там, где исходное назначение уже существует; доступное состояние сопоставления может позволить безопаснее применить полный ответ.

## Ожидалось (до прогона)
Test two observable application contracts for the existing target-model full-box output. Full-strength action is permitted only if the original first-stage assignment left this row unmatched; stricter free rule also requires proposed detection not assigned to any other row. This is an application-boundary experiment, not an intrinsic modernization of FlyVis or a service fix. Same original maps, region selection, NULL, support and full-XYWH proposal as prior prototype. Three fixed contracts perD/C/R1/geometry: unconditional full strength, unmatched-only, unmatched-and-free. Both new contracts use original baseline assignment computed before any intervention. No GT or labels in admission. Same complete assignment after one-row replacement. No alpha sweep: active alpha1, alpha0 only exact regression control.

## Что сделано
2026-10-25

## Получено
runs/z13_unmatched_box_20261025/report.json → paired/counts: {'paired': {'D_unmatched-baseline': {'mean_per_video': -0.27379394999387985, 'ci975': [-0.3204965669723698, -0.22829421249362486], 'better': 0, 'n': 26}, 'D_unmatched-D_full': {'mean_per_video': 0.0017334419676960494, 'ci975': [0.0006069269788106547, 0.0030617320087957083], 'better': 9, 'n': 26}, 'D_unmatched-Dold': {'mean_per_video': -0.27379394999387985, 'ci975': [-0.3204965669723698, -0.22829421249362486], 'better': 0, 'n': 26}, 'D_unmatched-C_unmatched': {'mean_per_video': 0.010748634157959612, 'ci975': [0.0022515573896269535, 0.019535873113856947], 'better': 15, 'n': 26}, 'D_unmatched-R1_unmatched': {'mean_per_video': 9.880690249142444e-05, 'ci975': [-0.020421025846625383, 0.020229428563579694], 'better': 14, 'n': 26}, 'D_unmatched-geometry_unmatched': {'mean_per_video': 0.007097143086132018, 'ci975': [-0.008522800181656803, 0.02204967107837681], 'better': 16, 'n': 26}, 'unmatched_effect_D-C': {'mean_per_video': -0.00040715549297372686, 'ci975': [-0.0009875611591976317, 6.4051328343759035e-18], 'better': 2, 'n': 26}, 'unmatched_effect_D-R1': {'mean_per_video': -0.007108339062048098, 'ci975': [-0.013535977579492249, -0.0029556527970782363], 'better': 3, 'n': 26}, 'D_free-baseline': {'mean_per_video': -0.025600724434621157, 'ci975': [-0.03944167432205979, -0.013909667776213289], 'better': 4, 'n': 26}, 'D_free-D_full': {'mean_per_video': 0.24992666752695478, 'ci975': [0.20137776581620978, 0.29740605076112125], 'better': 26, 'n': 26}, 'D_free-Dold': {'mean_per_video': -0.025600724434621157, 'ci975': [-0.03944167432205979, -0.013909667776213289], 'better': 4, 'n': 26}, 'D_free-C_free': {'mean_per_video': 0.0029762107210495386, 'ci975': [-0.001536032895427902, 0.00788585863144394], 'better': 11, 'n': 26}, 'D_free-R1_free': {'mean_per_video': 0.0008562087205631398, 'ci975': [-0.008683446645972068, 0.009390427849834912], 'better': 13, 'n': 26}, 'D_free-geometry_free': {'mean_per_video': -0.0019193574372834237, 'ci975': [-0.00952994901590414, 0.003923175334539082], 'better': 10, 'n': 26}, 'free_effect_D-C': {'mean_per_video': -0.0081795789298838, 'ci975': [-0.017136134916202216, 0.0006628020445067192], 'better': 9, 'n': 26}, 'free_effect_D-R1': {'mean_per_video': -0.006350937243976381, 'ci975': [-0.027843735134490573, 0.018245367372495068], 'better': 11, 'n': 26}}, 'counts': {'D_full': {'known': 5822, 'unknown': 2504, 'beneficial': 264, 'harmful': 1972}, 'D_unmatched': {'known': 5822, 'unknown': 2504, 'beneficial': 263, 'harmful': 1961}, 'D_free': {'known': 5822, 'unknown': 2504, 'beneficial': 90, 'harmful': 215}, 'C_full': {'known': 5822, 'unknown': 2504, 'beneficial': 265, 'harmful': 2035}, 'C_unmatched': {'known': 5822, 'unknown': 2504, 'beneficial': 264, 'harmful': 2021}, 'C_free': {'known': 5822, 'unknown': 2504, 'beneficial': 91, 'harmful': 237}, 'R1_full': {'known': 5822, 'unknown': 2504, 'beneficial': 182, 'harmful': 1931}, 'R1_unmatched': {'known': 5822, 'unknown': 2504, 'beneficial': 182, 'harmful': 1888}, 'R1_free': {'known': 5822, 'unknown': 2504, 'beneficial': 64, 'harmful': 206}, 'geometry_full': {'known': 5822, 'unknown': 2504, 'beneficial': 250, 'harmful': 2047}, 'geometry_unmatched': {'known': 5822, 'unknown': 2504, 'beneficial': 249, 'harmful': 2032}, 'geometry_free': {'known': 5822, 'unknown': 2504, 'beneficial': 86, 'harmful': 215}, 'Dold': {'known': 5822, 'unknown': 2504, 'beneficial': 0, 'harmful': 0}}}

## Интерпретация
Проверены два фиксированных контракта применения одного полного ответа. Они читают исходное назначение, а не разметку. Интервалы97,5% учитывают две альтернативы; парные C/R1/геометрия отделяют вклад мухи. Это применение сигнала, не внутренняя модернизация; повторныйtrain26 и первые назначения, не HOTA.

## Решение
If a guard passes all corrected criteria, preregister a separate frozen matched closed-loop gate; it still counts as application of FlyVis, not an intrinsic new model. If both fail, keep both negative and stop this rule family; do not tune confidence/displacement/admission thresholds on26. Report causes and full working results distinctly.

## Нельзя заявлять
The original tracker may be correctly unmatched when no compatible detection exists. A new box can create a false match or steal from a valid neighbor. The free rule may remove useful recovery of a detection incorrectly held by another track. Both guards may give the same effect without FlyVis, which would be an application improvement only. Reused26 was seen to formulate this new hypothesis; no independent validation claim. Внешний контракт применения не объявляется модернизацией ядра или считывания.

## Критерии приёмки (записаны до прогона)
- ✅ All46raw baseline outputs bit-exact, all8326evaluate and all fit canonical C/D/R1 map top1 centers<0.5px/scores<0.001, unchanged crops/pred/frame/label binding. Dold semantic outcomes exactly equal old replay for its selected top1, or neutral if refused. Complete mean restored, alpha0 no-action exact. All source hashes. First2fit conservative2*mean*44+90 must fit original1200s; completed videos not repeated. All alpha1guard decisions observable and alpha0exact. Prior unconditional-full outcomes are reproduced exactly on all26rows as part of the final audit, to isolate only the new admission contract. Two helper tests cover NULL, already-assigned row, and claimed detection. No scientific tolerance changed.  
  *runs/z13_unmatched_box_20261025/independent_audit.json*
- ❌ For either of2preregistered guards: Dguard-baseline mean>=2pp; every matched lower97.5%video interval>0 against baseline, unconditional Dfull, original Dold, same C/R1/geometry guard, and differential guard effects D-C and D-R1. Bonferroni for two guard alternatives: bootstrap5000seed1 quantiles0.0125/0.9875, field ci975. A better guard than a catastrophic unconditional policy is insufficient without absolute benefit and FlyVis-specific controls.  
  *runs/z13_unmatched_box_20261025/report.json*

## Доказательства
- Два заранее определённых правила, исправление множественности и фиксированная сила поправки: `runs/z13_unmatched_box_20261025/protocol.json` · 

