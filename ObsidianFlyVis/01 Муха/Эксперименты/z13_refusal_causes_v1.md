---
aliases:
- 'З13: причины неверного вмешательства — фиксированное правило и запас кандидатов'
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

# 🟢 З13: причины неверного вмешательства — фиксированное правило и запас кандидатов

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_refusal_causes_v1`

Фиксированный отказ центрированнойD пропускает все полезныеtop1в этой проверке;проверенные выбранные действия нейтральны.

## Вопрос
With the existing fixed top1% policy, do high map peaks select useful first-association corrections? Is lack of benefit caused by unnecessary intervention, missed useful top1, wrong candidate rank or no useful proposal?

## Гипотеза
Неверное вмешательство может возникать на разных узлах; высота пика не равна ожидаемой пользе относительно исходного назначения.

## Ожидалось (до прогона)
All8326 cached natural rows for eachC/D/R1; UID/video/frame/track binding to exact assignment-effect rows. Verify all saved top1 centers within0.5px and peak scores within0.001 before joining effects. Keep saved center tau unchanged, no optimization. All known5822rows: partition baseline correct/top1help/onlyotherhelp/nonehelp, split by fixed decision. Unknown2504reported separately. Rank metrics descriptive only, no fit. Compare aggregate runtime metadata on existing20val without using outcomes or tuning thresholds.

## Что сделано
2026-10-25

## Получено
runs/z13_refusal_causes_20261025/report.json → arms/D:из84выбранных64с известным исходом,20с неизвестным;полезных0,вредных0,нейтральных64. Все48полезныхtop1отвергнуты;ещё58ситуаций имеют полезныйинойкандидат,а895ошибочныхисходныхназначенийнеисправляетниодинизтрёх. runs/z13_refusal_causes_20261025/trace/report.json → reason counts: {"selected": {"no_correct_labelled_detection": 62, "correct_detection_assigned": 1, "correct_detection_only_below_high_threshold": 1}, "missed_beneficial_top1": {"correct_detection_only_below_high_threshold": 20, "no_correct_labelled_detection": 7, "correct_detection_assigned": 21}, "old_center_useful_harmful": {"correct_detection_only_below_high_threshold": 15, "correct_detection_cost_above_match_threshold": 2}} runs/z13_refusal_causes_20261025/explanation.json:выбранныеDпоkind={'0': 62, '-1': 20, '1': 2};отрицательные={'target_not_annotated_in_frame': 62}. Контрпример UID34278:ошибкацентра10.238265730626086→7.6474182069024605px,стоимостьправильнойдетекции0.766836404800415→0.8138771653175354при пороге0,8. Это отвергнутыйкандидат,нефактическоевмешательствополитики. runs/z13_refusal_causes_20261025/normalization/report.json → Dcentered_peak: {"selected": 84, "known": 48, "neutral": 47, "unknown": 36, "beneficial": 1};Dcentered_peak-Dz: {"mean_per_video": 8.471704506946798e-05, "ci95": [0.0, 0.000254151135208404], "better": 1, "n": 26}. primary_pass=False.

## Интерпретация
Управление запуском и отказ различны:трекер вызываетмухутолькопослепотери,центрированнаяDразрешаетзаменуцентрапоотносительновысокомупику,прежняяDиспользуетголовусметкойточностицентра. Ни одноизэтихправилсамопосебенепредсказываетвыигрышполногоназначенияотносительноисходноготрекера. Декомпозициядоказываетнесоответствиепровереннойполитикиоднокадровойпользе,нонеустанавливаетпричинупредставленийиливсейHOTA.

## Решение
Не внедрять замену нормировки: заранее записанный критерий не пройден. Не продолжать подбор скалярного порога на этих26видео и не повторять прежние отрицательные проверки абсолютного отклика/метки локализации/семантической выборки. Текущий разбор завершён: неподходящая цель отказа и ограниченность предложений разделены, но конкретная новая модель ещё не принята. Следующий ремонт должен менять внутренний узел отказа или предлагаемые муху состояния, иметь новый проверяемый механизм и контроль без мухи; прежний непройденный гейт обучения по меткам назначения сохраняется.

## Нельзя заявлять
Episodes on unchanged baseline states and sparse original labels, reused train26; no closed-loop counterfactual or HOTA. Top1selection here is only centered readout; legacy quality head remains separate. Statistics over known labels are not unconditional safety; unknown outcomes stay unknown.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-25 (69a3cb978a5971b62a657662dcb1e92aa927cbaf) — Завершён и проверен разбор существующей политики и точных назначений.
  > Диагностика существующей политики; без нового обучения и изменения порога.
- **Получено**, 2026-10-25 (69a3cb978a5971b62a657662dcb1e92aa927cbaf) — Уточнён смысл отрицательной метки и проверен конкретный механизм потери правильной детекции.
  > runs/z13_refusal_causes_20261025/report.json → arms/D:из84выбранных64с известным исходом,20с неизвестным;полезных0,вредных0,нейтральных64. Все48полезныхtop1отвергнуты;ещё58ситуаций имеют полезныйинойкандидат,а895ошибочныхисходныхназначенийнеисправляетниодинизтрёх. runs/z13_refusal_causes_20261025/trace/report.json → reason counts: {"selected": {"no_correct_labelled_detection": 62, "correct_detection_assigned": 1, "correct_detection_only_below_high_threshold": 1}, "missed_beneficial_top1": {"correct_detection_only_below_high_threshold": 20, "no_correct_labelled_detection": 7, "correct_detection_assigned": 21}, "old_center_useful_harmful": {"correct_detection_only_below_high_threshold": 15, "correct_detection_cost_above_match_threshold": 2}}
- **Получено**, 2026-10-25 (69a3cb978a5971b62a657662dcb1e92aa927cbaf) — Добавлен заранее зарегистрированный контроль нормировки; прежний текст сохранён.
  > runs/z13_refusal_causes_20261025/report.json → arms/D:из84выбранных64с известным исходом,20с неизвестным;полезных0,вредных0,нейтральных64. Все48полезныхtop1отвергнуты;ещё58ситуаций имеют полезныйинойкандидат,а895ошибочныхисходныхназначенийнеисправляетниодинизтрёх. runs/z13_refusal_causes_20261025/trace/report.json → reason counts: {"selected": {"no_correct_labelled_detection": 62, "correct_detection_assigned": 1, "correct_detection_only_below_high_threshold": 1}, "missed_beneficial_top1": {"correct_detection_only_below_high_threshold": 20, "no_correct_labelled_detection": 7, "correct_detection_assigned": 21}, "old_center_useful_harmful": {"correct_detection_only_below_high_threshold": 15, "correct_detection_cost_above_match_threshold": 2}} runs/z13_refusal_causes_20261025/explanation.json:выбранныеDпоkind={'0': 62, '-1': 20, '1': 2};отрицательные={'target_not_annotated_in_frame': 62}. Контрпример UID34278:ошибкацентра10.238265730626086→7.6474182069024605px,стоимостьправильнойдетекции0.766836404800415→0.8138771653175354при пороге0,8. Это отвергнутыйкандидат,нефактическоевмешательствополитики.
- **Решение**, 2026-10-25 (69a3cb978a5971b62a657662dcb1e92aa927cbaf) — Завершены три диагностические ступени; простой скалярный ремонт не принят.
  > Не подбирать порог по исходам26/20val. Следующийузелдляправки — считываниеотносительнойпользысучётомисходногопрогнозаиназначения,присохраненииобразцаиконкурирующихвариантов;контрольбезмухиобязателен. Сначала проверитьнаготовыхкартах,сохраниласьлитребуемаяинформация,неповторятьпрежнийпроваленныйпилотобученияпометкампоследствий. Дополнительныеизмерениязаписыватьотдельнымпротоколом,невыдаватьновуюцельзаужеисправленнуюмодель.

## Критерии приёмки (записаны до прогона)
- ✅ All26videos and8326rows each; known/unknown exhaustive, all four categories partition known, selected and refused exhaustive. Exact reproduction of original assignment counts and SHA. No legacyD center is joined to centeredD effects: distinct outputs. Runtime metadata frozen.  
  *runs/z13_refusal_causes_20261025/report.json;runs/z13_refusal_causes_20261025/trace/report.json;diagnosis.json*

## Доказательства
- Причины и правила разбора записаны до расчёта: `runs/z13_refusal_causes_20261025/protocol.json` · 
- Фиксированный отбор,пропуски,кандидаты и прежняя метрика: `runs/z13_refusal_causes_20261025/report.json` · 
- Полные назначения,конкуренты и точное повторение26исходныхвыходов: `runs/z13_refusal_causes_20261025/trace/report.json` · 
- Независимая проверка счётчиков и механизмов: `runs/z13_refusal_causes_20261025/diagnosis.json` · 
- Смысл отрицательных меток и точный контрпример геометрии: `runs/z13_refusal_causes_20261025/explanation.json` · 
- Контроль роли деления на разброс карты, отрицательный критерий перехода: `runs/z13_refusal_causes_20261025/normalization/report.json` · 

## Код
- `scripts/z13_refusal_causes_20261025.py`

