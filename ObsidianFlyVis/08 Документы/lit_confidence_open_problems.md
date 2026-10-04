---
tags:
- документ
источник: dissertation/lit_confidence_open_problems.md
---

# Уверенность детекций в трекинге: кто называет проблему, но не решает

Собрано 11.09.2026. Два прохода поиска: первый по рамке из 12 объектов
(взвешивание образцов, ранжирующие потери, метрики, связка детектора и
трекера), второй прицельный — искали **признания открытости**, а не решения.

Все цитаты ниже извлечены из полных текстов. Где текст не открылся — сказано
отдельно.

---

## 1. Главный вывод: пробел подтверждён

> **В проведённом поиске не найдено ни одной работы, которая на 2D-MOT-бенчмарке
> количественно отделила бы потери «объект детектирован верно, но балл ниже
> порога» от потерь «отклика нет вовсе».**

Наши 18.0% против 4.5% (стенд 960, полный val) аналога не имеют. Ближайшее —
SimpleTrack, но это 3D и LiDAR, и там нет ни оси возраста, ни заслонённости.

Формулировка «впервые» при этом **не обоснована**: отсутствие публикаций
новизной не является (правило проекта). Корректная форма для введения:

> «В проведённом поиске не найдено работ, количественно отделяющих потери
> недооценки уверенности от полной слепоты детектора на 2D-бенчмарках MOT, и не
> найдено работ, встраивающих цену объекта для трекинга в обучение детектора; на
> нерешённость отдельных частей этой задачи указано в [Feifel 2022; Luo 2021;
> Bashar 2026; Küppers 2022; Solano-Carrillo 2024; Sentinel 2026].»

---

## 2. Четыре цитаты, которые стоят всего обзора

### 2.1. Почему постобработка не могла сработать

**Küppers F. et al. Confidence Calibration for Object Detection and
Segmentation.** Springer, 2022, open access.
DOI: [10.1007/978-3-031-01233-4_8](https://doi.org/10.1007/978-3-031-01233-4_8)

> «the AUPRC is not affected by standard scaling methods (logistic/beta
> calibration) as these methods **perform a monotonically increasing mapping of
> the confidence estimates and thus do not affect the order of the samples**.»
> — раздел результатов

**Зачем нам.** Это литературное объяснение нашего отрицательного результата
(AUC 0.9025 → 0.9005). Монотонное отображение не может поднять **ни одного**
объекта выше порога — значит постобработочная калибровка исчерпана не по
случайности, а по устройству. Ставить под П1.

### 2.2. Прямое указание на наше направление

**Luo W. et al. Multiple object tracking: A literature review.** Artificial
Intelligence 293:103448, 2021. Рецензировано.
DOI: [10.1016/j.artint.2020.103448](https://doi.org/10.1016/j.artint.2020.103448)

> «the detection result for a specific video is not optimal since **the object
> detector is not trained for the given video**. This often limits the
> performance of multiple object tracking. **A customization of the object
> detector is necessary to improve MOT performance.**»
> — [[05 План/5. Что живо|§5.2]] Future Directions

**Зачем нам.** Рецензируемый обзор в топовом журнале называет обучение детектора
под трекинг открытым направлением. Лучшая опора для П4. Оговорка: они имеют в
виду адаптацию к домену конкретного видео, а не взвешивание по временной роли —
наш случай частный и не занятый.

### 2.3. Асимметрия порогов — не особенность нашего стенда

**Zhang Y. et al. ByteTrack.** ECCV 2022.
[arXiv:2110.06864](https://arxiv.org/abs/2110.06864)

> «we initialize new tracks **from the unmatched high score detection boxes**»
> — [[05 План/3.2. Трекер|§3.2]]

> «if we take every detection box into consideration, more false positives will
> be introduced immediately … **To the best of our knowledge, very few methods
> in MOT are able to handle this detection dilemma.**» — [[05 План/1. Тема, стенд и состояние на 11.09.2026|§1]]

> «**The detection score threshold τhigh is a sensitive hyper-parameter and needs
> to be carefully tuned** in the task of multi-object tracking.» — [[05 План/4. Что закрыто, и по какому основанию|§4]]

**Зачем нам.** Измеренное нами 0.25 против 0.1 — общепринятая конструкция, а не
частность стенда. Авторы сами называют дилемму почти никем не решённой.

### 2.4. Постобработочный подъём уверенности исчерпан

**Stanojević V., Todorović B. BoostTrack++.** Filomat 39(16), 2025.
DOI: [10.2298/FIL2516685S](https://doi.org/10.2298/FIL2516685S)

> «**3 Limitations of confidence boost based on detection of likely objects.** …
> **a different value of hyperparameter βc had to be specified depending on the
> dataset** … The most important issue is **the introduction of new IDs, which
> should not happen** if the detections with boosted confidence truly correspond
> to existing tracklets.»

> «this aspect of the problem is **mostly overlooked or mitigated by employing
> two-stage association** and utilizing low confidence detections in the second
> stage.» — abstract

**Зачем нам.** Ближайший инференс-аналог нашей идеи признаёт в Limitations, что
подъём балла на выводе требует подбора под датасет и **порождает новые
перебросы личности**. Разграничение по стадии вмешательства (мы — в обучении)
обязано быть в тексте явно.

---

## 3. Остальные признания открытости

| работа | что признано нерешённым | близость |
|---|---|---|
| **Feifel P. et al.**, AISafety 2022, [CEUR Vol-3215](https://ceur-ws.org/Vol-3215/11.pdf) | «**It is still open to what extent this requirement can be relaxed for DNNs for object tracking**»; порог «**must be determined post-hoc to PD training**» | высокая — единственная, кто прямо называет открытым перенос порога детектора в трекинг |
| **Bashar Mk. et al.**, обзор, [arXiv:2209.04796](https://arxiv.org/abs/2209.04796) | «**Most current trackers output point estimates with little calibrated uncertainty**»; «**Modular pipelines require hand-tuned heuristics for thresholds and birth/death policies**» — [[05 План/7. Действующий план|§7.9]] открытых направлений | высокая — перечислены ровно три наши оси |
| **Sentinel**, Sci Reports 2026, [DOI](https://doi.org/10.1038/s41598-026-43938-2) | «**managing detection confidence is not merely a matter of tuning thresholds**»; в Limitations — «**effectiveness can be limited in environments where the detector itself has low recall**» | высокая — решение на стороне ассоциации упирается в потолок детектора |
| **Solano-Carrillo E. et al.**, UTrack, ECCV 2024 W, [arXiv:2408.17098](https://arxiv.org/abs/2408.17098) | «There are methods to calibrate confidence estimates … **However, the resulting confidences are still point estimates. There are no variances that can be propagated to the quantities relevant for tracking**» | высокая — калибровки недостаточно для downstream |
| **Pang Z. et al.**, SimpleTrack, [arXiv:2111.09621](https://arxiv.org/abs/2111.09621) | «**the gap is mainly caused by the inconsistency between the scores and detection quality**»; «**purely pruning them by a score threshold will sacrifice the recall**» | высокая по постановке, средняя по домену (3D/LiDAR) |
| **OmniTrack++**, [arXiv:2511.00510](https://arxiv.org/abs/2511.00510) | «**A low initialization threshold tends to introduce noisy or short-lived tracks … whereas an excessively high threshold delays track activation and causes missed detections during early motion stages**» — [[05 План/5. Что живо|§5.3]].4 | высокая — дословно наш компромисс порога рождения |
| **Dave A. et al.**, [arXiv:2102.01066](https://arxiv.org/abs/2102.01066) | «**the default AP implementation neither achieves the goal of being independent per class, nor does it measure cross-category score calibration**» | средняя — объясняет, почему улучшение балла не видно в mAP |
| **Luiten J. et al.**, HOTA, IJCV 2021, [DOI](https://doi.org/10.1007/s11263-020-01375-2) | «**HOTA is designed to operate on final tracking results**» — метрика работает на уже отсечённом наборе, цена выбора порога в неё не входит | средняя — объясняет, почему наша потеря невидима стандартным метрикам |

---

## 4. Обходы вместо решений

| работа | обход | признание |
|---|---|---|
| **GHOST**, CVPR 2023, [arXiv:2206.04656](https://arxiv.org/abs/2206.04656) | ручной подбор порогов | критикуют ByteTrack за подбор по последовательностям и признают, что сами подбирают по датасетам, включая «**confidence threshold for new tracks**» |
| **OC-SORT**, CVPR 2023, [arXiv:2203.14360](https://arxiv.org/abs/2203.14360) | фиксированный порог ради простоты | «**Carefully tuning the parameters can probably further boost the performance … the adaptive detection threshold is proven useful**» |
| **MOTRv2**, CVPR 2023, [arXiv:2211.09791](https://arxiv.org/abs/2211.09791) | внешний YOLOX вместо совместного обучения | «**One major limitation of the end-to-end MOT frameworks is their poor detection performance**» — конфликт решают разъединением |
| **ICCAIS 2023**, [arXiv:2312.01650](https://arxiv.org/abs/2312.01650) | адаптивный порог из статистики детекций | «**we can only tune the confidence score threshold if we have prior knowledge of a video**» — сам балл не трогают |
| **UncertaintyTrack**, ICRA 2024, [arXiv:2402.12303](https://arxiv.org/abs/2402.12303) | фильтр по геометрической неопределённости вместо балла | прирост почти нулевой: «**detections with high localization uncertainty were well-filtered by ByteTrack's class confidence-based filter**» — полезно как чужой отрицательный результат |

---

## 5. Возражение, которое надо оговорить самим

**He Y.** Detector Confidence Signals Presence Rather Than Occlusion in
Cluttered Manipulation. [arXiv:2607.13361](https://arxiv.org/abs/2607.13361),
препринт 2026.

Утверждает **обратное** нашему П3: у open-vocabulary детекторов балл при
заслонении не падает — «the confidence stays nearly constant and uncorrelated
with visibility». Домен другой: робототехническая манипуляция, Grounding DINO,
OWLv2, SAM3.

Если П3 заявляет связь «заслонённость → занижение балла», это место лучше
оговорить в тексте самим, чем услышать на защите.

---

## 6. Чего не нашлось

- **Не найдено** статьи по 2D MOT, где в Limitations сказано бы «наш метод
  ограничен тем, что детектор занижает балл у заслонённых или молодых объектов,
  и мы этого не решаем».
- **Не найдено** рецензированного обзора MOT 2022–2026, где калибровка балла
  детектора вынесена в нумерованный перечень открытых вопросов. Свежий обзор
  Stanojević 2025 в Conclusion её не называет.
- **Не найдено** работы, которая прямо написала бы «решения на стороне детектора
  для порога рождения мы не рассматривали» — авторы просто подбирают порог.

## 7. Тексты, которые не открылись

- **Cao Y. et al.** Towards accurate dense pedestrian detection via
  occlusion-prediction aware label assignment and hierarchical-NMS. Pattern
  Recognition Letters, 2023.
  DOI: [10.1016/j.patrec.2023.08.019](https://doi.org/10.1016/j.patrec.2023.08.019)
  — ScienceDirect отдал 403. По поисковой выдаче формулировка крайне близка к
  нашей: «occluded pedestrians are often disregarded due to their **lower
  confidence scores, even when their IoU surpasses the detection threshold**».
  **Потенциально самый близкий источник. Достать отдельно и сверить по полному
  тексту; как цитату пока не использовать.**
- **Mandel T. et al.** Detection confidence driven multi-object tracking to
  recover reliable tracks from unreliable detections. Pattern Recognition
  135:109107, 2023.
  DOI: [10.1016/j.patcog.2022.109107](https://doi.org/10.1016/j.patcog.2022.109107)
  — платный доступ.

---

# Приложение. DRB: уверенность выросла, объектов стало меньше

Проверено по `experiments/results/` 11.09.2026 по запросу. Память не подвела:
у блока DRB действительно поднялась уверенность при падении полноты.

Ночной поднабор, baseline — три сида, поэтому есть разброс:

| | baseline (среднее ± std) | drb_fix | сдвиг |
|---|---|---|---|
| **precision** | 0.6550 ± 0.0074 | **0.6956** | **+0.0406 = 5.5σ** |
| **recall** | 0.3345 ± 0.0033 | **0.3246** | **−0.0099 = 3.0σ** |
| mAP50 | 0.3772 ± 0.0032 | 0.3728 | −0.0044 |
| mAP50-95 | 0.2317 ± 0.0007 | 0.2267 | −0.0050 = 7.1σ |

У варианта `drb_ad_seed123` то же направление, но слабее: precision 0.6722
(+2.3σ), recall 0.3337 — в шуме.

**Что это было.** DRB не улучшил детектор, а **сдвинул рабочую точку в сторону
точности**: стал увереннее в том, что оставляет, и оставлять стал меньше. Внешне
«уверенность выросла», по существу — объектов стало меньше, и mAP не вырос.

**Зачем это в диссертации.** Это собственный измеренный контрпример, на своём
стенде, с разбросом по трём сидам. Он показывает, что постановка «повысить
уверенность» **недостаточна**: её можно выполнить буквально и не получить
ничего. После него метод формулируется точнее — не «поднять уверенность», а
**поднять её избирательно, у истинных объектов определённой категории, не
трогая остальных**.

**Следствие для критериев Этапа 5.** Успехом считается рост доли найденных на
возрасте 0 **без** роста на возрасте 10+ и без падения общей точности. Просто
«полнота выросла» — половина результата, ровно как «уверенность выросла» у DRB.
