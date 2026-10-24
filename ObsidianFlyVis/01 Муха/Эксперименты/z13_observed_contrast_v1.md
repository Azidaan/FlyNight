---
aliases:
- 'З13: внутреннее считывание цели с фиксированными фоновыми образцами'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/fly-problem-P8
- метка/DaSiamRPN
- муха
исход: неопределённый
направление: FlyVis
начат: '2026-10-25'
---

# 🟡 З13: внутреннее считывание цели с фиксированными фоновыми образцами

↑ [[01 Муха/Темы/Прочее о мухе|Прочее о мухе]] · **Исход:** неопределённый · `z13_observed_contrast_v1`

Фиксированное сравнение с наблюдавшимися фоновыми пиками не прошло критерий применения.

## Вопрос
Does explicit comparison with specific previously observed background descriptors improve refusal beyond target-only cosine and the same node on receptors?

## Гипотеза
Сравнение с конкретным наблюдавшимся фоном может отличать похожий отклик от принадлежности цели лучше одной формы карты;это не установлено.

## Ожидалось (до прогона)
Cached68-dimensional candidate features and last-observed target mean. Background descriptors only from t=3 observed-frame response peaks with their entire lattice neighborhood outside the observed target mask. No query labels or future frames choose negatives. L2-unit descriptors; contrast=cos(query,target)-mean cos(query,observed backgrounds),coefficient1 fixed before results. If no background,exact target-only fallback. Same fixed maximum1% quota,nonpositive margins abstain; no threshold tuning. All8326rows/26reusedtrain perD/C/R1; original score decisions reproduce exactly. No model trained.

## Что сделано
2026-10-25

## Получено
runs/z13_observed_contrast_20261025/report.json → results: {"D": {"targetonly": {"selected": 84, "known": 65, "neutral": 65, "unknown": 19}, "contrast": {"selected": 84, "known": 63, "neutral": 62, "unknown": 21, "beneficial": 1}, "originalZ": {"selected": 84, "known": 64, "neutral": 64, "unknown": 20}}, "C": {"targetonly": {"selected": 84, "known": 72, "neutral": 72, "unknown": 12}, "contrast": {"selected": 84, "known": 67, "neutral": 65, "unknown": 17, "harmful": 2}, "originalZ": {"selected": 84, "known": 67, "neutral": 67, "unknown": 17}}, "R1": {"targetonly": {"selected": 84, "known": 46, "beneficial": 1, "neutral": 41, "unknown": 38, "harmful": 4}, "contrast": {"selected": 84, "known": 72, "neutral": 72, "unknown": 12}, "originalZ": {"selected": 84, "known": 48, "neutral": 48, "unknown": 36}}};paired: {"D_contrast-D_targetonly": {"mean_per_video": 0.0001622849724115547, "ci95": [0.0, 0.000486854917234664], "better": 1, "n": 26}, "D_contrast-D_originalZ": {"mean_per_video": 0.0001622849724115547, "ci95": [0.0, 0.000486854917234664], "better": 1, "n": 26}, "C_contrast-C_targetonly": {"mean_per_video": -0.00019398277933520226, "ci95": [-0.0004972312929361389, 0.0], "better": 0, "n": 26}, "C_contrast-C_originalZ": {"mean_per_video": -0.00019398277933520226, "ci95": [-0.0004972312929361389, 0.0], "better": 0, "n": 26}, "R1_contrast-R1_targetonly": {"mean_per_video": 0.0006864477735354028, "ci95": [-0.00032226492564035235, 0.0019964932462996875], "better": 4, "n": 26}, "R1_contrast-R1_originalZ": {"mean_per_video": 0.0, "ci95": [0.0, 0.0], "better": 0, "n": 26}, "D_contrast-R1_contrast": {"mean_per_video": 0.0001622849724115547, "ci95": [0.0, 0.000486854917234664], "better": 1, "n": 26}, "D_contrast-C_contrast": {"mean_per_video": 0.0003562677517467569, "ci95": [0.0, 0.0007972525485629817], "better": 3, "n": 26}}. Dнаблюдаемыхэпизодов2188,сфоновымиобразцами1149(prototype_statistics.D).

## Интерпретация
Новая конкретная реализация сравнения цели с отдельными наблюдавшимися фоновыми пиками дала1полезноеиз84выбранныхD,21неизвестныйисход. Толькоодновидеосположительнымизменением;нижняяграницаДИравна0. Это не подтверждённое улучшение отказа,непределпредставленияиливсейлинииFlyVis.

## Решение
Не переносить эту версию в HOTA. Сохранить прототип и отрицательный критерий. Простые фиксированные скалярные декодеры пока не решают отказ. Дальнейшая замена должна иметь иной механизм и собственный дешёвый критерий;не продолжать подбор косинусных весов на этих26видео.

## Нельзя заявлять
One-step cached baseline-state effects only,labels on sparse annotated times,reusedtrain26. Same top1 proposal means this version cannot solve absent useful candidates. NoGPU,newfeatures,weights,dev,20valtuning,HOTA,downloadsor75. Принцип цель−фон уже был вz13_candidate_foreground_v1:среднийконтекствнерасширеннойрамки,обучаемыйgain,722выхода. Здесьдругаяреализация:отдельныенаблюдаемыефоновыепики,L2косинус,фиксированныйкоэффициент1,замороженныепрототипы,сохранённыйцентр. Это не первыйвпроектеконтрастсфоном и не новаяэмпирическаяпричинапровалапрежнегоforeground.

## Прежние версии (до исправлений)
- **Суть**, 2026-10-25 (488b6868b4bae22d1c9dd720371edb098a78fed8) — Полный проверенный итог фиксированного прототипа.
  > Зарегистрирован один новый исполняемый узел считывания,без обучения.
- **Нельзя заявлять**, 2026-10-25 (488b6868b4bae22d1c9dd720371edb098a78fed8) — Полный проверенный итог фиксированного прототипа.
  > One-step cached baseline-state effects only,labels on sparse annotated times,reusedtrain26. Same top1 proposal means this version cannot solve absent useful candidates. NoGPU,newfeatures,weights,dev,20valtuning,HOTA,downloadsor75.

## Критерии приёмки (записаны до прогона)
- ✅ Все8326строк/26видео и исходный отбор84 точно воспроизведены;фон только наблюдаемый;источникинеизменны.  
  *report.json;sources_unchanged;all8326rows*
- ❌ All three paired-video95%lower bounds strictly>0 on net selected beneficial-minus-harmful actions over all known queries: Dcontrast-Dtargetonly;Dcontrast-DoriginalZ;Dcontrast-R1contrast. Any failure forbids implementation/HOTA promotion of this version. Report unknowns and actual coverage for every policy; no precision or safe-risk claim from known-only outcomes.  
  *runs/z13_observed_contrast_20261025/report.json → paired*

## Доказательства
- Новый механизм,отличие от прежних вариантов и критерии до исходов: `runs/z13_observed_contrast_20261025/protocol.json` · 
- Всеисходы,контрольбезмухи,парныеинтервалы: `runs/z13_observed_contrast_20261025/report.json` · 

## Код
- `scripts/z13_observed_contrast_20261025.py`
- `tracking/z13_observed_contrast.py`
- `tests/test_z13_observed_contrast.py`

