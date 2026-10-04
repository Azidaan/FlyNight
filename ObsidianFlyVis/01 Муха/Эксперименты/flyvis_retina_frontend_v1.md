---
aliases:
- 'Н9/М1: фронт-энд «фоторецептор + ламина» перед глазом FlyVis — восстанавливается ли сигнал мухи на тёмных видео'
tags:
- эксперимент
- исход/inconclusive
- направление/flyvis
- метка/registry-H9
- метка/modernization-M1
- метка/foveation
- метка/retina
- метка/night
- метка/learned-head
- муха
исход: неопределённый
направление: FlyVis
начат: '2026-09-18'
завершён: '2026-09-30'
---

# 🟡 Н9/М1: фронт-энд «фоторецептор + ламина» перед глазом FlyVis — восстанавливается ли сигнал мухи на тёмных видео

↑ [[01 Муха/Темы/Глаз и сетчатка-ламина|Глаз и сетчатка-ламина]] · **Исход:** неопределённый · `flyvis_retina_frontend_v1`

Муха получала gray/255 без адаптации, тогда как обучена при яркости ~0,45 и контрасте exp(N(0, 0,2)); на 27 тёмных роликах (motion_reliability_v1, стратификация по яркости) муха без DIS даёт −0,003 AP к геометрии против +0,013 на светлых; на тёмном val-видео b1d22449 ρ(|δ|, ошибка) = −0,036, голова geometry+fly 0,475 < geometry 0,502 (LOVO). Перед глазом ставится tracking/retina.py: фоторецептор R1–R6 (Нака–Раштон I/(I+I_adapt+dark_floor) с адаптацией I_adapt по τ = 0,3 с и σ = 24 px, усиление 2, темновой порог 0,05) и ламина L1/L2 (центр-окружение σ = 12 px, деление на локальный размах с порогом 0,1); контроль norm — глобальная нормализация кадра без биологии. Сама сеть, декодер и параметры мухи не меняются. Ступень A: пересчёт фовеального потока на val-семёрке для трёх режимов (build_foveated_readout_cache.py --decode-flow --retina <mode>), оценка scripts/evaluate_retina_frontend.py: ρ по видео, ошибка после поправки g·δ, головы «исключить одно видео». Ступень B (по итогам A, по разрешению — >1 ч GPU): пересчёт на тёмных роликах кэша ступени 1 и повтор лестницы Н2 на тёмной/светлой подвыборках.

## Вопрос
Возвращает ли адаптивный вход (достроенная сетчатка) мухе сигнал надёжности/движения на тёмных видео, где сейчас его нет?

## Гипотеза
Причина пустоты мухи ночью — вход вне обучающего распределения, а не коннектом; после адаптации входа ρ и вклад головы на тёмном видео станут положительными, на светлых не ухудшатся.

## Ожидалось (до прогона)
На b1d22449: ρ(|δ|, ошибка) > +0,1 (сейчас −0,036); прибавка головы geometry+fly над geometry больше, чем без сетчатки (сейчас −0,027) и выше shuffled. Средний LOVO AP geometry+fly по 7 видео не ниже, чем без сетчатки, минус 0,005. Режим norm — контроль: если он даёт то же, что фоторецептор, биологическая форма не нужна, достаточно нормализации.

## Что сделано
Карточка записана до расчёта (18.09.2026, 20:5x MSK). Модуль tracking/retina.py (8 тестов), --retina в сборщике фовеи, evaluate_retina_frontend.py. Прогон трёх режимов на val7 запущен 18.09 20:47 (PID 25068, лог runs/fovea_retina_val7.log). Прогон прерван 18.09 22:0x (отключение электричества), возобновлён 30.09 01:17; досчёт ламины и контроля norm 30.09 01:30–02:40 параллельным сторожем scripts/run_fovea_jobs.py (исправлены: утечка памяти разгона глаза 9 ГБ → 0,5 ГБ, порции глаз, потоки CPU на процесс); b1d22449 norm взят из одиночного прогона с теми же настройками (под сторожем трижды «молчал», одиночно — 53 с).

## Получено
ρ(|поток|, ошибка) среднее по 7 видео / тёмное b1d22449: none 0.068 / -0.036; norm 0.061 / -0.116; photoreceptor 0.101 / -0.011; lamina 0.154 / 0.044. LOVO AP geometry+fly (без DIS), среднее по 7: none 0.3931, norm 0.4046, photoreceptor 0.4127, lamina 0.4144 при geometry 0.3965. geometry+DIS+fly: none 0.4549, norm 0.4557, photoreceptor 0.4592, lamina 0.4618 при geometry+DIS 0.4577. Тёмное видео, прибавка geometry+fly над geometry: none -0.0266, lamina +0.0035.

## Интерпретация
Адаптивный вход возвращает мухе сигнал надёжности на val-семёрке, и важна биологическая форма: контроль norm (глобальная нормализация кадра без адаптации) даёт ρ как без сетчатки и хуже на тёмном видео, фоторецептор (локальная временная адаптация Вебера) — рост ρ в 1,5 раза, ламина (фоторецептор + центр-окружение) — в 2,3 раза и лучший AP с DIS и без. Муха с ламиной впервые не хуже и чуть лучше DIS-головы (+0,004 AP). По тёмным сценам вывод не сделан: одно тёмное видео (327 наблюдений), критерий ρ > 0,1 не выполнен ни одним режимом; контроль shuffled зависит от сида перестановки.

## Решение
Ступень A: среднее по 7 видео положительно, заранее записанный главный критерий по тёмному видео (ρ > 0,1) не выполнен → inconclusive. Выбран режим lamina для ступени B (200 обучающих роликов, 27 тёмных dev/test), которая решает вопрос о тёмных сценах; ступень B — быстрая ламина (гладкие поля на кадре ×1/4, отличие в среднем 0,16/255), val7 пересчитывается той же версией для согласованности.

## Ограничения
Ступень A — 7 видео, одно тёмное; LOVO-головы малы; параметры сетчатки выбраны заранее по физиологии, не подбирались (один набор на режим). Сжатие видео даёт блочный шум, который адаптация усиливает — гасится dark_floor и усреднением фасеток.

## Нельзя заявлять
Перенос на тёмные train-видео и пользу для HOTA до ступени B и встраивания.

## Условия
- **baseline:** режим none (runs/flyvis_fovea_flow, карточка [[01 Муха/Эксперименты/flyvis_fovea_motion_v1|flyvis_fovea_motion_v1]])
- **variant:** photoreceptor; lamina; norm (контроль)
- **dataset:** val-семёрка (runs/video_rate детекции и DIS; фовеальный поток по режимам в runs/flyvis_fovea_flow_retina_{norm,photo,lamina})
- **split:** LOVO по 7 видео; порог события τ_e = 75-й перцентиль ошибки на train-четвёрке режима none

## Критерии приёмки (записаны до прогона)
- ❌ ρ(|δ|, ошибка) на тёмном видео b1d22449 > +0,1 (без сетчатки −0,036)  
  *runs/retina_frontend/report_val7.json → modes.lamina.rho_dark = 0.0439 (лучший режим)*
- ✅ прибавка головы geometry+fly над geometry на тёмном видео больше, чем в режиме none, и выше shuffled  
  *runs/retina_frontend/report_val7.json → acceptance.lamina.dark_improves = true, fly_beats_shuffled_dark = true (одно видео, 327 наблюдений)*
- ✅ средний LOVO AP geometry+fly по 7 видео не ниже режима none − 0,005  
  *runs/retina_frontend/report_val7.json → acceptance.lamina.mean_not_worse = true (AP 0.4144 против 0.3931)*

## Метрики
| метрика | значение | baseline | подмножество | источник |
|---|---|---|---|---|
| [rho↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%CF%81) | 0.15378420782897242 | 0.06763739475648298 | val7, среднее по видео: lamina / none | `runs/retina_frontend/report_val7.json` · modes.lamina.rho_mean |
| [rho↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%CF%81) | 0.10077352153209283 | 0.06763739475648298 | val7, среднее по видео: photoreceptor / none | `runs/retina_frontend/report_val7.json` · modes.photoreceptor.rho_mean |
| [rho↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2F%CF%81) | 0.060620619148203804 | 0.06763739475648298 | val7, среднее по видео: norm (контроль) / none | `runs/retina_frontend/report_val7.json` · modes.norm.rho_mean |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.4143718293501378 | 0.3931429092099007 | val7 LOVO, geometry+fly: lamina / none | `runs/retina_frontend/report_val7.json` · modes.lamina.heads.geometry_fly.mean_ap |
| [AP↗](obsidian://open?vault=ObsidianFlyVis&file=04%20%D0%A2%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D1%8B%20%D0%B8%20%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%2FAP) | 0.46178473696468164 | 0.4577276846668016 | val7 LOVO, geometry+DIS+fly (lamina) / geometry+DIS | `runs/retina_frontend/report_val7.json` · modes.lamina.heads.geometry_dis_fly.mean_ap |

## Доказательства
- вход вне распределения при слабом освещении; решение — фронт-энд: 1.7. Реестр, Н9
- диагноз по стратификации motion_reliability_v1 и постановка М1: `dissertation/Исследование_модернизация_мухи.md` · 1.1, 3 (М1)
- ступень A на val7: `runs/retina_frontend/report_val7.json` · modes, acceptance

## Код
- `scripts/build_foveated_readout_cache.py`
- `scripts/evaluate_retina_frontend.py`
- `scripts/run_fovea_jobs.py`
- `tests/test_retina.py`
- `tracking/foveated_eyes.py`
- `tracking/retina.py`

## Данные
- `runs/retina_frontend`
- `runs/flyvis_fovea_flow_retina_photo`
- `runs/flyvis_fovea_flow_retina_lamina`
- `runs/flyvis_fovea_flow_retina_norm`

