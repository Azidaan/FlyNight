"""Record measured bounded result, preserving prior text and foreign drafts."""
from common import *
start()
import numpy as np
import re

verify();s=read(OUT/'summary.json');a=read(OUT/'audit.json');assert s['passed'] and a['passed']
assert s['preliminary_C34_gate'] is False
p=read(OUT/'predictions.json'); m=s['metrics']; identity='flyvis_two_integration_state_readout_20261111'
rows=p['records'];shifts={n:sum(r['variants'][n]['winner']!=r['variants'][n]['old_winner'] for r in rows) for n in m}
maps=read(OUT/'maps_complete.json');rel=np.array([r['first_vs_last_relative_l2'] for r in maps['records']])
report='# Два состояния FlyVis: итог фиксированного сравнения\n\n'
report+='## 1. Короткий итог\n\nРавное совместное считывание не изменило ни одного решения FlyVis: приняты17своих и2чужие пары, как у последнего состояния. Этот вариант не прошёл заранее записанный допуск. Ядро и веса не менялись; новое обучение не проводилось. Источник всех итоговых чисел: `summary.json`, ключи `metrics`, `decision_changes`, `paired_video_intervals`.\n\n'
report+='## 2. Сравнение на одинаковых парах\n\n26ранее изученных разработческих пар:20своих/6чужих,20видео. Порог строго0, ячейки и19сдвигов прежние; головы с исключением целого видео взяты готовыми.\n\n'
report+='| Представление | Свои, последнее → совместное | Чужие, последнее → совместное | Сбалансированная точность, последнее → совместное | Изменение к своей базе |\n|---|---:|---:|---:|---:|\n'
names={'C34':'FlyVis,34канала','R134_temporal':'Обычные временные признаки','R134_spatial':'Обычные пространственные признаки'}
for n, v in m.items():
    x,y=v['last'],v['joint'];delta=y['balanced_accuracy']-x['balanced_accuracy'];relative=100*delta/x['balanced_accuracy']
    report+=f'| {names[n]} | {x["own_accept"]} → {y["own_accept"]} | {x["foreign_accept"]} → {y["foreign_accept"]} | {100*x["balanced_accuracy"]:.2f}% → {100*y["balanced_accuracy"]:.2f}% | {100*delta:+.2f}п.п.; {relative:+.2f}% |\n'
report+='\nСбалансированная точность — среднее доли принятых своих и доли отвергнутых чужих; число чужих желательно уменьшать. Изменения решений перечислены по реальным записям ниже.\n\n'
for c in s['decision_changes']:
    report+=f'Видео `{c["video"]}`, кадр{c["frame"]}, старыйID{c["old_id"]}, рамка{c["raw_id"]}: {names[c["signal"]]}, подтверждённая своя; оценка {c["old"]:.6f} → {c["joint"]:.6f}.\n\n'
report+='## 3. Что изменилось в картах и оценках\n\n'
report+=f'Новые общие сдвиги отличаются от прежних у C34 в{shifts["C34"]}из26пар, временного контроля в{shifts["R134_temporal"]}, пространственного — в{shifts["R134_spatial"]}. Это изменение позиции максимума, не число исправленных ID. Медианное относительное различие первой и последней карт C34 по норме составляет{float(np.median(rel)):.6f}; источник `maps_complete.json/records/*/first_vs_last_relative_l2`. Карты действительно различаются.\n\n'
report+='Порядок оценок C34 немного изменился: AUC (доля правильно упорядоченных сочетаний своей и чужой пары)0,8500→0,8667, то есть улучшились2из120сочетаний с учётом возможных равенств; источник `summary.json/metrics/C34/*/auc`. Это не перенос готового порога: ни одна оценка не сменила сторону нуля. Пары неоднородны и уже изучались; показатель не является независимой проверкой узнавания.\n\n'
report+='95%парный интервал по видео для добавки совместного C34 к последнему — [0;0]п.п. Преимущество над совместным временным контролем: точка+14,17п.п., интервал[−7,50;+36,25]; над пространственным: +5,83п.п., интервал[−10,00;+25,00]. Оба интервала включают ноль; дополнительная польза FlyVis не подтверждена. Источник `summary.json/paired_video_intervals`.\n\n'
report+='## 4. Корректность и воспроизводимость\n\nМатематика проверена до карт: `math_checks.json`. Восстановлены52промежуточные карты из прежних причинных последовательностей. Последние C34 совпали с архивом точно, ошибка0; обычный последний вход также совпал точно. Все готовые78поисков последнего состояния и прежние обученные головы переиспользованы. Дублирование последнего вектора сохраняет выход головы с погрешностью меньше1e−12; источник `predictions.json/frozen_head_identity_max_error`.\n\n'
report+='Независимый оценщик заново проверил1482вектора сдвигов по сохранённым картам; максимальное расхождение1,11e−15 при прежнем допуске1e−10. Источник `audit.json`. Сначала он обнаружил усреднение соседей в float32 вместо float64 и затем различие округления формулы точности на1,11e−16. Исправления касаются только оценщика, числа назначений совпадали; замороженный исходник, ошибки и журналы сохранены. Источники `audit_recovery2_diagnosis.json`, `audit_recovery3_diagnosis.json`.\n\n'
report+='Первый запуск остановился при импорте библиотеки: локальный `queue.py` затенял стандартный модуль. До вызова сети. Восстановление выполнено отдельной обёрткой `launch_stage.py`, паспорт `recovery1.json`; первоначальные ошибки не затирались. Повторных научных расчётов ядра после получения52карт не было.\n\n'
report+='## 5. Граница вывода и решение\n\nПроверено равное объединение34поканальных свидетельств двух шагов с готовой головой. Для FlyVis оно не дало дополнительных правильных решений. Сильнейшее объяснение наблюдаемой прибавки простых контролей — польза обычного временного объединения представлений; вклад ядра в эту прибавку отсутствует по постановке контроля.\n\nНельзя заключить, что промежуточные состояния бесполезны вообще: обучаемое объединение отдельных состояний и иные считыватели здесь не проверялись. Это два шага численной динамики, не разные глубины графа. Судьба публичного ID и HOTA не измерялись, независимого финального теста нет. К полному трекеру этот вариант не продвигается; конкуренция слотов не запускалась.\n\nЛитературное основание и отличие от прежних опытов: `ОСНОВАНИЕ.md`. Сводка к своей базе: C34 — 0%; временной контроль — +8,82%; пространственный — +3,70% по сбалансированной точности. Доля измеренного потолка не вычисляется: согласованный потолок этой проверки не измерен.\n'
report=re.sub(r'(?<=\d)\.(?=\d)', ',', report)
report=re.sub(r'(?<=[а-яА-ЯёЁ])(?=\d)|(?<=\d)(?=[а-яА-ЯёЁ])', ' ', report)
(OUT/'РЕЗУЛЬТАТ.md').write_bytes(report.encode())
path=ROOT/'experiments/project_history.json';catalog=read(path);card=next(c for c in catalog['experiments'] if c['id']==identity);assert card['state']=='running'
card.update(state='completed_negative',done='2026-11-11',
    obtained='На26парах(20своих/6чужих) C34:17своих/2чужие как у последнего состояния; временной контроль16/4→18/4, пространственный17/3→18/3. Все52последнихсостояния совпали точно с архивом, независимая корреляционная сверка1,11e-15. Источник research_multistate_readout_20261111/summary.json:metrics,decision_changes,final_state_parity; audit.json.',
    interpretation='Равное смешивание с готовой головой не изменило решений FlyVis; критерий строгого улучшения не выполнен. Частная прибавка обычных контролей не является прибавкой ядра. Интервалы сравнений с обоими обычными совместными представлениями включают ноль.',
    decision='Не продвигать эту фиксированную версию к полному трекеру. Сохранить промежуточные карты для будущих отдельных протоколов, без автоматического обучения и без запуска слотов.')
criteria=card['acceptance'];passed=[True,True,True,False,False]
for item,met,proof in zip(criteria,passed,['math_checks.json','maps_complete.json','audit.json','summary.json','summary.json']):item.update(met=met,evidence=OUT.name+'/'+proof)
card['evidence'].extend([dict(kind='file',ref=OUT.name+'/summary.json',claim='Сравнение решений и интервалы без подбора порога'),dict(kind='file',ref=OUT.name+'/audit.json',claim='Независимая сверка сохранённых карт'),dict(kind='file',ref=OUT.name+'/РЕЗУЛЬТАТ.md',claim='Итог, ограничения и технические восстановления')])
path.write_bytes((json.dumps(catalog,ensure_ascii=False,indent=2)+'\n').encode())
path=ROOT/'experiments/system_problems.json';before=path.read_bytes();save=OUT/'before/experiments/system_problems.json';save.parent.mkdir(parents=True,exist_ok=True);save.write_bytes(before)
system=json.loads(before);problem=next(x for x in system['problems'] if x['id']=='С1')
problem['cases'].append(dict(date='2026-11-11',component='Считыватель полного модуля FlyVis; ядро неизменно',
    fact=card['obtained'],hypothesis='Два близких состояния могут менять ранги сходства, но равное смешивание с готовой головой недостаточно для исправления решений.',
    discriminating_check='Эта версия не прошла предварительный допуск. Не заменять её незарегистрированным подбором весов или порога; иное обучаемое объединение требует отдельной постановки и тех же обычных контролей.',
    limitation=card['cannot_claim'],evidence=[OUT.name+'/summary.json',OUT.name+'/audit.json'],source_sha256=sha(OUT/'summary.json')))
problem['updated']='2026-11-11'
if identity not in problem['cards']:problem['cards'].append(identity)
path.write_bytes((json.dumps(system,ensure_ascii=False,indent=2)+'\n').encode())
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();marker='## Журнал изменений'.encode();index=before.index(marker)+len(marker)
entry='\n\n**'+datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')+'** — Два состояния FlyVis: решения не улучшились, контроль вернул свои пары\n\n'
entry+='*Постановка (`'+identity+'`).* Фиксированное равное объединение двух внутренних состояний и готовые считыватели; протокол,26пар, порог0 и ядро неизменны.\n\n'
entry+='*Получено.* C34:17своих/2чужие→17/2; временной контроль16/4→18/4; пространственный17/3→18/3. Сбалансированная точность C34:0,758333→0,758333, то есть0%; у контролей+8,82%и+3,70%к своим базам.52промежуточные карты восстановлены, последнее состояние точно совпало с архивом; независимая ошибка1,11e−15. Источники `'+OUT.name+'/summary.json`, `audit.json`, `maps_complete.json`; доля потолка неизвестна.\n\n'
entry+='*Техническое восстановление.* Ошибка импорта из-за локального queue.py устранена изоляцией пути, до вычисления карт; затем в отдельной версии оценщика исправлены точность усреднения соседей и порядок арифметики метрики. Научные выходы и строгий допуск не менялись. Первоначальные ошибки и контрольные суммы сохранены: `'+OUT.name+'/recovery1.json`, `audit_recovery3_diagnosis.json`.\n\n'
entry+='*Вывод.* Предварительный критерий C34 не выполнен; все решения FlyVis прежние. Превосходство над обычными совместными контролями не подтверждено интервалами. Это отрицательный итог равного смешивания с готовой головой, не всех способов чтения нескольких состояний и не ядра.\n\n'
entry+='*Решение.* Не запускать полный трекер этой версии; карты сохранить. Итог включён в случай С1. Гипотезы, их статусы и конкуренция слотов не изменялись.\n\n'
path.write_bytes(before[:index]+entry.encode()+before[index:].lstrip(b'\r\n'));(OUT/'result_entry.md').write_bytes(entry.encode())
write('recording.json',dict(passed=True,card=identity,scientific_state='completed_negative',source_sha256=sha(OUT/'summary.json'),shift_changes=shifts,first_last_median_relative_l2=float(np.median(rel))))
print('RESULT_RECORDED')
