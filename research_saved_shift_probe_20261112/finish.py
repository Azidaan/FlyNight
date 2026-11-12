from common import *
a=read(OUT/'audit.json');r=read(OUT/'raw.json');assert a['passed']
obtained=('Все 93 исходные оценки воспроизведены; независимо проверены 279 перекрёстных оценок. '
 'Для своей232 FlyVis даёт −1,452064 при собственном сдвиге, −2,400672 при пространственном и −1,922243 при временном сдвиге контроля. '
 'Временной контроль обычного изображения на сдвиге FlyVis принимает232 с оценкой0,252832; пространственный отвергает с −0,337327. '
 'У832 и574 знак FlyVis не исправляется ни одной заменой. У1102 пространственный сдвиг превращает правильное принятие в отказ −0,432049. '
 'На26связях FlyVis: собственный сдвиг17своих/1чужая; пространственный12/1; временной8/1. Источники research_saved_shift_probe_20261112/raw.json, audit.json.')
interpret=('Замена только одного из трёх сохранённых сдвигов недостаточна для исправления232. Различие FlyVis и временного контроля остаётся при одинаковом сдвиге; '
 'оно находится далее в различии представлений и/или их обученных считывателей. Для пространственного контроля выбор сдвига существенен: на сдвиге FlyVis он тоже ошибается. '
 'Не доказаны физическая правильность сдвигов, утрата информации ядром или невозможность других соответствий. Внеисходные сдвиги могут менять распределение признаков относительно обучения.')
decision='Не заменять выбор сдвига готовым сдвигом другой ветви. Эта конкретная правка прекращена по отрицательному результату. Пороги, веса и полный трекер сохранить; физическая проверка третьего этапа остаётся незавершённой.'
cannot='Нет улучшения модели, независимого теста, HOTA или точной причины всех отказов. Проверены только три сохранённых сдвига. Последовательность видео, рекуррентное состояние и ядро не менялись.'
lines=['# Проверка одинакового сдвига сравнения','', '1. Получено','',obtained,'','2. Значение результата','',interpret,'','3. Решение','',decision,'',cannot,'','4. Полные пять случаев','']
for row in r['records']:
    if row['dataset']=='screen':lines += [f"Номер {row['old_id']}; представление {row['representation']}; источник сдвига {row['shift_selector']}; сдвиг {row['shift_pixels']}; оценка {row['score']:.9f}.",'']
lines+=['5. Проверки','',f"Независимый расчёт использовал другой интерполятор, формулу согласия через среднеквадратическое различие и скалярное суммирование. Максимальные расхождения: интерполяция {a['max_interpolation_error']:.3e}, признаки {a['max_feature_error']:.3e}, оценки {a['max_score_error']:.3e}. Общие карты, разметка, триангуляция и обученные слои явно сохранены. Исходные файлы неизменны.",'',
 'На 26 связях число принятых своих у FlyVis относительно исходных17: пространственный сдвиг −29,411765 %, временной −52,941176 %; принятых чужих по-прежнему1. Это описательная проверка на прежних данных, без вывода о независимом обобщении. Доля измеренного потолка не рассчитывается: потолок здесь не измерен.']
with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
path=ROOT/'experiments/project_history.json';raw=path.read_bytes();d=json.loads(raw);c=next(c for c in d['experiments'] if c['id']==ID);c.update(state='completed_negative',done='2026-11-12',obtained=obtained,interpretation=interpret,decision=decision,cannot_claim=cannot)
for item,value in zip(c['acceptance'],[True,True,a['case232_shift_only_explanation_supported']]):item.update(met=value,evidence='research_saved_shift_probe_20261112/audit.json')
c['evidence'] += [dict(kind='file',ref='research_saved_shift_probe_20261112/'+n) for n in ['raw.json','audit.json','РЕЗУЛЬТАТ.md']]
assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/system_problems.json';raw=path.read_bytes();d=json.loads(raw);problem=next(x for x in d['problems'] if x['id']=='С1')
problem['cases'].append(dict(date='2026-11-12',component='Считыватель полного модуля FlyVis',fact=obtained,hypothesis='Причина различия представлений и обученных функций ещё не разделена; замена сохранённого сдвига не является достаточным исправлением232.',discriminating_check='Следующий уровень — разделить временное состояние, вход и функцию считывания; новых причин по этому опыту не объявлять.',limitation=interpret,evidence=['research_saved_shift_probe_20261112/audit.json'],source_sha256=sha(OUT/'audit.json')))
problem['cards'].append(ID);assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Получено (`'+ID+'`): '+obtained+' Вывод. '+interpret+' Решение. '+decision)
print('RECORDED_COMPLETED_NEGATIVE')
