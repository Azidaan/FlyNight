import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.now(timezone(timedelta(hours=3)))
scores=read(OUT/'scores.json')
summary='На своей паре 3→273 четыре центральных EMA, повторённые до восьми каналов, дали 0,325505→0,908726; EMA и средние соседей дали 0,280763→0,904837. На чужой 390→448 соответственно −0,699152→0,737942 и −0,540123→0,707595. Готовые T4/T5 дали 0,794335→0,826478 своей и 0,349265→0,349265 чужой. Источник: runs/singleton_neighbor_input_20261108/scores.json.'
interpretation='Оба новых простых контроля ранжируют свою пару выше чужой до и после поиска. После поиска соседи снижают оценку своей на 0,003889 и чужой на 0,030347. Исправление порядка относительно прежних восьми EMA+остаток нельзя приписать соседям, поскольку тот же порядок уже получен четырьмя центральными EMA. Существенны состав и относительный вес каналов. Причина различия с FlyVis и польза полному трекеру не установлены.'
limitation='Два события по три опорные ячейки. Нормировка прежняя обучающая, у центрального канала и его соседнего среднего общий масштаб. Соседи берутся из всей сетки, в том числе вне рамки и Q. Постоянные времени прежние, шаги по сохранённым наблюдениям. Разные сцены и размер вырезки 64/261 пиксель. У своей пары последняя разметка прошлого трека на кадре 61, шаблон на неразмеченном 71, текущая рамка размечена на 73. Баллы разных представлений не откалиброваны как вероятность правильного ID.'
decision='Первая проблема открыта. Завершить эту серию диагностик малой поддержки. Сформулировать запрос Gemini о минимальном правиле пространственного соответствия и отказа с первичными источниками. Порог и новый полный прогон по двум событиям не назначать.'
report='# Обычная обработка соседних ячеек\n\n'+summary+'\n\n'
report+='Новый пространственно-временной вариант содержит четыре готовых EMA входа и четыре средних тех же EMA по шести соседним ячейкам. Контроль без соседей повторяет четыре центральных EMA дважды. Длительности 1,5; 3,9032; 12,2976; 32 шага сохранённой истории. Использованы прежние HexGeometry и neighbor_aug. Нормировка центрального и соседнего канала одинаковая. Усреднение линейно, поэтому усреднение уже нормированных карт эквивалентно нормировке усреднённых значений с теми же средним и масштабом.\n\n'
report+='| Пара | Представление | По прогнозу | После поиска | Изменение |\n| --- | --- | --- | --- | --- |\n'
for pair in scores['pairs']:
    for key,title in [('center8','Четыре EMA, повторённые дважды'),('neighbor8','Четыре EMA и четыре средних соседей'),('input8_temporal','Прежние четыре EMA и четыре отклонения'),('FlyVis','Готовые T4/T5')]:
        v=pair['variants'][key]
        kind='Своя' if pair['kind']=='own' else 'Чужая'
        report+=f'| {kind} {pair["old"]}→{pair["new"]} | {title} | {v["fixed"]:.6f} | {v["local"]:.6f} | {v["gain"]:+.6f} |\n'
report+='\n'+interpretation+'\n\n'+limitation+'\n\n'
report+='Контроль без соседей отделяет эффект усреднения от замены временных каналов и их весов. Пространственное среднее немного снизило обе максимальные оценки, сильнее на чужой паре. Один правильный порядок двух событий не измеряет частоту отказов, безопасность восстановления, HOTA или причинный вклад нейродинамики. Сильнейшая альтернатива объяснению через нейродинамику — различие состава признаков, их нормировки и обычной пространственной фильтрации. Новая проверка не даёт основания выбирать порог по этим событиям.\n\n'
report+='Проверка среднего соседей независимым суммированием дала точное совпадение. Интерполяция и скалярное произведение независимо сверены до 10⁻⁹. Готовые оценки и временные карты повторно не вычислялись.\n\n'+decision+'\n\nОтносительный процент качества трекера и доля потолка не определены, поскольку измерены баллы сходства двух пар.\n'
(OUT/'РЕЗУЛЬТАТ.md').write_text(report,encoding='utf8')
path=ROOT/'experiments/project_history.json'
catalog=read(path)
card=next(c for c in catalog['experiments'] if c['id']==OUT.name)
assert card['state']=='running'
card.update(state='inconclusive',done='2026-11-08',obtained=summary,interpretation=interpretation,decision=decision,cannot_claim=limitation)
card['acceptance'][0].update(met=True,evidence='runs/'+OUT.name+'/audit.json')
card['evidence'] += [dict(kind='file',ref='runs/'+OUT.name+'/'+name,claim='Первичный результат и проверка') for name in ['scores.json','audit.json','РЕЗУЛЬТАТ.md']]
card['metrics']=[dict(key='map_centered_similarity',value=v['local'],baseline=v['fixed'],subset=p['kind']+' / '+name,
    source='runs/'+OUT.name+'/scores.json',locator=f'pairs.{i}.variants.{name}.local') for i,p in enumerate(scores['pairs']) for name,v in p['variants'].items() if name in ['center8','neighbor8','FlyVis']]
write(path,catalog)
path=ROOT/'experiments/obsidian_experiment_topics.json'
topics=read(path);topics['topics'][OUT.name]=topics['topics']['singleton_eight_input_20261108'];write(path,topics)
path=ROOT/'experiments/system_problems.json'
registry=read(path);problem=next(p for p in registry['problems'] if p['id']=='С2')
problem['cases'].append(dict(date='2026-11-08',component='Простое среднее соседних ячеек',fact=summary,
    hypothesis='Состав и относительный вес временных каналов влияют на случайный максимум малой пространственной поддержки.',
    discriminating_check='Выбрать минимальное правило соответствия и отказа на основе первичных работ, с согласованным контролем входа глаза.',
    limitation=limitation,evidence=['runs/'+OUT.name+'/scores.json']))
problem['evidence'].append(dict(source='runs/'+OUT.name+'/scores.json',sha256=sha(OUT/'scores.json'),claim='Согласованное сравнение с обычным средним соседей'))
write(path,registry)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
plan=path.read_text(encoding='utf8')
entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Пространственно-временной контроль соседних ячеек.\n\n*Постановка.* Те же пары и 3 ячейки, прежние 19 сдвигов. Карточка `{OUT.name}`.\n\n*Получено (`{OUT.name}`).* {summary} Средние соседей независимо совпали точно, новые оценки сверены до 10⁻⁹ (`runs/{OUT.name}/audit.json`).\n\n*Вывод.* {interpretation} {limitation}\n\n*Решение.* {decision}\n\n'
path.write_text(plan.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
