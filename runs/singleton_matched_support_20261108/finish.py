"""Record the measured diagnostic without changing earlier conclusions."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf8'))
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
scores=read(OUT/'scores.json')
protocol=read(OUT/'protocol.json')
foreign,own=scores['pairs']
previous=read(ROOT/'runs/singleton_two_map_scores_20261108/scores.json')['pairs'][0]
assert {k:v for k,v in foreign.items() if k!='reused_from'}==previous
summary='На равных трёх ячейках своя пара 3→273 дала FlyVis 0,794335→0,826478, яркость −0,999726→0,997012, градиенты 0,553791→0,553791. Чужая 390→448 из готового расчёта дала FlyVis 0,349265→0,349265, яркость −0,773286→0,999658, градиенты −0,563739→0,920580. Источник: runs/singleton_matched_support_20261108/scores.json.'
interpretation='В этом сравнении FlyVis ранжирует свою пару выше чужой до и после поиска. После поиска оба контроля изображения ранжируют чужую выше своей. Разное число ячеек не объясняет весь прежний эффект. Причина различия представлений и безопасность решения трекера по двум событиям не установлены.'
limitation='Одна своя и одна чужая пара из разных видео. Вырезки 64 и 522 пикселя дают разный физический размер ячейки и радиус поиска. C содержит восемь нормированных каналов T4/T5, яркость один, градиенты два. Контроль изображения — яркость и градиенты уже подготовленного входа глаза. Прошлая личность своей пары подтверждена разметкой кадра 61, последнее наблюдение шаблона на кадре 71 разметки не имеет. Будущее используется только для метки проверки. Порог и качество траекторий не оценивались.'
decision='Первая проблема открыта. Два примера дают основание сохранить пространственный сигнал как кандидата для проверки отказа, но не вводить по ним порог или передачу ID.'
report='# Сравнение карт при одинаковых трёх ячейках\n\n'+summary+'\n\n'+interpretation+'\n\n'
report+='| Пара | Сигнал | По прогнозу | После поиска | Изменение |\n| --- | --- | --- | --- | --- |\n'
for pair in scores['pairs']:
    for name,variant in pair['variants'].items():
        label={'FlyVis':'T4/T5 FlyVis','brightness':'Яркость','gradient':'Градиенты','permuted':'Перестановка T4/T5'}[name]
        report+=f'| {pair["kind"]} {pair["old"]}→{pair["new"]} | {label} | {variant["fixed"]:.6f} | {variant["local"]:.6f} | {variant["gain"]:+.6f} |\n'
report+='\n'+limitation+'\n\n'
report+='Из шести прямых однокандидатных событий с постоянным масштабом и минимум тремя общими ячейками в сохранённом реестре только одно чужое. Единственная своя пара с равным числом ячеек выбрана до расчёта, без отбора по оценке. Восстановлены пять отсутствовавших входов глаза и две карты. Общий префикс FlyVis пройден один раз. Проверочное сохранённое изображение глаза воспроизведено точно. Параметры модели до и после совпадают. Чужие оценки прочитаны из готового файла без пересчёта. Независимые способы интерполяции и скалярного произведения новых оценок согласуются в пределах 10⁻⁹.\n\n'
report+='При трёх точках центрированная яркость имеет только две независимые координаты. Перебор сдвигов позволяет найти сходное направление этого короткого вектора у разных объектов. Это математически возможное объяснение высокой корреляции, причинность на этих событиях не доказана. Сильнейшая альтернатива — преимущество даёт набор восьми каналов и особенности сцены. Проверка, различающая объяснения, должна сохранять поддержку и геометрию, сравнивая временно обработанный вход глаза с T4/T5 при одинаковом числе каналов.\n\n'
report+='Одна фиксированная перестановка дала своей паре 0,550214→0,754168, чужой −0,529622→0,512881. Это контроль примера, а не распределение случайных совпадений.\n\n'+decision+'\n\nОтносительный процент качества трекера и доля потолка не определены, поскольку измерено сходство карт, которое может менять знак.\n'
(OUT/'РЕЗУЛЬТАТ.md').write_text(report,encoding='utf8')
path=ROOT/'experiments/project_history.json'
catalog=read(path)
card=next(c for c in catalog['experiments'] if c['id']==OUT.name)
assert card['state']=='running'
card.update(state='inconclusive',done='2026-11-08',obtained=summary,interpretation=interpretation,decision=decision,cannot_claim=limitation)
card['acceptance'][0].update(met=True,evidence='runs/'+OUT.name+'/scores.json')
card['evidence'] += [dict(kind='file',ref='runs/'+OUT.name+'/'+name,claim='Первичный результат и проверка') for name in ['scores.json','maps_complete.json','audit.json','РЕЗУЛЬТАТ.md']]
card['metrics']=[dict(key='map_centered_similarity',value=v['local'],baseline=v['fixed'],subset=f'{p["kind"]} / {name}',
    source='runs/'+OUT.name+'/scores.json',locator=f'pairs.{i}.variants.{name}.local') for i,p in enumerate(scores['pairs']) for name,v in p['variants'].items()]
write(path,catalog)
path=ROOT/'experiments/obsidian_experiment_topics.json'
topics=read(path)
topics['topics'][OUT.name]=topics['topics']['singleton_two_map_scores_20261108']
write(path,topics)
now=datetime.now(timezone(timedelta(hours=3)))
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
plan=path.read_text(encoding='utf8')
entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Сравнение своих и чужих машин при равной поддержке.\n\n*Постановка.* Поддержка равна 3 ячейкам, 19 сдвигов, прежняя формула сходства. Карточка `{OUT.name}`.\n\n*Получено (`{OUT.name}`).* {summary} Восстановлены только 5 отсутствовавших входов и 2 карты своей пары, точное совпадение проверочного входа, веса неизменны. Чужая пара повторно не вычислялась. Интерполяция и оценки своей пары проверены независимо с расхождением до 10⁻⁹.\n\n*Вывод.* {interpretation} {limitation}\n\n*Решение.* {decision}\n\n'
path.write_text(plan.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
path=ROOT/'experiments/system_problems.json'
registry=read(path)
problem=next(p for p in registry['problems'] if p['id']=='С2')
problem['cases'].append(dict(date='2026-11-08',component='Пространственная проверка продолжения объекта',fact=summary,
    hypothesis='Малое число ячеек и максимум по сдвигам позволяют получить случайно высокое сходство изображения.',
    discriminating_check='Согласовать поддержку, геометрию и число каналов у пространственного сигнала и контроля входа глаза.',
    limitation=limitation,evidence=['runs/'+OUT.name+'/scores.json']))
problem['evidence'].append(dict(source='runs/'+OUT.name+'/scores.json',sha256=sha(OUT/'scores.json'),claim='Сравнение при равных трёх ячейках'))
write(path,registry)
write(OUT/'audit.json',dict(passed=True,foreign_scores_reused_verbatim=True,equal_support_cells=3,
    new_score_independent_check_tolerance=1e-9,source_checks_before_after=True,
    retinal=read(OUT/'maps_complete.json'),limitations=limitation,
    sources={p:sha(ROOT/p) for p in protocol['sources']},
    outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='audit.json'}))
