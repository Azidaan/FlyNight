import hashlib
import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.now(timezone(timedelta(hours=3)))
scores=read(OUT/'scores.json')
summary='На своей паре 3→273 восемь простых признаков входа дали 0,004349→0,878890, на чужой 390→448 −0,727158→0,953052. Готовые оценки T4/T5 FlyVis соответственно 0,794335→0,826478 и 0,349265→0,349265. Источник: runs/singleton_eight_input_20261108/scores.json.'
interpretation='До поиска оба восьмиканальных представления ранжируют свою пару выше чужой. После поиска простые признаки меняют порядок, T4/T5 сохраняют его. Равенство числа каналов у этого конкретного контроля не устраняет наблюдаемое различие. Дополнительная польза FlyVis для трекера и причина различия представлений не установлены.'
limitation='Два события, три ячейки на событие. Четыре EMA и четыре отклонения связаны линейно. Длительности 1,5; 3,9032; 12,2976; 32 шага истории выбраны до оценки из старой R1, нормировка прежняя обучающая. Фильтры обновляются по сохранённым наблюдениям, не по каждому кадру видео. Пространственная фильтрация и временная динамика FlyVis этим контролем не разделены. Вырезки 64 и 261 пиксель. Прошлая метка своей машины на кадре 61, образец на неразмеченном кадре 71, текущая метка на кадре 73.'
decision='Первая проблема открыта. Порог восстановления ID по двум событиям не устанавливать. Для следующего выбора метода учитывать риск случайного максимума на трёх ячейках.'
report='# Восемь простых признаков входа глаза и T4/T5\n\n'+summary+'\n\n'
report+='Восемь каналов взяты из прежнего R1. Четыре экспоненциальных скользящих средних с постоянными 1,5; 3,9032; 12,2976; 32 шага истории и четыре разности текущего входа с этими средними. Индексы 0, 5, 11, 16 выбраны как ближайшие к равномерным точкам старой логарифмической сетки до расчёта. Каждый канал использует старую обучающую нормировку. Чужие признаки уже были сохранены, свои восстановлены из пяти готовых входов глаза. Оценки FlyVis и прежнего изображения прочитаны готовыми.\n\n'
report+='| Пара | Представление | По прогнозу | После поиска | Прибавка |\n| --- | --- | --- | --- | --- |\n'
for pair in scores['pairs']:
    for key,title in [('input8','Восемь простых каналов'),('FlyVis','T4/T5'),('input8_permuted','Перестановка ячеек входа'),('FlyVis_permuted','Перестановка ячеек T4/T5')]:
        v=pair['variants'][key]
        kind='Своя' if pair['kind']=='own' else 'Чужая'
        report+=f'| {kind} {pair["old"]}→{pair["new"]} | {title} | {v["fixed"]:.6f} | {v["local"]:.6f} | {v["gain"]:+.6f} |\n'
report+='\n'+interpretation+'\n\n'+limitation+'\n\n'
report+='Корреляция является нормированным скалярным произведением. При копировании одной яркости в восемь одинаковых каналов числитель и произведение норм умножаются на восемь, оценка остаётся прежней. Это точное тождество, готовая оценка яркости повторно не вычислялась. Осмысленно различающиеся каналы могут изменить результат, как в новом временном контроле.\n\n'
report+='Максимум по 19 положениям поднял оценку чужой пары выше своей для простых признаков. Это наблюдаемое изменение порядка. Рабочее объяснение — подбор похожего короткого рисунка при малой поддержке. Сильнейшая альтернатива объяснению через нейродинамику FlyVis — полезную информацию даёт обычная пространственная обработка соседних ячеек. Небольшая различающая проверка должна использовать восемь простых пространственно-временных каналов на тех же ячейках и сдвигах, с условиями до новых оценок.\n\n'
report+='Проверка рекуррентной формулы EMA независимой закрытой формой дала ошибку 2,22·10⁻¹⁶. Независимая интерполяция и скалярные произведения согласовались до 10⁻⁹. Контрольные суммы входов стабильны.\n\n'+decision+'\n\nИзменение качества трекера в процентах и доля потолка не определены, поскольку измерено сходство двух пар.\n'
(OUT/'РЕЗУЛЬТАТ.md').write_text(report,encoding='utf8')
catalog_path=ROOT/'experiments/project_history.json'
catalog=read(catalog_path)
card=next(c for c in catalog['experiments'] if c['id']==OUT.name)
assert card['state']=='running'
card.update(state='inconclusive',done='2026-11-08',obtained=summary,interpretation=interpretation,decision=decision,cannot_claim=limitation)
card['acceptance'][0].update(met=True,evidence='runs/'+OUT.name+'/audit.json')
card['evidence'] += [dict(kind='file',ref='runs/'+OUT.name+'/'+name,claim='Первичный результат и проверка') for name in ['scores.json','audit.json','РЕЗУЛЬТАТ.md']]
card['metrics']=[dict(key='map_centered_similarity',value=v['local'],baseline=v['fixed'],subset=p['kind']+' / '+name,
    source='runs/'+OUT.name+'/scores.json',locator=f'pairs.{i}.variants.{name}.local') for i,p in enumerate(scores['pairs']) for name,v in p['variants'].items() if name in ['input8','FlyVis','input8_permuted']]
previous=next(c for c in catalog['experiments'] if c['id']=='singleton_matched_support_20261108')
old=previous['cannot_claim']
assert '522' in old
previous.setdefault('revisions',[]).append(dict(date='2026-11-08',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    field='cannot_claim',old=old,reason='Исправлена описательная ошибка размера вырезки. Реальная ширина 1280−1019=261. Максимальное смещение содержит два шага сетки, из него ошибочно выведена удвоенная ширина.'))
previous['cannot_claim']=old.replace('64 и 522','64 и 261')
previous['evidence'].append(dict(kind='file',ref='runs/'+OUT.name+'/crop_correction.json',claim='Исправление описания масштаба без пересчёта оценок'))
write(catalog_path,catalog)
correction=dict(previous_report='runs/singleton_matched_support_20261108/РЕЗУЛЬТАТ.md',previous_claim_px=522,
    corrected_extent_px=261,crop=[1019,51,1280,312],source='runs/singleton_matched_support_20261108/protocol.json',
    reason='Winning offset used two lattice steps, so deriving width assuming one step doubled the value.',
    old_results_reused=True)
write(OUT/'crop_correction.json',correction)
path=ROOT/'experiments/obsidian_experiment_topics.json'
topics=read(path);topics['topics'][OUT.name]=topics['topics']['singleton_matched_support_20261108'];write(path,topics)
path=ROOT/'experiments/system_problems.json'
registry=read(path);problem=next(p for p in registry['problems'] if p['id']=='С2')
problem['cases'].append(dict(date='2026-11-08',component='Восемь простых временных признаков входа',fact=summary,
    hypothesis='Поиск максимума на малой поддержке находит случайно похожий рисунок.',
    discriminating_check='Сопоставить с простым пространственно-временным контролем фиксированной размерности.',
    limitation=limitation,evidence=['runs/'+OUT.name+'/scores.json','runs/'+OUT.name+'/crop_correction.json']))
problem['evidence'].append(dict(source='runs/'+OUT.name+'/scores.json',sha256=sha(OUT/'scores.json'),claim='Равное число каналов у простого контроля'))
write(path,registry)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
plan=path.read_text(encoding='utf8')
entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Восемь простых каналов входа глаза против T4/T5.\n\n*Постановка.* Те же свои и чужие пары, 3 ячейки и 19 сдвигов. Карточка `{OUT.name}`.\n\n*Получено (`{OUT.name}`).* {summary} EMA сверена закрытой формой с ошибкой 2,22·10⁻¹⁶, новые оценки независимо сверены до 10⁻⁹ (`runs/{OUT.name}/audit.json`).\n\n*Вывод.* {interpretation} {limitation}\n\n*Поправка прежнего описания.* В записи 20:29 размер вырезки чужой пары ошибочно указан как 522 пикселя. По сохранённым координатам 1280−1019 он равен 261 пикселю. Перепутан один шаг решётки с двумя в выигравшем сдвиге. Прежняя запись и исходный отчёт сохранены, прежнее поле карточки перенесено в revisions. Оценки сходства взяты готовыми. Источник `runs/{OUT.name}/crop_correction.json`.\n\n*Решение.* {decision}\n\n'
path.write_text(plan.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
