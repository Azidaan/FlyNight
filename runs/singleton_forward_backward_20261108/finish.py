import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
results=read(OUT/'results.json')
now=datetime.now(timezone(timedelta(hours=3)))
geometry=[]
for row in results['records']:
    forward=row['forward_reused']['winning_offset_px']
    axial=row['forward_axes']
    geometry.append(dict(kind=row['kind'],representation=row['representation'],
        explanation='Use the same selected current patch but return only by -d0, without reading the old map.',
        error_px=math.hypot(*forward),error_grid_steps=max(abs(axial[0]),abs(axial[1]),abs(sum(axial)))))
write(OUT/'geometry_control.json',dict(role='Explanatory fixed-prediction control, formula follows from registered residual; no fitted parameters.',records=geometry,
    baseline_without_any_map='Forward and backward predictions d0 and -d0 have zero cycle error for both pairs.'))
summary='Ошибка возврата FlyVis равна 0 пикселей/0 шагов решётки на своей 3→273 и 9,702019 пикселя/1 шаг на чужой 390→448. Контроль среднего соседей дал 2,127877 пикселя/1 шаг своей и 8,677749 пикселя/1 шаг чужой. Центральные EMA дали 2,127877 пикселя/1 шаг своей и 15,644035 пикселя/2 шага чужой. Прежние EMA+отклонения дали ноль обеим парам. Источник: runs/singleton_forward_backward_20261108/results.json.'
interpretation='В этой реализации независимый обратный поиск по T4/T5 различил две пары. Простой контроль с соседями дал одинаковую ошибку в шагах, а прежний временной контроль подтвердил и чужую машину. Разница пикселей простого контроля связана с разным масштабом вырезки. Нулевой возврат не является гарантией личности.'
limitation='Два события и три точки, прежняя разметка своей машины на кадре 61, шаблон на неразмеченном 71, текущая рамка на 73. Обратный поиск использует весь сохранённый глаз; часть проверяемых гипотез выходит за прежнюю рамку, все выбранные максимумы лежат внутри неё. Порог не подбирался. Это согласованность готовых карт, не обращение динамики ядра и не оценка сохранения ID полным трекером.'
decision='Первую проблему оставить открытой. Не принимать правило нулевой ошибки по двум событиям. Результат даёт конкретного кандидата на отказ по T4/T5, который требует самостоятельной проверки на других подтверждённых своих и чужих соответствиях.'
names={'FlyVis':'T4/T5 FlyVis','center8':'Четыре центральных EMA, повторённые дважды','neighbor8':'EMA и среднее соседей','input8_temporal':'EMA и отклонения'}
report='# Прямое и независимое обратное сопоставление\n\n'+summary+'\n\n'
report+='## Математика до сравнения\n\nДля исходных центров x_i готовый прямой поиск дал y_i=x_i+d0+δf. Новый шаблон извлечён из текущей карты в y_i. Обратный поиск независимо сравнивает его с прошлой картой в y_i−d0+δb по тем же 19 значениям δb. Ошибка возврата равна норме δf+δb. Обратный прогноз −d0 задан геометрией; выигравший δb выбирается по прошлой карте.\n\n'
report+='Точный искусственный перенос возвращается в начало. Другой специально построенный пример даёт прямой сдвиг (1;0), обратный (0;1), остаток (1;1). Поэтому код не подставляет противоположный прямому сдвиг. При одинаковых признаках двух разных объектов нулевая ошибка возможна и для чужого объекта. Постоянный шаблон даёт отказ из-за нулевой нормы.\n\n'
report+='## Измеренные ошибки\n\n| Пара | Представление | Прямой сдвиг по осям | Обратный сдвиг по осям | Ошибка, пиксели кадра | Ошибка, шаги решётки |\n| --- | --- | --- | --- | --- | --- |\n'
for r in results['records']:
    kind='Своя' if r['kind']=='own' else 'Чужая'
    report+=f'| {kind} {r["old"]}→{r["new"]} | {names[r["representation"]]} | {r["forward_axes"]} | {r["backward_axes"]} | {r["error_px"]:.6f} | {r["error_grid_steps"]} |\n'
report+='\nШаги указаны в осевой метрике гексагональной решётки max(|u|, |v|, |u+v|). Дополнительно в results.json сохранена евклидова ошибка в координатах глаза.\n\n'+interpretation+'\n\n'
report+='## Геометрический контроль\n\nЕсли выбранный текущий участок переносить обратно только по прогнозу −d0, без чтения прошлой карты, остаток равен δf. Для FlyVis это один шаг на своей и ноль на чужой. Независимое чтение прошлой карты поменяло этот порядок на 0/1. Для контроля с соседями обратный максимум совпал с обратным прогнозом, поэтому его 1/1 шага полностью воспроизводятся таким геометрическим возвратом. Полностью геометрическая пара прогнозов d0 и −d0 замыкается в ноль у обеих пар. Первичные значения в geometry_control.json.\n\n'+limitation+'\n\n'
report+='## Проверка исполнения\n\nПервый запуск остановлен до сохранения результатов из-за индексации Delaunay-преобразования для массива (3;2). Замороженный study.py сохранён. recovery1.py исправляет только обращение к последним двум измерениям, условия опыта сохранены в исходном протоколе и recovery1_protocol.json. Линейные поля в формах (3;2) и (3;19;2) восстановлены с ошибкой 4,44·10⁻¹⁶. Интерполяция и скалярное произведение сверены независимо до 10⁻⁹. Готовые прямые оценки, карты и проходы FlyVis повторно не вычислялись.\n\n'
report+='## Литературное основание\n\nKalal, Mikolajczyk, Matas, 2010, Forward-Backward Error: Automatic Detection of Tracking Failures. DOI 10.1109/ICPR.2010.675. Раздел 2 задаёт независимый возврат; авторы явно исключают методы, согласованные по построению. Наша адаптация сравнивает готовые карты, а не воспроизводит Lucas–Kanade. Источник помог выбрать математический контроль и меру ошибки. Подробности и полезная работа Bolme сохранены в ЛИТЕРАТУРА.md и experiments/useful_literature.json.\n\n'+decision+'\n\nПроцент прироста качества трекера и доля потолка не определены, поскольку измерены ошибки переноса двух пар.\n'
(OUT/'РЕЗУЛЬТАТ.md').write_text(report,encoding='utf8')
literature='# Использованная литература\n\nKalal, Z., Mikolajczyk, K., Matas, J. (2010). Forward-Backward Error: Automatic Detection of Tracking Failures. DOI 10.1109/ICPR.2010.675. [Первичный текст](https://cmp.felk.cvut.cz/ftp/articles/matas/kalal-2010-fb_track-icpr.pdf).\n\nРаздел 2 и рисунок 1 использованы для независимого обратного поиска. Конец раздела 2 предупреждает о согласованности по построению. Это привело к отдельному математическому примеру, где обратный максимум отличается от обратного прямому. Раздел 3 исследует пороги ошибки, поэтому правило точного нуля не выдается за опубликованную гарантию. Наши результаты в results.json не воспроизводят весь Median Flow и не закрывают С2.\n\nBolme, D. S., Beveridge, J. R., Draper, B. A., Lui, Y. M. (2010). Visual Object Tracking using Adaptive Correlation Filters. DOI 10.1109/CVPR.2010.5539960. [Первичный текст](https://iacl.ece.jhu.edu/proceedings/cvpr2010/papers/1905.pdf).\n\nРаздел 3.5 задаёт PSR — отношение пика к боковым откликам. В источнике исключено окно 11×11 в плотной карте. Это помогло отклонить прямой перенос критерия на 19 сдвигов и толкование нулевого прироста как плато. В готовых scores.json опыта singleton_neighbor_input_20261108 у чужой пары FlyVis максимум 0,349265, минимум −0,410454. PSR в текущем опыте не вычислялся.\n\nОба источника связаны с проблемой С2 и соответствующими экспериментами через генератор. Граф хранит литературное основание, нашу адаптацию и границы вывода раздельно.\n'
(OUT/'ЛИТЕРАТУРА.md').write_text(literature,encoding='utf8')
path=ROOT/'experiments/metrics_glossary.json'
glossary=read(path)
glossary['metrics']['forward_backward_error_px']=dict(title='Ошибка возврата в пикселях кадра',ru='Евклидово расстояние между исходной точкой и положением после прямого и независимого обратного переноса.',better='lower',unit='пиксели',caveat='Нулевой возврат не доказывает личность. При разных размерах вырезки пиксельные ошибки требуют нормировки.',source='runs/'+OUT.name+'/protocol.json')
glossary['metrics']['forward_backward_error_grid']=dict(title='Ошибка возврата в шагах гексагональной решётки',ru='Осевая метрика остатка сдвигов max(|u|,|v|,|u+v|), одинаковая при переводе глаза в исходный кадр.',better='lower',unit='шаги решётки',caveat='Малая ошибка согласованности не является калиброванной вероятностью правильного ID.',source='runs/'+OUT.name+'/protocol.json')
write(path,glossary)
path=ROOT/'experiments/project_history.json'
catalog=read(path);card=next(c for c in catalog['experiments'] if c['id']==OUT.name)
assert card['state']=='running'
card.update(state='inconclusive',done='2026-11-08',obtained=summary,interpretation=interpretation,cannot_claim=limitation,decision=decision)
card['acceptance'][0].update(met=True,evidence='runs/'+OUT.name+'/audit.json')
card['evidence'] += [dict(kind='file',ref='runs/'+OUT.name+'/'+name,claim='Первичный результат, проверка или литературное основание') for name in ['results.json','audit.json','geometry_control.json','execution_receipt.json','РЕЗУЛЬТАТ.md','ЛИТЕРАТУРА.md']]
card['metrics']=[dict(key=key,value=r[field],baseline=None,subset=r['kind']+' / '+r['representation'],source='runs/'+OUT.name+'/results.json',locator=f'records.{i}.{field}') for i,r in enumerate(results['records']) for key,field in [('forward_backward_error_px','error_px'),('forward_backward_error_grid','error_grid_steps')]]
write(path,catalog)
path=ROOT/'experiments/obsidian_experiment_topics.json';topics=read(path);topics['topics'][OUT.name]=topics['topics']['singleton_neighbor_input_20261108'];write(path,topics)
path=ROOT/'experiments/system_problems.json';registry=read(path);problem=next(p for p in registry['problems'] if p['id']=='С2')
problem['cases'].append(dict(date='2026-11-08',component='Независимая обратная проверка готовых карт',fact=summary,hypothesis='Ошибка возврата по T4/T5 может выявлять часть чужих соответствий.',discriminating_check='Проверить фиксированное правило на других подтверждённых своих и чужих парах при согласованной геометрии.',limitation=limitation,evidence=['runs/'+OUT.name+'/results.json','runs/'+OUT.name+'/geometry_control.json']))
problem['evidence'].append(dict(source='runs/'+OUT.name+'/results.json',sha256=sha(OUT/'results.json'),claim='Прямое и независимое обратное сопоставление'))
write(path,registry)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=path.read_text(encoding='utf8')
entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Результат обратного сопоставления и сохранение полезной литературы.\n\n*Постановка.* Прямые оценки готовые, обратный максимум вычислен независимо. Карточка `{OUT.name}`.\n\n*Получено (`{OUT.name}`).* {summary} Геометрический возврат выбранного участка только по −d0 не разделяет эти пары как полноценный сигнал; точные значения в `runs/{OUT.name}/geometry_control.json`.\n\n*Вывод.* {interpretation} {limitation}\n\n*Исполнение.* Сбой индексации сохранён отдельно, замороженный источник не переписан. recovery1 исправляет только форму индексов; проверка линейного поля дала 4,44·10⁻¹⁶, независимые оценки согласованы до 10⁻⁹.\n\n*Литература.* Kalal et al. 2010 использован для независимого возврата и ограничения применимости. Bolme et al. 2010 использован для проверки смысла выраженности пика. Реестр `experiments/useful_literature.json`, материал `runs/{OUT.name}/ЛИТЕРАТУРА.md`; отдельные содержательные узлы и двусторонние стелс-связи формирует генератор.\n\n*Решение.* {decision}\n\n'
path.write_text(plan.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
