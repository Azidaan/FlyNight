"""Record the finished bounded comparison without changing any old conclusion."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(OUT/'protocol.json');result=read(OUT/'results.json');rows=result['records']
assert read(OUT/'audit.json')['passed'] and read(OUT/'inventory_provenance.json')['passed']
assert len(rows)==28 and all(r['backward_supported'] for r in rows)
groups={name:{kind:[r for r in rows if r['representation']==name and r['kind']==kind] for kind in ['own','foreign']} for name in ['FlyVis','center8','neighbor8','input8_temporal']}
summary={name:{kind:dict(count=len(rr),exact_returns=sum(r['error_grid_steps']==0 for r in rr),
    errors_grid=[r['error_grid_steps'] for r in rr],errors_px=[r['error_px'] for r in rr]) for kind,rr in kinds.items()} for name,kinds in groups.items()}
save(OUT/'summary.json',summary)
obtained='На 7 новых прямо размеченных парах (4 своих, 3 чужих) точный возврат FlyVis получен у 1/4 своих и 2/3 чужих. Ошибки в шагах у своих 1, 1, 2, 0; у чужих 0, 0, 3. Центральные временные средние дали точный возврат у 3/4 своих и 3/3 чужих; средние с соседями — у 2/4 и 3/3; средние с отклонениями — у 2/4 и 0/3. Источники `runs/'+OUT.name+'/results.json`, `runs/'+OUT.name+'/summary.json`.'
interpretation='Различие первого опыта на двух парах не перенеслось. Ошибка независимого возврата по T4/T5 в этой трёхточечной реализации не отделяет свою машину от чужой. Два чужих нулевых возврата лежат внутри старой рамки, поэтому ограничение обратной области этой рамкой их не устраняет. У трёх своих пар FlyVis ошибка ненулевая. Сравнение с простыми признаками также не даёт чистого разделения.'
limitation='Только 7 событий, 3 опорные ячейки, без подбора порогов. Все 3 чужих случая меняют масштаб вырезки глаза, 4 своих — с постоянным масштабом; группы не согласованы по этому фактору. Между кадрами разметки история старого ID подтверждается сохранёнными опорными наблюдениями, а текущая чужая или своя рамка проверена напрямую. Целые карты, другие опоры и качество полного трекера здесь не проверены. Из сравнения нельзя заключать о причинном преимуществе ядра FlyVis или назначать рабочий порог.'
decision='Не использовать ошибку возврата этой схемы как самостоятельное подтверждение личности или готовое основание отказа. Первую проблему оставить открытой. Сильнейшая альтернатива — неустойчивость пространственного соответствия при трёх ячейках и изменении масштаба, а не отсутствие информации в полном модуле FlyVis. Для следующего решения нужны согласованная геометрия и подтверждённое сравнение внутри самой карты; автоматически новый этап не запускался.'
names={'FlyVis':'T4/T5 FlyVis','center8':'Четыре временных средних, повторённые до восьми каналов','neighbor8':'Временные средние и средние по соседям','input8_temporal':'Временные средние и отклонения'}
report='# Проверка возврата на других размеченных парах\n\n'+obtained+'\n\n'+interpretation+'\n\n'
report+='## 1. Выборка и математика\n\nВсе другие прямо размеченные пары из существующей проверки 86 видео, у которых есть минимум три общие допустимые ячейки. Две свои пары с нулевой поддержкой исключены до оценок. Прежние 390→448 и 3→273 не входят. Своя 1738→1834 ранее сравнивалась с восемью ячейками; нынешние три геометрически выбранные ячейки — новая постановка, прежние оценки на восьми не пересчитаны.\n\n'
report+='Три опоры выбираются без чтения активности: наименьший допустимый индекс, затем наиболее удалённые от уже выбранных. У всех четырёх представлений одинаковы опоры, 19 сдвигов, нормировка и мера сходства. При неодинаковых размерах глаза координаты переводятся в пиксели кадра отдельно для прошлой и текущей карты. Шаг поиска задан большим из двух масштабов, как в прежнем геометрическом допуске. Это перенос по общей системе координат, без оценки масштаба по сходству. Обратный поиск независимый; остаток равен δf+δb. Инверсия координат при разных размерах вырезки проверена до оценок. Прежняя проверка нетривиальности обратного поиска использована готовой.\n\n'
report+='## 2. Ошибки возврата в шагах поиска\n\n| Пара | Видео | Принадлежность | Масштаб глаза | FlyVis | Центр | Соседи | Отклонения |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n'
for c in p['cases']:
    rr=[r for r in rows if (r['video'],r['old'],r['new'])==(c['video'],c['old'],c['new'])]
    by={r['representation']:r for r in rr}
    values=' | '.join(str(by[n]['error_grid_steps']) for n in names)
    report+=f'| {c["old"]}→{c["new"]} | {c["video"]} | {"своя" if c["kind"]=="own" else "чужая"} | {"постоянный" if c["same_scale"] else "меняется"} | {values} |\n'
report+='\nШаги — max(|u|, |v|, |u+v|) для остатка двух сдвигов. Ноль здесь описывает точное замыкание пути и не является выбранным рабочим порогом.\n\n'
report+='## 3. Полные значения\n\n| Пара | Представление | Прямой сдвиг | Обратный сдвиг | Ошибка, пиксели | Внутри прежней рамки | Геометрический возврат, шаги |\n| --- | --- | --- | --- | --- | --- | --- |\n'
for r in rows:
    report+=f'| {r["old"]}→{r["new"]} | {names[r["representation"]]} | {r["forward_axes"]} | {r["backward_axes"]} | {r["error_px"]:.6f} | {"да" if r["winner_inside_old_box"] else "нет"} | {r["geometry_only_return_grid"]} |\n'
report+='\nГеометрический возврат использует тот же выбранный текущий участок, но возвращает его только на −d0, без чтения прошлой карты; остаток δf. Полностью геометрические переносы d0 и −d0 замыкаются в ноль для всех пар и ничего не подтверждают о личности. У чужих 405→417 и 141→173 FlyVis выбрал нулевые сдвиги в обе стороны, итог совпал с этим геометрическим контролем.\n\n'
report+='## 4. Исполнение и ограничения\n\nВзяты готовыми 6 карт C/R1, восстановлены 8 отсутствующих. Для недостающих карт использована одна общая причинная последовательность с сохранением результата на конце прошлого шаблона и текущего поиска. Из 20 нужных входов глаза 13 взяты готовыми и 7 восстановлены; два свидетеля рендеринга совпали побайтно. Веса и исходные файлы не изменились. Интерполяция и оценки сверены другим вычислением до 10⁻⁹, среднее соседей совпало точно. Все 28 обратных поисков полностью поддержаны сохранённым глазом. См. audit.json, maps_complete.json, inventory_provenance.json.\n\n'+limitation+'\n\n'
report+='## 5. Литература и решение\n\nПринцип независимого возврата взят из Kalal, Mikolajczyk, Matas (2010), DOI 10.1109/ICPR.2010.675. Статья и область применения уже сохранены в experiments/useful_literature.json, раздел 2 и рисунок 1. Новый результат добавлен к использованию источника, отдельная копия статьи не создавалась.\n\n'+decision+'\n\nПроцент изменения качества трекера и доля потолка не определены: измерялась ошибка переноса, полный трекер не запускался.\n'
with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8') as f:f.write(report)
catpath=ROOT/'experiments/project_history.json';cat=read(catpath)
card=next(c for c in cat['experiments'] if c['id']==OUT.name);assert card['state']=='running'
card.update(state='inconclusive',done='2026-11-08',obtained=obtained,interpretation=interpretation,cannot_claim=limitation,decision=decision)
card['acceptance'][0].update(met=True,evidence='runs/'+OUT.name+'/audit.json')
card['evidence'] += [dict(kind='file',ref=f'runs/{OUT.name}/{file}',claim='Измеренный результат и проверка') for file in ['results.json','summary.json','audit.json','maps_complete.json','inventory_provenance.json','РЕЗУЛЬТАТ.md']]
card['metrics']=[dict(key=key,value=r[field],baseline=None,subset=f'{r["video"]} / {r["old"]}→{r["new"]} / {r["kind"]} / {r["representation"]}',source=f'runs/{OUT.name}/results.json',locator=f'records.{i}.{field}') for i,r in enumerate(rows) for key,field in [('forward_backward_error_px','error_px'),('forward_backward_error_grid','error_grid_steps')]]
save(catpath,cat)
path=ROOT/'experiments/obsidian_experiment_topics.json';topics=read(path)
topics['topics'][OUT.name]=topics['topics']['singleton_forward_backward_20261108'];save(path,topics)
path=ROOT/'experiments/system_problems.json';problems=read(path);problem=next(p for p in problems['problems'] if p['id']=='С2')
problem['cases'].append(dict(date='2026-11-08',component='Перенос проверки возврата карт на другие прямо размеченные пары',fact=obtained,
    hypothesis='Критерий может терять избирательность из-за трёхточечного шаблона и несогласованного масштаба.',
    discriminating_check='Разделить влияние пространственной поддержки и изменения масштаба на новых подтверждённых парах, до внедрения критерия в трекер.',
    limitation=limitation,evidence=[f'runs/{OUT.name}/results.json',f'runs/{OUT.name}/protocol.json']))
problem['evidence'].append(dict(source=f'runs/{OUT.name}/results.json',sha256=sha(OUT/'results.json'),claim='Прежнее различие ошибок возврата не перенеслось на 7 новых пар'))
save(path,problems)
path=ROOT/'experiments/useful_literature.json';literature=read(path)
source=next(s for s in literature['sources'] if s['id']=='kalal2010_forward_backward')
source['application']+='\n\n08.11.2026. Та же независимая проверка перенесена на 7 других прямо размеченных пар. Прежнее различие не сохранилось; первичный результат runs/'+OUT.name+'/results.json. Польза источника — проверяемая диагностика согласованности, она не даёт гарантии личности.'
source['limitation']+=' Указанные два события относятся к первой проверке. В последующем переносе на семь других событий группы дополнительно различались изменением масштаба глаза; результат не позволяет принять правило назначения ID.'
source['related'].append('exp:'+OUT.name)
source['evidence'] += [f'runs/{OUT.name}/protocol.json',f'runs/{OUT.name}/results.json']
save(path,literature)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=path.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)))
entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Перенос проверки возврата не подтвердил различие своих и чужих машин.\n\n*Постановка.* Другие прямо размеченные пары, прежние четыре представления, 3 геометрические опоры и 19 сдвигов. Карточка `{OUT.name}`.\n\n*Получено (`{OUT.name}`).* {obtained}\n\n*Вывод.* {interpretation} {limitation}\n\n*Исполнение.* Готовые 6 карт использованы без повторения, 8 отсутствующих восстановлены. Добавлены 7 недостающих входов глаза, 13 готовых использованы. Совпадение свидетелей точное, независимые оценки согласованы до 10⁻⁹, источники сохранены. Полезная статья Kalal дополнена ссылками на перенос проверки через реестр литературы.\n\n*Решение.* {decision}\n\n'
path.write_text(plan.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
files=['protocol.json','mathematical_checks.json','results.json','summary.json','audit.json','maps_complete.json','inventory_provenance.json','РЕЗУЛЬТАТ.md','study.py','verify_inventory.py','finish.py']
save(OUT/'execution_receipt.json',dict(finished=now.isoformat(),checksums={file:sha(OUT/file) for file in files}))
print(obtained)
