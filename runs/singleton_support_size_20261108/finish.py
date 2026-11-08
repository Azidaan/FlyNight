"""Save the support-size comparison and its bounded interpretation."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'runs/singleton_forward_backward_transfer_20261108'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(OUT/'protocol.json');rows=read(OUT/'results.json')['records'];baseline=read(BASE/'results.json')['records']
assert read(OUT/'audit.json')['passed'] and len(rows)==28 and all('error_grid_steps' in r for r in rows)
names={'FlyVis':'T4/T5 FlyVis','center8':'Временные средние, повторённые до восьми каналов','neighbor8':'Временные средние и средние соседей','input8_temporal':'Временные средние и отклонения'}
summary={}
for name in names:
    summary[name]={}
    for kind in ['own','foreign']:
        rr=[r for r in rows if r['representation']==name and r['kind']==kind]
        summary[name][kind]=dict(count=len(rr),three_errors=[r['baseline_error_grid'] for r in rr],
            all_errors=[r['error_grid_steps'] for r in rr],three_exact=sum(r['baseline_error_grid']==0 for r in rr),
            all_exact=sum(r['error_grid_steps']==0 for r in rr),
            error_decreased=sum(r['error_grid_steps']<r['baseline_error_grid'] for r in rr),
            error_increased=sum(r['error_grid_steps']>r['baseline_error_grid'] for r in rr))
summary['FlyVis']['directional_question']=dict(own_exact_increased=summary['FlyVis']['own']['all_exact']>summary['FlyVis']['own']['three_exact'],
    foreign_exact_decreased=summary['FlyVis']['foreign']['all_exact']<summary['FlyVis']['foreign']['three_exact'])
# Descriptive inequalities on the saved scores, no classifier or fitted cut-off.
dominance=[]
for foreign in [r for r in rows if r['representation']=='FlyVis' and r['kind']=='foreign']:
    for own in [r for r in rows if r['representation']=='FlyVis' and r['kind']=='own']:
        if foreign['forward_score']>=own['forward_score'] and foreign['error_grid_steps']<=own['error_grid_steps'] and (foreign['forward_score']>own['forward_score'] or foreign['error_grid_steps']<own['error_grid_steps']):
            dominance.append(dict(foreign=[foreign['video'],foreign['old'],foreign['new']],own=[own['video'],own['old'],own['new']],
                foreign_score=foreign['forward_score'],own_score=own['forward_score'],foreign_return=foreign['error_grid_steps'],own_return=own['error_grid_steps']))
summary['descriptive_monotone_order']=dict(role='Post-result fixed-order explanation, no fitting. Higher similarity and lower return error are the conventional directions.',pairs=dominance)
save(OUT/'summary.json',summary)
obtained='На тех же 7 парах увеличение с 3 до всех 8–57 допустимых ячеек повысило число точных возвратов FlyVis своих с 1/4 до 3/4; у чужих осталось 2/3. Для 141→173 ошибка изменилась 0→1 шаг, для 349→402 — 3→0, для 405→417 — 0→0. У центральных временных средних свои 3/4→4/4 и чужие 3/3→3/3; у средних с соседями свои 2/4→4/4 и чужие 3/3→3/3; у средних с отклонениями свои 2/4→2/4 и чужие 0/3→2/3. Источники `runs/'+OUT.name+'/results.json`, `runs/'+OUT.name+'/summary.json`.'
interpretation='Сокращение шаблона до трёх выбранных опор ухудшало согласованность возврата у части своих пар. Полная поддержка исправила два ненулевых возврата FlyVis своих, но не уменьшила число нулевых возвратов чужих. Два простых контроля замкнули путь на всех семи парах. Поэтому увеличение поддержки стабилизирует сопоставление и для чужих объектов и не превращает согласованность в подтверждение личности. У FlyVis все семь выбранных обратных максимумов лежат внутри прежней рамки; ограничение этой рамкой оставит их прежними.'
limitation='Только 7 пар и разные числа полных опор. Все чужие пары меняют масштаб вырезки, свои сохраняют его. Внутри пары масштаб одинаков для обеих версий шаблона. Добавление ячеек меняет и их число, и содержимое, включая фон внутри рамки; вклад одного числа ячеек не выделен. Пороги и классификатор не подбирались. Прирост сохранения ID и HOTA не измерялся, причинный вывод о ядре FlyVis невозможен.'
decision='Одного расширения пространственного шаблона недостаточно для решения первой проблемы. Ошибку возврата не принимать самостоятельным критерием личности. Полные результаты сохранить как вариант сравнения. Сильнейшая альтернатива объяснению нехваткой точек — сходный пространственный сигнал у разных машин и фона, который текущая нормированная корреляция сопоставляет согласованно. Новый результат локализует недостаточность текущего пространственного считывателя; наличие информации в полном модуле FlyVis остаётся отдельным вопросом. Следующую научную постановку автоматически не запускать.'
report='# Три ячейки и вся допустимая поддержка\n\n'+obtained+'\n\n'+interpretation+'\n\n'
report+='## 1. Математика и неизменные условия\n\nЦентрирование N наблюдений задаёт проекцию I−11ᵀ/N ранга N−1. Три ячейки дают в каждом канале только два независимых пространственных отклонения. Большее число разных ячеек может добавить информацию, однако не гарантирует разделение машин. Простое повторение тех же наблюдений не меняет нормированную корреляцию; это проверено до оценок в mathematical_checks.json.\n\n'
report+='Изменён только набор опор: три прежние геометрически выбранные ячейки заменены всем прежним списком eligible_cells. Карты, история, масштаб, прогноз, 19 сдвигов, восемь каналов каждого представления, центрирование и общий знаменатель сохранены. Внутри каждой пары все представления используют одинаковые опоры. Обратный поиск независимый, по всей сохранённой прошлой карте глаза.\n\n'
report+='## 2. Ошибка возврата, три ячейки → все допустимые\n\n| Пара | Принадлежность | Число ячеек | FlyVis | Средние | Средние + соседи | Средние + отклонения |\n| --- | --- | --- | --- | --- | --- | --- |\n'
for c in p['cases']:
    rr=[r for r in rows if (r['video'],r['old'],r['new'])==(c['video'],c['old'],c['new'])];by={r['representation']:r for r in rr}
    values=' | '.join(f'{by[n]["baseline_error_grid"]}→{by[n]["error_grid_steps"]}' for n in names)
    report+=f'| {c["old"]}→{c["new"]} | {"своя" if c["kind"]=="own" else "чужая"} | 3→{c["points"]} | {values} |\n'
report+='\nОшибки указаны в шагах прежней решётки поиска. Ноль — точное замыкание пути, а не подобранный рабочий порог. Полные ошибки в пикселях и все оценки сохранены в results.json.\n\n'
report+='## 3. Прямой выбор и возврат\n\n| Пара | Представление | Прямой сдвиг 3→все | Обратный сдвиг 3→все | Максимум прямого сходства 3→все | Ошибка в пикселях 3→все |\n| --- | --- | --- | --- | --- | --- |\n'
for r in rows:
    old=baseline[r['three_cell_result_index']]
    report+=f'| {r["old"]}→{r["new"]} | {names[r["representation"]]} | {r["baseline_forward_axes"]}→{r["forward_axes"]} | {r["baseline_backward_axes"]}→{r["backward_axes"]} | {old["forward_score"]:.6f}→{r["forward_score"]:.6f} | {r["baseline_error_px"]:.6f}→{r["error_px"]:.6f} |\n'
report+='\n## 4. Почему одного порога недостаточно\n\nЭто дополнительное описательное сопоставление готовых чисел, без настройки. У чужой 405→417 с полной поддержкой сходство FlyVis 0,750330 и ошибка возврата 0, у своей 770→844 — 0,696433 и 1 шаг. Если использовать только эти два числа и считать большее сходство и меньшую ошибку лучшими, чужая пара не хуже своей по обоим признакам. Монотонное правило на этих двух числах не сможет принять эту свою пару и отвергнуть эту чужую. Это не доказательство невозможности другого считывания карт или другого признака личности. Точные значения и остальные такие неравенства находятся в summary.json, descriptive_monotone_order.\n\n'
report+='## 5. Граница вывода и решение\n\n'+limitation+'\n\n'+decision+'\n\nРазличающая следующая проверка для объяснения сходством сигнала — сравнить согласованность с маской принадлежности объекту или с прямыми конкурентами на одной сцене, сохранив контроль без FlyVis. Она здесь не выполнялась; один рост числа опор уже проверен и не решил задачу.\n\n'
report+='## 6. Повторное использование и проверки\n\nВсе 28 результатов трёх ячеек прочитаны из прежнего файла. Прямой поиск FlyVis для 1738→1834 на восьми ячейках тоже прочитан готовым после точного совпадения опор, прогноза, смещений и сумм карт. Выполнены 27 новых прямых и 28 обратных поисков по 19 сдвигам. Все обратные области поддержаны сохранённым глазом. Карты C/R1, видео, вход глаза и сеть не пересчитывались. Интерполяция и скалярные произведения независимо согласованы до 10⁻⁹, средние соседей совпали точно, исходные суммы сохранены.\n\nЛитературный принцип независимого возврата остаётся тем же: Kalal, Mikolajczyk, Matas, 2010, DOI 10.1109/ICPR.2010.675. Реестр experiments/useful_literature.json дополнен результатом этой проверки.\n\nПроцент изменения качества полного трекера и доля потолка не определены: измерялись переносы карт, а не выдача ID полным трекером.\n'
# Bind displayed explanatory decimals to their primary values.
foreign=next(r for r in rows if r['representation']=='FlyVis' and r['old']==405)
own=next(r for r in rows if r['representation']=='FlyVis' and r['old']==770)
report=report.replace('0,750330',f'{foreign["forward_score"]:.6f}'.replace('.',',')).replace('0,696433',f'{own["forward_score"]:.6f}'.replace('.',','))
with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8') as f:f.write(report)
path=ROOT/'experiments/project_history.json';cat=read(path);card=next(c for c in cat['experiments'] if c['id']==OUT.name)
assert card['state']=='running'
card.update(state='inconclusive',done='2026-11-08',obtained=obtained,interpretation=interpretation,cannot_claim=limitation,decision=decision)
card['acceptance'][0].update(met=True,evidence=f'runs/{OUT.name}/audit.json')
card['evidence'] += [dict(kind='file',ref=f'runs/{OUT.name}/{file}',claim='Результат сравнения и его проверка') for file in ['mathematical_checks.json','results.json','summary.json','audit.json','РЕЗУЛЬТАТ.md']]
card['metrics']=[dict(key=key,value=r[field],baseline=r[oldfield],subset=f'{r["video"]} / {r["old"]}→{r["new"]} / {r["kind"]} / {r["representation"]}',
    source=f'runs/{OUT.name}/results.json',locator=f'records.{i}.{field}') for i,r in enumerate(rows) for key,field,oldfield in [('forward_backward_error_px','error_px','baseline_error_px'),('forward_backward_error_grid','error_grid_steps','baseline_error_grid')]]
save(path,cat)
path=ROOT/'experiments/obsidian_experiment_topics.json';topics=read(path);topics['topics'][OUT.name]=topics['topics'][BASE.name];save(path,topics)
path=ROOT/'experiments/system_problems.json';problems=read(path);problem=next(x for x in problems['problems'] if x['id']=='С2')
problem['cases'].append(dict(date='2026-11-08',component='Пространственное считывание готовых карт: число опор и нормированная корреляция',fact=obtained,
    hypothesis='Разные машины и фон могут давать сходный рельеф движения и контраста, замыкающийся при обратном сопоставлении.',
    discriminating_check='Отделить принадлежность сигналов объекту от общего контекста; сравнивать те же области с простыми признаками без FlyVis.',
    limitation=limitation,evidence=[f'runs/{OUT.name}/results.json',f'runs/{OUT.name}/summary.json']))
problem['evidence'].append(dict(source=f'runs/{OUT.name}/results.json',sha256=sha(OUT/'results.json'),claim='Увеличение поддержки помогло части своих возвратов, но чужие нулевые возвраты сохранились'))
save(path,problems)
path=ROOT/'experiments/useful_literature.json';literature=read(path);paper=next(x for x in literature['sources'] if x['id']=='kalal2010_forward_backward')
paper['application']+='\n\n08.11.2026. На тех же семи парах проверено расширение шаблона с трёх ячеек до всех допустимых. Возврат стал устойчивее у части своих пар, но точное замыкание осталось у двух из трёх чужих. Это конкретное ограничение использования согласованности как признака личности; первичный результат runs/'+OUT.name+'/results.json.'
paper['evidence'] += [f'runs/{OUT.name}/protocol.json',f'runs/{OUT.name}/results.json'];paper['related'].append('exp:'+OUT.name);save(path,literature)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';plan=path.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)))
entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Полная поддержка стабилизировала возврат, но не отделила чужую машину.\n\n*Постановка.* Те же семь пар, единственное изменение — три опоры заменены всем прежним допустимым набором. Карточка `{OUT.name}`.\n\n*Получено (`{OUT.name}`).* {obtained}\n\n*Вывод.* {interpretation} {limitation}\n\n*Исполнение.* Готовые 28 базовых результатов и один полный прямой результат использованы без повтора. Новые 27 прямых и 28 обратных поисков сверены до 10⁻⁹. Дополнительное сравнение готовых чисел показало доминирование чужой 405→417 над своей 770→844 по двум обычным направлениям: больше сходства и меньше ошибка. Определение и точные числа в summary.json, descriptive_monotone_order; классификатор не обучался. Литература Kalal дополнена ссылками на проверку.\n\n*Решение.* {decision}\n\n'
path.write_text(plan.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
save(OUT/'execution_receipt.json',dict(finished=now.isoformat(),checksums={name:sha(OUT/name) for name in ['study.py','finish.py','protocol.json','mathematical_checks.json','results.json','summary.json','audit.json','РЕЗУЛЬТАТ.md']}))
print(obtained)
