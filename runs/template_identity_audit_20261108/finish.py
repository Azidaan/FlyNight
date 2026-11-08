"""Record the fixed audit conclusions without changing old labels or scores."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=600)
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
if (OUT/'execution_receipt.json').exists():raise SystemExit('Ready reporting receipt exists; no rewrite.')
p=read(OUT/'protocol.json');r=read(OUT/'results.json');assert r['passed'] and len(r['records'])==200
for path,digest in p['sources'].items():assert sha(ROOT/path)==digest,path
for prior in ['background_patch_gate_20261108','decoder34_identity_gate_20261108']:
    for path,digest in read(ROOT/'runs'/prior/'results.json')['files'].items():assert sha(ROOT/path)==digest,path
from collections import Counter
counts=Counter();known=Counter();keys=set()
for row in r['records']:
    key=(row['video'],row['frame'],row['old'],row['new']);assert key not in keys;keys.add(key)
    ref=row['reference_history'][-1];cur=row['current'];label=row['conditional_label']
    for value in row['reference_history']+[cur,row['native_current']]:
        qualified=[o for o in value['overlaps'] if o['iou']>=.5]
        if value['status']=='unannotated':assert not value['overlaps'] and value['identity'] is None
        else:
            expected='unique' if len(qualified)==1 else 'ambiguous' if qualified else 'unmatched'
            assert value['status']==expected
            assert value['identity']==(qualified[0]['identity'] if len(qualified)==1 else None)
    kind='unknown' if ref['identity'] is None or cur['identity'] is None else 'own' if ref['identity']==cur['identity'] else 'foreign'
    assert row['direct_kind']==kind
    fields=['events','reference_'+ref['status'],'current_'+cur['status'],'direct_'+kind]
    counts.update(fields)
    if label['known']:known.update(['conditional_known',*fields[1:]])
assert dict(counts)==r['totals'] and dict(known)==r['conditional_known_counts']
n=len(r['reference_conflicts']);m=len(r['raw_native_conflicts']);k=r['conditional_known_counts'];c=r['totals']
fact=(f'На 200 сохранённых fit20 событиях прямой однозначный якорь последнего шаблона есть у {c.get("reference_unique",0)}, сырого кандидата у {c.get("current_unique",0)}; оба конца известны у {c.get("direct_own",0)+c.get("direct_foreign",0)}: свои {c.get("direct_own",0)}, чужие {c.get("direct_foreign",0)}. '
      f'У {k.get("conditional_known",0)} прежних условно известных пар: прямых прошлых якорей {k.get("reference_unique",0)}, текущих {k.get("current_unique",0)}. '
      f'Подтверждённых расхождений последнего шаблона с прежней меткой цели — {n}; расхождений сырого кандидата и текущего состояния нового трека — {m}. Источник `runs/{OUT.name}/results.json`.')
interpretation=('Найдены прямые основания отдельно проверить загрязнение памяти в перечисленных событиях.' if n else 'На прямо размеченной части подмена последнего шаблона не обнаружена. Это не подтверждает чистоту неразмеченной части и не оправдывает замену памяти как уже установленное исправление.')
limitations='Соответствие определено рамочной разметкой, одним классом и прежним IoU≥0,5. Это не попиксельная принадлежность активности. Неаннотированные кадры не интерполировались. Прежние условные метки траекторий и результаты45пар сохранены; прямые метки не используются в рабочем решении. Качество полного трекера, новые пороги и польза ядра FlyVis не оценивались.'
decision=('Разбирать указанные расхождения памяти до выбора защищённого шаблона; не внедрять GT-зависимый допуск.' if n else 'Не выдавать защищённую память за подтверждённое лечение. Следующий механизм выбирать по различению экземпляров и отделению фона, сохраняя контроль без FlyVis; исходный полный трекер остаётся базой сравнения.')
summary=dict(passed=True,totals=c,conditional_known_counts=k,reference_conflicts=n,raw_native_conflicts=m,interpretation=interpretation,decision=decision,limitation=limitations)
save(OUT/'summary.json',summary)
(OUT/'РЕЗУЛЬТАТ.md').write_text('# Принадлежность последнего шаблона\n\n'+fact+'\n\n'+interpretation+'\n\n'+limitations+'\n\n'+decision+'\n\nИдея защищённой памяти уже была записана29.10.2026 в `research_identity_memory_solution_20261029/СВОДКА.md`; эта сверка проверяет её применимость, а не объявляет новым методом. STARK, раздел3.2 и официальный `stark_st.py`, строки42–54и84–92, помогли разделить начальный и обновляемый шаблоны. Наше управление памятью ещё не воспроизведено и не проверено.\n\nПроцент прироста сохраненияID не определён: это аудит входных свидетельств, не опыт полного трекера.\n',encoding='utf8')
path=ROOT/'experiments/project_history.json';cat=read(path);card=next(x for x in cat['experiments'] if x['id']==OUT.name);assert card['state']=='running'
card.update(state='infrastructure',done='2026-11-08',obtained=fact,interpretation=interpretation,cannot_claim=limitations,decision=decision)
card['acceptance'][0].update(met=True,evidence=f'runs/{OUT.name}/results.json')
card['evidence'] += [dict(kind='file',ref=f'runs/{OUT.name}/{name}',claim='Прямое свидетельство и его ограничения') for name in ['results.json','summary.json','РЕЗУЛЬТАТ.md']]
save(path,cat)
path=ROOT/'experiments/obsidian_experiment_topics.json';topics=read(path);topics['topics'][OUT.name]=topics['topics']['singleton_support_size_20261108'];save(path,topics)
path=ROOT/'experiments/system_problems.json';problems=read(path);problem=next(x for x in problems['problems'] if x['id']=='С2')
problem['cases'].append(dict(date='2026-11-08',component='Принадлежность последнего шаблона памяти',fact=fact,hypothesis='Загрязнение памяти остаётся версией с явно измеренным покрытием; альтернативой остаётся недостаточная различимость пространственного считывания.',discriminating_check='Использовать перечисленные прямые якоря и отдельно неизвестные кадры; не подставлять траекторную метку вместо разметки текущего кадра.',limitation=limitations,evidence=[f'runs/{OUT.name}/results.json',f'runs/{OUT.name}/summary.json']))
problem['evidence'].append(dict(source=f'runs/{OUT.name}/summary.json',sha256=sha(OUT/'summary.json'),claim='Проверено фактическое покрытие и принадлежность шаблонов'))
save(path,problems)
path=ROOT/'experiments/useful_literature.json';literature=read(path);assert not any(x['id']=='yan2021_stark_template_memory' for x in literature['sources'])
literature['sources'].append(dict(id='yan2021_stark_template_memory',title='Начальный и обновляемый шаблоны цели — Yan и соавторы',problem_group='Сохранение личности',method_group='Надёжность памяти',publication='Bin Yan, Houwen Peng, Jianlong Fu, Dong Wang, Huchuan Lu. Learning Spatio-Temporal Transformer for Visual Tracking. ICCV,2021,10448–10457.',doi='',url='https://arxiv.org/pdf/2103.17154',sections='Раздел3.2; официальный https://raw.githubusercontent.com/researchmm/Stark/main/lib/test/tracker/stark_st.py, строки42–54и84–92. CVF PDF403; arXiv и официальный код открыты.',established='В опубликованном однократном сопровождении заданной цели сохраняется начальный шаблон и по уверенности обновляется динамический. Код не заменяет начальный элемент при динамической записи.',problem='С2 — геометрически допустимый чужой объект может получить прежнийID.',application='Разделение ролей памяти использовано для постановки аудита: прежде чем менять память, проверить принадлежность последнего шаблона по прямой разметке. Аудит выполнен без запуска STARK и без изменения трекера.',limitation='Первый шаблон нашего автоматически рождаемого трека не задан истинной разметкой. Порог0,5 из STARK не переносился; устойчивый банк и нейросеть STARK не реализованы. Работа не доказывает, что наша ошибка вызвана памятью.',evidence=[f'runs/{OUT.name}/protocol.json',f'runs/{OUT.name}/summary.json'],related=[f'exp:{OUT.name}','sys:С2']))
save(path,literature)
path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)))
entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Проверены прямые якоря последнего шаблона и сырого кандидата.\n\n*Получено (`{OUT.name}`).* {fact}\n\n*Вывод.* {interpretation} {limitations}\n\n*Решение.* {decision} Идея защищённой памяти уже была в разборе29.10.2026; не объявляется новой.\n\n'
path.write_text(text.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
for path,digest in p['sources'].items():assert sha(ROOT/path)==digest,path
save(OUT/'independent_verification.json',dict(passed=True,records=200,status_and_identity_recount=True,source_hashes_stable=True,old_labels_and_scores_unchanged=True))
save(OUT/'execution_receipt.json',dict(checksums={name:sha(OUT/name) for name in ['audit.py','finish.py','protocol.json','results.json','summary.json','РЕЗУЛЬТАТ.md','independent_verification.json']}))
print(fact,interpretation)
