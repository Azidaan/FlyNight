"""Generate the sampling-only pilot from the verified fixed-budget worker."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'scripts/z13_localization_labels_20261023.py').read_text(encoding='utf-8')
source=source.replace('z13_localization_labels_20261023','z13_semantic_sampling_20261023').replace('z13_localization_labels_v1','z13_semantic_sampling_v1')
start=source.index('def prepare():');end=source.index('\ndef train_steps(',start)
source=source[:start]+'''def prepare():
    assert not RUN.exists()
    from scripts.z13_localization_labels_20261023 import RUN as BASE
    base=read(BASE/'protocol.json');verify_sources(base)
    names=['tracking/z13_semantic_sampling.py','scripts/z13_semantic_sampling_20261023.py','scripts/z13_semantic_audit_20261023.py','scripts/z13_data.py','tests/test_z13_semantic_sampling.py','scripts/z13_semantic_sampling_console.ps1']
    old=read(OLD/'protocol.json')
    for video in base['train']:
        names.extend([f'runs/z13_data/{video}.npz',f'runs/train_cache/{video}/alignment.json',f'data/bdd100k/labels/box_track_20/train/{video}.json'])
    deadline=now().replace(hour=7,minute=40,second=0,microsecond=0);assert now()<deadline
    p=dict(id=IDENT,created=now().isoformat(),authorization='Author: autonomous necessary experiments until07:40MSK23.10.2026.',device='cpu',budget_seconds=3600,save_at=(deadline-timedelta(minutes=5)).isoformat(),hard_stop=(deadline-timedelta(seconds=30)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),arms=list(ARMS),seeds=[0,1,2],train=base['train'],evaluate=base['evaluate'],
      action='Only training row sampling changes: half uniform, half semantic-hard rows. Original utility labels, class weights, architecture, random initialization,1150steps and145950row presentations remain fixed. Same sampled row stream for all arms. Old18models reused.',
      hard_definition='Training104 only. Candidate center uniquely covered by one annotated box of another identity but the same category as target; no overlapping boxes, unknown kind excluded, old useful positives excluded. Union of hard rows across D/C/R1 is the COMMON sampling pool. Current GT is used only for training sampling, never evaluation inference. No new negative labels or losses.',
      metadata='Video/local episode identity and frame from existing episode NPZ; original annotations via saved alignment. Every known-present target center must agree within0.001px. All new sources hashed. No interpolation or fabricated distractor IDs.',
      gates='Nonempty D hard rows and common pool, correct identity/frame binding, unchanged old labels; all18old models replayed;32old steps reproduced; conservative18model estimate fits3600s.',
      primary='Dnew activity-Dold activity >=0.02 AND paired-video95%lower>0; Dnew beats Dold geometry and R1new activity with lower95>0. All required; C cannot rescue.',
      secondary='All26videos,three individual seeds, geometry ablations, top1/Kalman, fixed1/2/5percent confidence quotas with unknown and strict worse separately. No efficacy-driven parameter or seed selection.',
      counterargument='Box containment is only a conservative semantic proxy: a peak can represent overlapping texture rather than an object. Rare hard rows can be oversampled excessively and displace useful easy rows. Fixed50percent is one prespecified test, not a tuned optimum. This changes data frequency, not representation capacity.',
      paper='https://arxiv.org/html/1808.06048v1 section3.2, semantic same-category negatives. Adaptation of sampling principle, not DaSiamRPN reproduction or transferred magnitude.',
      sha256={**base['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p)
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: обучение выборщика на похожих размеченных соседях',area='flyvis',state='planned',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','engineering-future','DaSiamRPN'],question='Поможет ли увеличение частоты похожих соседей при неизменном считывателе и бюджете?',hypothesis='Обычные отрицательные примеры могут недостаточно обучать различению экземпляров одного класса; проверяется только частота таких примеров.',expected='D/C/R1,18моделей,3сида,104train и26оценки;3600с CPU+20с завершения, до07:40. Один общий набор трудных строк; только выбор строк меняется.',acceptance=[dict(criterion='Мишень непустая, исходная разметка связана с кадрами/идентичностями; старые метки,18контролей,32шага и бюджет воспроизведены,хеши проверены.',met=None),dict(criterion='Dnew-Dold activity >=2п.п.,нижняя95%повидеоДИ>0.',met=None),dict(criterion='Dnew лучше Dold geometry и R1new: обе нижние95%ДИ>0.',met=None)],summary='Протокол до исходов: половина пакета из общего набора строк с похожими соседями.',done='',obtained='',interpretation='',decision='',cannot_claim='Поисковый train26, не HOTA и не безопасный отказ. Нахождение пика внутри рамки не доказывает, что его активность принадлежит соседу. Не полная DaSiamRPN; эффект частоты не доказывает новизну.',metrics=[],commits=[],files=names[:4],evidence=[dict(kind='file',ref='runs/z13_semantic_sampling_20261023/protocol.json',claim='Критерии, единый бюджет, мишень и хеши до обучения')]))
    write(cp,cat)
    journal('постановка проверки похожих отрицательных примеров','*Постановка.* По DaSiamRPN §3.2 меняется только частота строк обучения: половина пакета из общего для D/C/R1 набора с пиком внутри единственной рамки другого объекта того же класса. Неизвестные и прежние положительные метки исключены из мишени; метки вообще не меняются. Сначала исходная разметка/кадры/идентичности и непустой набор, затем18равнобюджетных моделей. Протокол `runs/z13_semantic_sampling_20261023/protocol.json`. Критерии до результатов: D против старого чтения >=2п.п. с положительным ДИ, также превосходство над картой D и входом глаза R1. CPU3600с, своё дерево до07:40. Нет GPU/ядра/dev/HOTA/75.')

''' +source[end:]
source=source.replace('def train_steps(torch, model, tensors, nref, seed, weight, maxsteps=None, progress=None):','def train_steps(torch, model, tensors, nref, seed, weight, maxsteps=None, progress=None, hard=None):')
source=source.replace('    for step, indices in enumerate(matched_batches(len(tensors[0]),nref,50,128,seed)):',"    from tracking.z13_semantic_sampling import semantic_batches\n    batches=matched_batches(len(tensors[0]),nref,50,128,seed) if hard is None else semantic_batches(len(tensors[0]),nref,seed,hard)\n    for step, indices in enumerate(batches):")
source=source.replace('    from tracking.z13_localization_labels import localization_targets','    from scripts.z13_semantic_audit_20261023 import metadata,audit as semantic_audit')
source=source.replace('    arrays={};audit={};replay={};manifest={};result={};seedpv={}',"    base=read(OLD/'protocol.json');m1,h1,c1=metadata(base['fit'],labels20);m2,h2,c2=metadata(base['evaluate'],extra);meta=m1+m2\n    arrays={};audit={};replay={};manifest={**h1,**h2};result={};seedpv={};hard_union=np.zeros(len(labels),bool)")
a=source.index('        newy=localization_targets');b=source.index('        arrays[arm]=',a)
source=source[:a]+'''        newy=ta[4];negative,counts=semantic_audit(labels,ta[5],ta[3],meta)
        assert not np.any(negative & (newy>0));hard_union|=negative.any(1)
        audit[arm]=counts
''' +source[b:]
source=source.replace("write(RUN/'label_audit.json',audit)","write(RUN/'semantic_audit.json',dict(arms=audit,gt_centers_checked=c1+c2,common_hard_rows=int(hard_union[ids].sum()),common_indices=np.flatnonzero(hard_union[ids]).tolist()))")
source=source.replace("    assert audit['D']['changed_candidates']>0, 'No target for intervention'","    assert audit['D']['hard_rows']>0, 'No semantic target for intervention'\n    hard=np.flatnonzero(hard_union[ids]);assert len(hard)>0")
source=source.replace("train_steps(torch,model,tensors,nref,seed,ref['pos_weight'],progress=progress)","train_steps(torch,model,tensors,nref,seed,ref['pos_weight'],progress=progress,hard=hard)")
source=source.replace('Обучение новой метки при прежнем бюджете','Обучение на похожих соседях при прежнем бюджете').replace('Проверка кэшей, меток и прежних моделей','Проверка похожих соседей и прежних моделей')
source=source.replace('Only training labels changed.','Only training row sampling changed; common semantic-hard pool, unchanged original labels.')
source=source.replace("os.environ['CUDA_VISIBLE_DEVICES']='-1'","os.environ['CUDA_VISIBLE_DEVICES']=''")
source=source.replace('"""Bounded CPU label-only intervention. Frozen candidate caches and core."""','"""Bounded sampling-only intervention. Original labels, frozen caches/core."""')
target=ROOT/'scripts/z13_semantic_sampling_20261023.py';assert not target.exists();target.write_text(source,encoding='utf-8')
console=(ROOT/'scripts/z13_learned_association_console.ps1').read_text(encoding='utf-8-sig').replace('z13_learned_association_20261023','z13_semantic_sampling_20261023').replace('обучаемые соответствия кандидатов','похожие отрицательные примеры').replace('соответствия кандидатов и пустая пара','похожие соседи при обучении')
(ROOT/'scripts/z13_semantic_sampling_console.ps1').write_text(console,encoding='utf-8-sig')

# The result recorder has its own independent row/model/control verification.
rec=(ROOT/'scripts/z13_record_localization_labels_20261023.py').read_text(encoding='utf-8')
rec=rec.replace('z13_localization_labels_20261023','z13_semantic_sampling_20261023')
rec=rec.replace(". label_audit.json → '+json.dumps({a:{k:v for k,v in read(RUN/'label_audit.json')[a].items() if k!='per_video'} for a in ARMS},ensure_ascii=False)",". semantic_audit.json → '+json.dumps({a:{k:v for k,v in read(RUN/'semantic_audit.json')['arms'][a].items() if k!='per_video'} for a in ARMS},ensure_ascii=False)")
rec=rec.replace("    head=subprocess.check_output", "    assert read(RUN/'semantic_audit.json')==read(RUN/'preflight_target.json')\n    assert all(sha(ROOT/n)==h for n,h in read(RUN/'cache_manifest.json').items())\n    head=subprocess.check_output")
rec='\n'.join(line for line in rec.split('\n') if not line.strip().startswith("card['expected']+="))
rec=rec.replace('новой метки','новой частоты примеров').replace('Новая метка','Новая частота примеров').replace('замена обучающей метки','замена частоты примеров').replace('смены обучающей метки','изменения частоты похожих соседей')
rec=rec.replace('удаление сравнения с Калманом решает выбор','увеличение частоты похожих соседей решает выбор')
rec=rec.replace('Следующее разрешённое действие — обучаемое межкадровое соответствие кандидатов; сначала проверить доступные пары и сопоставимый контроль, не повторять подбор этой метки.','Не подбирать долю трудных строк по train26. Дальнейшее действие требует иной проверяемой мишени; эта реализация не доказывает непригодность семантических отрицательных примеров вообще.')
rec=rec.replace('новая метка точности, прежняя оценка полезности','общая для D/C/R1 половина строк с похожими соседями; прежние метки полезности')
rec=rec.replace('Новая метка выбора:','Похожие соседи при обучении:').replace('label_audit.json','semantic_audit.json')
rec=rec.replace('Record completed label-only pilot','Record completed semantic sampling pilot')
(ROOT/'scripts/z13_record_semantic_sampling_20261023.py').write_text(rec,encoding='utf-8')
