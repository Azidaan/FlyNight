from common import *
import subprocess, shutil
fate=read(NATIVE/'id_fate.json')
selected=[r for r in fate['records'] if r['arm']=='C34' and r['kind'] in ['own','foreign'] and (r['scored'] or r['reason']=='outside_observed_training_gap')]
assert len(selected)==29
old=read(NATIVE/'protocol.json');inputs={}
def bind(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p;inputs[p.relative_to(ROOT).as_posix()]=sha(p)
for n in ['protocol.json','id_fate.json','deployment.json','execution_protocol.json']:bind(NATIVE/n)
for v in sorted({r['video'] for r in selected}):
    e=old['videos'][v]
    for k in ['detections','alignment','video_path']:bind(e[k]);assert sha(ROOT/e[k])==e['sources'][e[k]]
    for suffix in ['.events.json.gz','.npz','.receipt.json']:bind(NATIVE/'rollouts/C34'/(v+suffix))
    bind(LEGACY/'runtime_inputs'/(v+'.json'));bind(f'data/bdd100k/labels/box_track_20/val/{v}.json')
for n in ['protocol.json','predictions.json','features.json','patches.npz']:bind(ROOT/'research_agreement_readout_20261112'/n)
ag=read(ROOT/'research_agreement_readout_20261112/protocol.json')
for r in ag['screen']:
    for key in r['keys']:
        for item in ag['snapshots'][key]['files']:bind(item['path']);assert inputs[item['path']]==ag['inputs'][item['path']]
for n in ['runs/full_template_native_20261109/native_runtime.py','tracking/spatial_identity_assignment.py','tracking/full_template_correspondence.py','tracking/hex_linear_interpolation.py','tracking/channel_identity_head.py','tracking/channel_agreement_correspondence.py','runs/z13_local_peak_20261021/geometry.json','.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py']:bind(n)
save('protocol.json',dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 authorization='User: run three stages in order, exact causes, no full tracker and no threshold changes.',
 stages=['Recompute direct identities and first recorded failure for all 29 events; trace 1102 frames252..264.',
 'Replay exact saved complete assignment matrices, identity control then change only the known proposal permission; fixed alpha0.2 and cost_limit0.8. Foreign means remove its bonus, not forbid the pair.',
 'Inspect original frames and frozen maps of five cases; independently identifiable same physical details required before correspondence intervention. Identity and matched nonsemantic displacement controls required; no score-driven selection. If detail correspondence cannot be established, retain explicit missing evidence rather than proxy ground truth.'],
 fixed=dict(head_threshold=0,alpha=old['alpha'],beta=old['beta'],match_thresh=old['native_config']['match_thresh'],new_maps=0,new_fits=0,tracker_calls=0),
 acceptance=['29 direct endpoint labels independently reproduced','Saved assignment exactly reproduced and alternative solver optimal objective agrees','Each altered permission effect recorded with collateral assignments and direct labels; no causal claim beyond one matrix','Physical correspondence intervention requires independent landmarks; absence recorded explicitly','All precise root causes established only when a distinguishing test supports them'],
 selected=selected,inputs=inputs,limitations='Known diagnostic cases, not independent performance evaluation. GT allowed only in explicit oracle diagnostic. No update of public trajectories or historical results.'))
before=OUT/'before';before.mkdir()
for n in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:shutil.copy2(ROOT/n,before/Path(n).name)
p=ROOT/'experiments/project_history.json';raw=p.read_bytes();d=json.loads(raw)
assert not any(c['id']==ID for c in d['experiments'])
d['experiments'].append(dict(id=ID,title='Локализация отказов личности — путь связи, назначение и физические детали',area='flyvis',state='running',started='2026-11-12',
 question='Где впервые возникает каждый отказ в 29 сохранённых событиях и достаточна ли правильная оценка для назначения?',
 hypothesis='Раздельная подстановка разрешения и проверка соответствий позволяют отделить доступность, считывание и назначение.',
 expected='Установить причины в пределах наблюдаемого; обозначить отсутствующие измерения без подмены гипотезой.',
 acceptance=[dict(criterion=s,met=None) for s in read(OUT/'protocol.json')['acceptance']],
 evidence=[dict(kind='file',ref='research_failure_localization_20261112/protocol.json',claim='Условия до проверки исходов')]))
assert p.read_bytes()==raw;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Три этапа локализации отказов (`'+ID+'`). Автор разрешил проверку 29 сохранённых событий, подстановку разрешений на полных матрицах и проверку физических соответствий. Протокол: research_failure_localization_20261112/protocol.json. Пороги неизменны, новые карты, обучение и полный трекер запрещены. Результатов ещё нет.')
print('REGISTERED',len(selected),len(inputs))
