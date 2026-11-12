"""Exploratory source/metadata audit. No neural execution or inferred hidden frames."""
from common import *
import ast
paths=['research_agreement_readout_20261112/protocol.json','runs/domain_matched_native_20261109/execution_protocol.json','runs/domain_matched_native_20261109/runtime_adapter.py','runs/full_template_native_20261109/native_runtime.py','tracking/z13_readout.py','scripts/z13_target_model.py','research_domain_matched_reader_20261109/compute.py']
hashes={n:sha(ROOT/n) for n in paths};execution=read(ROOT/paths[1]);bound={}
for n in paths[2:6]:assert execution['sources'][n]==hashes[n];bound[n]=True
source=(ROOT/paths[3]).read_text(encoding='utf8');tree=ast.parse(source)
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='get' for m in x.body));get=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='get')
step_calls=[x for x in ast.walk(get) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='step'];assert len(step_calls)==1
call=step_calls[0];assert len(call.args)==3 and not call.keywords
assert [ast.unparse(x) for x in call.args]==['state','x[:, j - 1]','x[:, j]']
loops=[n for n in ast.walk(get) if isinstance(n,ast.For) and call in list(ast.walk(n))];assert len(loops)==1 and ast.unparse(loops[0].iter)=='range(1, len(hx))'
ag=read(ROOT/paths[0]);rows=[]
for dataset in ['train','screen']:
    for r in ag[dataset]:
        endpoints=[]
        for key in r['keys']:
            s=ag['snapshots'][key]['sequence'];frames=[v[0] for v in s];deltas=[b-a for a,b in zip(frames,frames[1:])];assert all(d>=0 for d in deltas)
            endpoints.append(dict(key=key,frames=frames,frame_deltas=deltas,observed_transitions=len(frames)-1,video_frame_span=frames[-1]-frames[0],duplicate_transitions=sum(d==0 for d in deltas),skipped_video_frames=sum(max(0,d-1) for d in deltas),uniform_unit_intervals=all(d==1 for d in deltas)))
        ref,cur=endpoints;assert cur['frames'][:len(ref['frames'])]==ref['frames']
        rows.append(dict(dataset=dataset,video=r['video'],frame=r['frame'],old_id=r['old_id'],label=r['label'],endpoints=endpoints,reference_to_current_video_gap=cur['frames'][-1]-ref['frames'][-1],reference_to_current_update_calls=cur['observed_transitions']-ref['observed_transitions']))
for n,h in hashes.items():assert sha(ROOT/n)==h
save('time_path_observation.json',dict(created=now(),inspection='Exploratory: five sequences were inspected before writing this checker. No preregistered causal outcome claimed.',sources=hashes,deployed_source_matches=bound,source_call=ast.unparse(call),source_line=call.lineno,records=rows,counts={dataset:dict(pairs=sum(r['dataset']==dataset for r in rows),nonunit_current_sequences=sum(r['dataset']==dataset and not r['endpoints'][1]['uniform_unit_intervals'] for r in rows),mismatched_reference_to_current_count=sum(r['dataset']==dataset and r['reference_to_current_video_gap']!=r['reference_to_current_update_calls'] for r in rows)) for dataset in ['train','screen']},finding='Feature replay uses one fixed backbone.step per successive stored observation, without passing elapsed frame count. Histories contain skipped frames and padded duplicate frames. Initial fade-in is excluded from these transition counts.',limitation='Source-level and sequence-level fact, not measured causal contribution to a wrong identity decision. Duplicate warmup is explicit padding; do not call it an accidental extra observation. Missing video inputs are not reconstructed. Training and runtime both use observation-index time; equality of this convention does not establish physical-time fidelity.',new_core_calls=0))
msg='При разборе источника (`saved_shift_cross_probe_20261112`) установлено: воспроизведение активности делает один шаг на соседнюю пару сохранённых наблюдений без передачи разницы номеров кадров. У832 есть524→534; у1102 история251,251,251,252. Это подтверждено исходниками архивного запуска; причинный вклад в ошибки не измерен. Источник research_saved_shift_probe_20261112/time_path_observation.json.'
journal(msg)
path=ROOT/'experiments/system_problems.json';raw=path.read_bytes();d=json.loads(raw);problem=next(x for x in d['problems'] if x['id']=='С1');problem['cases'].append(dict(date='2026-11-12',component='Подготовка временной истории полного модуля FlyVis',fact=msg,hypothesis='Влияние времени по индексу наблюдений на ошибки232/832/574 не проверено; не считать найденной причиной.',discriminating_check='Сначала восстановить точные входы и штатные карты; затем отдельным ограниченным опытом изменить только временную семантику с явными контролями отсутствующих кадров и повторов.',limitation='Нет нового запуска ядра. Пропущенные изображения не известны из этой проверки; кадровые повторы являются явным дополнением короткой истории.',evidence=['research_saved_shift_probe_20261112/time_path_observation.json'],source_sha256=sha(OUT/'time_path_observation.json')))
assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('TIME_PATH_OBSERVATION',json.dumps(read(OUT/'time_path_observation.json')['counts']))
