"""Preregister one domain-matched head; inventory exact saved inputs first."""
import os,sys,json,hashlib,subprocess,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
DATA=ROOT/'research_working_domain_identity_20261109';OLD=ROOT/'runs/identity_channel_head_20261109'
MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,23,54,tzinfo=MSK)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=min(900,(STOP-datetime.now(MSK)).total_seconds()))
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
def atomic(path,data,before):
    temp=path.with_suffix(path.suffix+'.matched.pending');assert not temp.exists();temp.write_bytes(data)
    for _ in range(30):
        assert path.read_bytes()==before,path
        try:temp.replace(path);return
        except PermissionError:time.sleep(.1)
    raise PermissionError(str(path))
def sequence_key(video,seq):return hashlib.sha256(json.dumps([video,seq],separators=(',',':')).encode()).hexdigest()[:32]
mode=sys.argv[1]
if mode=='register':
    data=read(DATA/'results.json');assert data['passed'] and data['readiness']
    edges=[e for e in data['records'] if e['runtime_supported'] and e['kind']!='unknown']
    assert len(edges)==26 and sum(e['kind']=='own' for e in edges)==20
    assert read(ROOT/'research_coordinate_reader_20261109/comparison.json')['gate'] is False
    old=read(OLD/'protocol.json')
    videos=sorted({e['video'] for e in edges});val=set(read(ROOT/'runs/full_template_native_20261109/protocol.json')['videos'])
    assert len(videos)==20 and not set(videos)&val
    files=[Path(__file__),DATA/'results.json',DATA/'protocol.json',OLD/'protocol.json',OLD/'maps_complete.json',
        ROOT/'tracking/channel_identity_head.py',ROOT/'tracking/full_template_correspondence.py',ROOT/'tracking/observed_spatial_reader.py',
        ROOT/'tracking/hex_linear_interpolation.py',ROOT/'tracking/channel_balanced_correspondence.py',ROOT/'tracking/z13_readout.py',
        ROOT/'scripts/z13_target_model.py',ROOT/'runs/z13_local_peak_20261021/geometry.json',
        ROOT/'runs/z13_target_center_dev_20261018/C/normalization.json',ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json',
        ROOT/'runs/flyvis/results/flow/0000/000/best_chkpt',ROOT/'runs/flyvis/results/flow/0000/000/_meta.yaml']
    for video in videos:files += [ROOT/f'runs/z13_data/{video}.npz',DATA/'videos'/(video+'.json')]
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    # Archived model/input passports must still describe the actual implementation.
    witness=['tracking/channel_identity_head.py','tracking/hex_linear_interpolation.py','scripts/z13_target_model.py',
        'runs/flyvis/results/flow/0000/000/best_chkpt','runs/z13_target_center_dev_20261018/C/normalization.json','runs/z13_target_center_dev_20261018/R1/normalization.json']
    for name in witness:assert sources[name]==old['sources'][name],name
    for video in videos:
        name=f'runs/z13_data/{video}.npz';assert sources[name]==old['sources'].get(name,read(ROOT/'runs/historical_direct_query_inventory_20261109/protocol.json')['sources'][name])
    save('protocol.json',dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        authorization='research_identity_synthesis_20261109/authorization.json',save_stop=STOP.isoformat(),sources=sources,edges=edges,videos=videos,
        action='Change the head training population to the directly labeled native operating domain. Keep the old full candidate-relative spatial reader and the exact previous fixed fitting procedure. Do NOT combine with the failed geometric-coordinate intervention.',
        data='Exactly26directlyknownruntime-supportededges,20own/6foreign,20trainingvideos;137unknownruntimeedges excluded from supervised loss, not relabeled. All20nativevalvideos disjoint. These training videos and the previously viewed val pilot are development, not an independent final test.',
        input_gate='Find each exact causal4warm-plus-search sequence in saved unperturbed z13_data episodes by oldtrackID,frame and all crop coordinates. Never nearest-frame/crop substitution. Record absent inputs; reuse only exact sequence-key maps with matching checkpoint/input/feature/normalization passports. No fr_hex decompression or core inference in readiness.',
        reader='Original relative_template_plan with full reference support,all19physicalshiftsineye,perchannelPearson and old equal-channel shift choice. Old candidate-eroded support is used ONLY in unchanged7geometryfeatures/admission. Minimum3unchanged. No candidate/background/channel/threshold selection.',
        models=['G7','C34','R134_spatial','R134_temporal'],
        features='Same7causalgeometryfeatures standardized on training videos only. Signal heads add the same34channelcorrelations without standardization; exact previous constant-channel behavior and G7fallback. Same input histories,coordinates,cells and folds for allthree signals.',
        fit='Class-balanced mean logistic loss + ||w||^2/2, fixedlambda1, free intercept, exact previous deterministicNewton, thresholdz>0. No hyperparameter/threshold/representation search. Frozen FlyVis checkpoint, no core training.',
        validation='Leave one full video out,20fixedfolds. Both classes required in every training fold. Undefined feature or class loss uses that separately fitted G7fold. A final fit on all26only if admitted to actual native pilot.',
        primary='C34versusG7 on identical heldout-video edges. Native-pilot admission: own accepts>=G7 and foreign accepts<G7, hence balanced accuracy>G7. Ordinary controls reported identically. This point gate is not confirmed extra-core benefit or retainedID.',
        uncertainty='Paired bootstrap of20videoclusters,4000draws,seed20261109; descriptive95%intervals for C34vsG7 and bothordinarysignals. ExtraFlyVisbenefit requires later native ID/HOTA with matchedcontrols and an independent final test.',
        stop='If primary admission fails, preserve the negative result and do not change threshold,lambda,reader,channels or sample selection after scores. Do not run a full native rollout merely to stay busy.',
        diagnostic_budget='At most3for this action: exact-input/math readiness, fixed matched heldout-video comparison, one independent implementation audit. Then a bounded actual-native pilot if admitted or stop this version.',
        literature='Existing verified Lin,Weng,Keerthi2008regularizedlogisticobjective supports the optimization only; no source or training count guarantees object identity or conditional-domain generalization.'))
    identity='domain_matched_identity_head_20261109'
    path=ROOT/'experiments/project_history.json';before=path.read_bytes();c=json.loads(before);assert all(x['id']!=identity for x in c['experiments'])
    c['experiments'].append(dict(id=identity,title='Принадлежность рамки — обучение в рабочем геометрическом допуске',area='flyvis',state='planned',started='2026-11-09',
        question='Поможет ли обучение на реально предъявляемых своих и чужих связях при неизменном чтении карт?',
        hypothesis='Прежние чужие примеры были вне рабочего допуска; соответствующее области обучение может перенести признак на допустимых соседей. Сильнейшая альтернатива — сами пространственные карты и обычные входы не содержат достаточного различия или26примеров недостаточно.',
        expected='Точные сохранённые истории и карты, затем одна20видеомежвидеоваяпроверка четырёх одинаковых голов. Ядро, координатный считыватель и пороги прежние.',
        acceptance=[dict(criterion='Точные причинные последовательности и паспорта, математические/сохранённые свидетели, отсутствие утечки видео и одинаковые сигнальные условия.',met=None),dict(criterion='C34принимает не меньше своих и строго меньше чужих, чем G7, при фиксированном нулевом пороге; только тогда реальный пилот. Это критерий допуска, не решениеID.',met=None)],
        obtained='',interpretation='',decision='',cannot_claim='Малая разработческая выборка; нет гарантии общей точности и нет независимого финального теста. Полезность ядра отделяется от геометрии и обычного34канальноговхода.',
        evidence=[dict(kind='file',ref='research_domain_matched_reader_20261109/protocol.json',claim='Одна правка и критерии до новых входов и оценок')],tags=['fly-problem-P1']))
    atomic(path,(json.dumps(c,ensure_ascii=False,indent=2)+'\n').encode(),before)
    path=ROOT/'experiments/obsidian_experiment_topics.json';before=path.read_bytes();topics=json.loads(before)
    topics['topics'][identity]=['Сохранение личности','Выбор продолжения среди рамок','Рабочая область обучения']
    atomic(path,(json.dumps(topics,ensure_ascii=False,indent=2)+'\n').encode(),before)
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
    entry='**'+datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')+'** — Зарегистрировано обучение считывателя внутри рабочего допуска\n\n*Постановка (`'+identity+'`).* Только26новыхпрямыхсвязей:20своих/6чужих,20обучающихвидео. Меняется обучающий пул, прежнее полное относительное чтение и34канала/19сдвигов остаются. Геометрия и оба обычных входа обучаются одинаково; штраф1,порог0,исключениецелоговидео. Сначала точные сохранённые последовательности и паспорта, затем фиксированное сравнение. Критерий допускаC34:не меньше своих и строго меньше чужих, чем G7. Иначе полный пилот не запускать и параметры не подбирать. Источник `research_domain_matched_reader_20261109/protocol.json`. Раздел7плана не меняется.\n\n'
    atomic(path,before.replace(marker,marker+nl+entry.replace('\n',nl.decode()).encode(),1),before)
    print('REGISTERED_DOMAIN_MATCHED_HEAD',len(edges),flush=True);raise SystemExit(0)
assert mode=='readiness';p=read(OUT/'protocol.json');verify(p)
import numpy as np,zipfile
old=read(OLD/'protocol.json');records=[];requests={};sources={}
for video in p['videos']:
    path=ROOT/f'runs/z13_data/{video}.npz'
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k] for k in ['ep_tid','ep_pert','ep_offset','fr_frame','fr_crop']}
    for edge in [e for e in p['edges'] if e['video']==video]:
        seq=edge['sequence'];n=len(seq);matches=[]
        for episode in np.flatnonzero((data['ep_tid']==edge['old_id'])&(~data['ep_pert'])):
            offset=int(data['ep_offset'][episode]);end=int(data['ep_offset'][episode+1])
            if end-offset<n:continue
            actual=[[int(data['fr_frame'][i]),data['fr_crop'][i].tolist()] for i in range(offset,offset+n)]
            if actual==seq:matches.append(dict(episode=int(episode),offset=offset))
        record=dict(video=video,frame=edge['frame'],old_id=edge['old_id'],raw_id=edge['raw_id'],exact_episode_matches=matches,
                    input_ready=bool(matches),label=edge['kind'],requests=[])
        if matches:
            offset=matches[0]['offset']
            for length in [4,n]:
                prefix=seq[:length];key=sequence_key(video,prefix);slots=set();files=[]
                if key in old['requests']:
                    assert old['requests'][key]['video']==video and old['requests'][key]['sequence']==prefix
                    for slot,item in old['requests'][key]['ready'].items():
                        slotpath=ROOT/item['path'];sources[item['path']]=sha(slotpath);files.append(dict(path=item['path'],array=item['array'],slot=slot));slots.add(slot)
                    own=OLD/'maps'/(key+'.npz')
                    if own.exists():
                        meta=read(own.with_suffix('.json'));assert meta['sha256']==sha(own)
                        with zipfile.ZipFile(own) as z:names=[Path(k).stem for k in z.namelist()]
                        sources[own.relative_to(ROOT).as_posix()]=sha(own);sources[own.with_suffix('.json').relative_to(ROOT).as_posix()]=sha(own.with_suffix('.json'))
                        for slot in names:slots.add(slot);files.append(dict(path=own.relative_to(ROOT).as_posix(),array=slot,slot=slot))
                core='C34' in slots or {'C8','C_OTHER'}<=slots
                requests[key]=dict(video=video,sequence=prefix,data=path.relative_to(ROOT).as_posix(),
                    indices=list(range(offset,offset+length)),ready_slots=sorted(slots),ready_files=files,
                    core34_ready=core,ordinary_ready='R1' in slots)
                record['requests'].append(key)
        records.append(record)
    print('INPUT_INVENTORY',video,len(records),flush=True)
verify(p)
ready=all(r['input_ready'] for r in records);assert len(records)==26
save('input_inventory.json',dict(passed=True,exact_sequences_ready=ready,records=records,requests=requests,
    sources=sources,core34_snapshots_ready=sum(v['core34_ready'] for v in requests.values()),
    ordinary_snapshots_ready=sum(v['ordinary_ready'] for v in requests.values()),total_snapshots=len(requests),
    no_fr_hex_decompression_or_network_calls=True,
    next='Restore only missing maps from these exact saved receptor arrays; preserve old maps and recipes. Missing sequence requires an explicit separate recovery passport, never approximate a crop/frame.'))
print('INPUT_READY',ready,len(requests),'core-ready',sum(v['core34_ready'] for v in requests.values()),flush=True)
