"""One image-border geometry correction, with exact reuse off that domain."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import ast,copy,hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
PARENT=ROOT/'runs/background_patch_gate_20261108';WIDE=ROOT/'runs/decoder34_identity_gate_20261108';BAL=ROOT/'runs/channel_balance_attribution_20261109'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=900)
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from tracking.clipping_aware_correspondence import visible_edge_translation
from tracking.channel_balanced_correspondence import channel_balanced_similarity
from tracking.hex_background_filter import unit_patch
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import neighbor_aug,R1_TAUS
OTHER=[0,1,2,3]+list(range(12,34));SELECT=[0,5,11,16,17,22,28,33]
BASE=ROOT/'runs/bbox_fit_maps_20261108';FULL=ROOT/'runs/full_decoder_outputs_20261108'
NAMES=['FlyVis','center8','neighbor8','input8_temporal','C34','R134_temporal','R134_spatial']
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cast(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [cast(x) for x in v]
    return v
def write(name,v):
    dest=OUT/name;dest.parent.mkdir(exist_ok=True)
    with dest.open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def filename(e):return '_'.join(map(str,[e['video'],e['frame'],e.get('old_tid',e.get('old')),e.get('new_tid',e.get('new'))]))+'.json'
def verify(p):
    for path,digest in p['sources'].items():assert sha(ROOT/path)==digest,path
def helpers():
    source=BAL/'study.py';nodes=[n for n in ast.parse(source.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name in ['sampler','arrays']]
    ns=dict(np=np,ast=ast,LinearNDInterpolator=LinearNDInterpolator,PARENT=PARENT,BASE=BASE,FULL=FULL,OTHER=OTHER,SELECT=SELECT,neighbor_aug=neighbor_aug)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns);return ns['sampler'](),ns['arrays']
def register():
    assert read(OUT/'mathematical_checks.json')['passed'] and read(BAL/'summary.json')['local_gate'] is False
    events=read(PARENT/'protocol.json')['events'];videos=sorted({e['video'] for e in events})
    files=[Path(__file__),ROOT/'tracking/clipping_aware_correspondence.py',ROOT/'tests/test_clipping_aware_correspondence.py',OUT/'math_check.py',OUT/'mathematical_checks.json',PARENT/'protocol.json',BAL/'study.py',BAL/'protocol.json',BAL/'results.json',BAL/'summary.json',BAL/'attribution.json',ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json',ROOT/'.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py',ROOT/'.venv/Lib/site-packages/ultralytics/trackers/utils/stracks.py']
    for video in videos:files += [ROOT/f'runs/bbox_confirmation_inventory_20261108/native/{video}.npz',ROOT/f'runs/train_cache/{video}/detections/raw_video.npz']
    p=dict(registered=datetime.now(timezone(timedelta(hours=3))).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),deadline='2026-11-09T07:40:00+03:00',
        hypothesis='A template centered on the Kalman-state box can compare different visible parts when the image clips a detection. A shared unclipped raw-detection edge gives a better translation anchor in that axis.',
        intervention='Recover the actual last detection from the saved native output index, confidence subset and class. On a clipped axis only, replace the old prediction translation with displacement of the shared visible edge. Preserve other axes, old maps, raw current candidate,19 physical offsets and the common-support rule. Unsupported opposite-edge clipping preserves the old prediction.',
        evidence_trigger='Own88->144 has current raw x2=1280. Previous attribution samples a central/upper old strip against a left current strip. This motivates a geometry test, not a proof of the desired warp. Foreign73->106 has no image-border clipping.',
        arms='Global cosine with the new anchor versus ready global cosine; equal-channel cosine with the same new anchor versus the ready equal-channel cosine. Both applied to all seven fixed FlyVis/image representations. No change to channel weights, threshold, neural state or detector.',
        reuse='For events whose anchor is unchanged, copy ready score grids exactly. Compute new scores only where this geometry actually changes. Recover only raw reference-box metadata, no tracker replay/model/image calls. New support is reported, not hidden.',
        exactness='Native idx is local to the high/low confidence subset created by parse_bboxes; it is not a global raw-cache row. Recover via that subset and assert exact score/class equality and exact float32 state-box parity. Assert the independently recovered current observation equals the archived candidate box.',
        criteria=[dict(criterion='Clipping algebra, exact raw-observation provenance, baseline reuse off the changed domain, source hashes and identical signal/control geometry pass.',met=None),dict(criterion='The global-cosine anchor corrects own>foreign for both C8 and C34 on the requested pairs; equal-channel combination is reported separately, not chosen after evaluation.',met=None),dict(criterion='Report coverage and matched45-pair AUROC changes for both normalizations and all controls, including new undefined pairs. No tracker or independent ID-preservation claim from this diagnostic.',met=None)],
        strongest_alternative='Detector edges are noisy or occluded rather than physical object boundaries; perspective/shape changes violate the rigid-translation model. The image control may benefit equally or more. A failed test stops this anchor, without moving edges or fitting scale on the two examples.',
        literature=dict(observation_principle='OC-SORT, Cao et al., CVPR2023. Abstract at https://arxiv.org/abs/2203.14360 and official ocsort.py lines94-124 explicitly distinguish stored detector observations and filtered state. The original clipping-edge formula here is NOT claimed to be part of OC-SORT.',code='https://raw.githubusercontent.com/noahcao/OC_SORT/master/trackers/ocsort_tracker/ocsort.py',access='Official abstract and code opened. CVF403; arXiv PDF exceeded tool size. No claim of reading unavailable full-text sections.'),
        sources={f.relative_to(ROOT).as_posix():sha(f) for f in sorted(set(files))})
    write('protocol.json',p)
    path=ROOT/'experiments/project_history.json';cat=read(path)
    first=next(c for c in cat['experiments'] if c['id']==BAL.name);assert first['state']=='running';s=read(BAL/'summary.json')
    first.update(state='completed_negative',done='2026-11-09',obtained='Равная нормировка исправила порядок только для34карт:0,381588своей против0,326055чужой. Для8T4/T5:0,215584против0,333270, порядок неверен. На45условныхпарах AUROC C8:0,574519→0,600962; C34:0,593750→0,620192; интервалы включаютноль. Контроль временного входа34:0,685096. Источник `runs/'+BAL.name+'/summary.json`.',interpretation='Общий вес каналов влияет на оценки, но эта нормировка не исправляет задачу целиком. Максимум чужой34-карты лежит выше размеченной рамки; новая нормировка не меняет этот максимум по положению.',decision='Не принимать как общее исправление и не подбирать порог или маску каналов. Проверить конкретную геометрическую причину обрезки изображения отдельным опытом.',cannot_claim='Ошибка нейронного ядра или полезность полного трекера не установлены. Две пары диагностические;45пар имеют условные траекторные метки, семь исторических случаев не являются новым независимым тестом.')
    for a,met in zip(first['acceptance'],[True,s['local_gate'],s['broader_gate']]):a.update(met=bool(met),evidence=f'runs/{BAL.name}/summary.json')
    first['evidence'] += [dict(kind='file',ref=f'runs/{BAL.name}/{n}',claim='Первичный разбор и зафиксированная проверка') for n in ['attribution.json','summary.json','results.json','transfer.json']]
    cat['experiments'].append(dict(id=OUT.name,title='Обрезка краем изображения — привязка пространственного шаблона к видимому краю',area='flyvis',state='running',started='2026-11-09',question='Искажает ли центр усечённой рамки выбор сравниваемого участка карты?',hypothesis=p['hypothesis'],expected='Одна геометрическая привязка только на оси наблюдаемой обрезки; те же считыватели с FlyVis и без него.',acceptance=p['criteria'],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Условия до новых оценок')]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)))
    entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Нормировка не прошла полный критерий; зарегистрирована проверка видимого края.\n\n*Получено (`{BAL.name}`).* {first["obtained"]}\n\n*Вывод.* {first["interpretation"]}\n\n*Постановка (`{OUT.name}`).* Математика обрезки и четыре синтетическихтеста проверены до восстановления сырых наблюдений. Восстановить истинную прошлую детекцию по индексу её confidence-подмножества; только обрезанная ось получает перенос видимого края. Остальные случаи читают готовые оценки. Обе нормировки и семь прежних представлений с одинаковой геометрией. Это геометрическая правка чтения, не свойство ядра FlyVis. Источник `runs/{OUT.name}/protocol.json`.\n\n'
    path.write_text(text.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8');print('Registered clipping-edge correction',flush=True)
def raw_observation(native,raw,tid,frame,cfg):
    rows=native[(native[:,0]==frame)&(native[:,5]==tid)];assert len(rows)==1,(frame,tid,len(rows));row=rows[0]
    ids=np.flatnonzero(raw['frame_idx']==frame);conf=raw['conf'][ids]
    mask=conf>=cfg['track_high_thresh'] if row[6]>=cfg['track_high_thresh'] else (conf>cfg['track_low_thresh'])&(conf<cfg['track_high_thresh'])
    chosen=ids[mask];index=int(row[8]);assert row[8]==index and 0<=index<len(chosen)
    index=int(chosen[index]);assert np.float32(raw['conf'][index])==row[6] and int(raw['cls'][index])==int(row[7])
    return dict(raw_id=index,box=raw['xyxy'][index].astype(float),state_box=row[1:5].astype(float),confidence=float(row[6]),cls=int(row[7]),frame=frame,subset_index=int(row[8]))
def new_geometry(e,g,mesh,translation):
    h=e['row_data']['history'][-1];c0=np.asarray(h['crop']);ct=np.asarray(e['row_data']['current_crop']);s0=(c0[2:]-c0[:2])/g['eye_size'];st=(ct[2:]-ct[:2])/g['eye_size'];points=np.asarray(g['points']);oldbox=np.asarray(h['box']);box=np.asarray(e['box']);off=np.asarray(e['geometry']['offsets'])
    x=points*s0+c0[:2];trial=x[:,None]+translation+off;query=(trial-ct[:2])/st
    keep=((x>=oldbox[:2])&(x<=oldbox[2:])).all(1)&((trial>=box[:2])&(trial<=box[2:])).all((1,2));keep&=(mesh.find_simplex(query)>=0).all(1);idx=np.flatnonzero(keep)
    return dict(cells=idx.tolist(),query=query[idx].tolist(),prediction=translation.tolist(),offsets=off.tolist())
def run():
    if (OUT/'results.json').exists():raise SystemExit('Ready anchor scores exist; no repetition.')
    p=read(OUT/'protocol.json');verify(p);sample,arrays=helpers();g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');mesh=Delaunay(g['points']);nb=HexGeometry(np.array(g['points']),g['eye_size']).neighbors;cfg=read(ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json')['native_config']
    grouped={}
    for e in read(PARENT/'protocol.json')['events']:grouped.setdefault(e['video'],[]).append(e)
    files={};changed=0;reused=0;traces=[]
    for video,events in grouped.items():
        with np.load(ROOT/f'runs/bbox_confirmation_inventory_20261108/native/{video}.npz',allow_pickle=False) as z:native=z['outputs']
        with np.load(ROOT/f'runs/train_cache/{video}/detections/raw_video.npz',allow_pickle=False) as z:raw={k:z[k] for k in ['xyxy','conf','cls','frame_idx']}
        for e in events:
            old=raw_observation(native,raw,e['old_tid'],e['row_data']['history'][-1]['frame'],cfg);current=raw_observation(native,raw,e['new_tid'],e['frame'],cfg)
            assert np.array_equal(np.asarray(e['row_data']['history'][-1]['box'],np.float32),np.asarray(old['state_box'],np.float32))
            assert np.array_equal(current['box'],e['box']),(video,e['frame'],e['new_tid'],'Current raw box provenance')
            anchor=visible_edge_translation(old['box'],current['box'],e['geometry']['prediction'],[1280,720]);prior8=read(PARENT/'events'/filename(e));prior34=read(WIDE/'events'/filename(e));balanced=read(BAL/'events'/filename(e));variants={};geom=e['geometry']
            if anchor['changed']:
                changed+=1;geom=new_geometry(e,g,mesh,anchor['translation']);idx=np.asarray(geom['cells'],dtype=int)
                if len(idx)>=3:
                    a=arrays(e['reference_key'],nb);b=arrays(e['current_key'],nb)
                    for n in NAMES:
                        template=a[n][idx];z0=unit_patch(template);patches=sample(mesh,np.asarray(geom['query']),b[n]).transpose(1,0,2)
                        global_scores=[float(z0@z) if z0 is not None and (z:=unit_patch(v)) is not None else None for v in patches]
                        equal=[channel_balanced_similarity(template,v) for v in patches]
                        variants[n]=dict(global_scores=global_scores,balanced_scores=equal)
                else:
                    variants={n:dict(global_scores=[None]*19,balanced_scores=[None]*19) for n in NAMES}
            else:
                reused+=1
                for n in NAMES:
                    v=(prior8 if n in prior8['variants'] else prior34)['variants'].get(n)
                    variants[n]=dict(global_scores=v['correlation_scores'] if v else [None]*19,balanced_scores=balanced['variants'][n]['scores'] if n in balanced['variants'] else [None]*19)
            for n,v in variants.items():
                for field in ['global','balanced']:
                    values=v[field+'_scores'];valid=[i for i,x in enumerate(values) if x is not None];winner=max(valid,key=lambda i:values[i]) if valid else None;v[field+'_max']=values[winner] if winner is not None else None;v[field+'_winner']=winner
                oldv=(prior8 if n in prior8['variants'] else prior34)['variants'].get(n)
                v.update(baseline_global=oldv['correlation_max'] if oldv else None,baseline_balanced=balanced['variants'][n]['maximum'] if n in balanced['variants'] else None)
            record=dict(video=video,frame=e['frame'],old=e['old_tid'],new=e['new_tid'],kind=prior8['kind'],anchor=anchor,reference_observation=old,current_observation=current,geometry=geom,baseline_cells=e['geometry']['cells'],variants=variants,reused=not anchor['changed'])
            write(Path('events')/filename(e),record);files[filename(e)]=sha(OUT/'events'/filename(e))
            if (video,e['old_tid'],e['new_tid']) in [('00982f76-83a093ef',88,144),('00d7268f-fd4487be',73,106)]:traces.append(record)
        print('Observed-box provenance',video,'changed',changed,'reused',reused,flush=True)
    verify(p);write('results.json',dict(passed=True,events=200,changed=changed,reused=reused,files=files));write('pair_traces.json',dict(records=traces));print('Anchor comparison completed',changed,reused,flush=True)
{'register':register,'run':run}[sys.argv[1]]()
