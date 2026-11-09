"""Direct-label availability for a causal learned channel reader; no features."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;DATA=ROOT/'runs/historical_direct_query_inventory_20261109';MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,23,54,tzinfo=MSK)
if not (OUT/'status.json').exists():(OUT/'status.json').write_text(json.dumps(dict(stage='starting',updated=datetime.now(MSK).isoformat())),encoding='utf8')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(STOP-datetime.now(MSK)).total_seconds();assert remaining>0;guard_entrypoint(timeout=min(900,remaining))
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def cast(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [cast(x) for x in v]
    return v
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def status(stage,**kw):(OUT/'status.json').write_text(json.dumps(dict(stage=stage,updated=datetime.now(MSK).isoformat(),**kw)),encoding='utf8')
def verify(p):
    for f,h in p['sources'].items():assert sha(ROOT/f)==h,f
def journal(title,body):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=p.read_text(encoding='utf8');p.write_text(s.replace('## Журнал изменений\n',f'## Журнал изменений\n\n**{datetime.now(MSK):%d.%m.%Y, %H:%M} MSK** — {title}.\n\n{body}\n\n',1),encoding='utf8')
def register():
    source=read(DATA/'protocol.json');exclude=set();files=[Path(__file__),DATA/'protocol.json',DATA/'results.json',ROOT/'research_identity_synthesis_20261109/authorization.json',ROOT/'research_identity_synthesis_20261109/math_checks.json',ROOT/'tracking/observed_spatial_reader.py',ROOT/'tracking/hex_linear_interpolation.py',ROOT/'runs/z13_local_peak_20261021/geometry.json']
    for name in ['historical_two_candidate_comparison_20261109','relative_reader_transfer_20261109']:
        path=ROOT/'runs'/name/'protocol.json';files.append(path);exclude.update(c['video'] for c in read(path)['cases'])
    assert len(exclude)==15;videos={k:v for k,v in source['videos'].items() if k not in exclude};assert len(videos)==69
    for i,(video,info) in enumerate(videos.items()):
        for key in ['data','raw','native','temporal','annotation','alignment','context']:
            path=ROOT/info[key];assert sha(path)==source['sources'][info[key]],info[key];files.append(path)
        if i%10==0:status('fingerprints',videos_done=i+1,total=69);print('FINGERPRINT',i+1,69,flush=True)
    p=dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),authorization='research_identity_synthesis_20261109/authorization.json',save_stop=STOP.isoformat(),hard_stop='2026-11-09T23:59:00+03:00',action='Proposed replacement of fixed equal mean of channel-wise SPATIAL correlations by one regularized learned identity score, with the same learned no-FlyVis input controls. This gate decides whether a causal directly labeled training population exists; it does not fit the head.',diagnostic_number=1,maximum_diagnostics_for_action=3,videos=videos,excluded_development_videos=sorted(exclude),population='Reuse ALL stored direct-reference candidate records from the previous84-video metadata inventory, not only exactly one-own/one-foreign clean pairs. Exclude all15 videos used to inspect the new relative reader. Each known candidate is an isolated input; unknown GT stays unknown.',causal_reference='Separate STRICT_LAST (reference_index==ep_offset+3, actual last observed warm frame directly annotated) from EARLIER_LABELLED (earlier actual annotation selected offline). Never pool the latter into the deployment-ready training claim. Require two distinct input timestamps; same four-observation warm-prefix clock.',geometry='All reference STATE-box cells, one candidate-specific observed-box affine transform, original19 physical-pixel offsets and hull. Same minimum3cells, no other candidate used to select support. This is a NEW isolated-candidate geometry, not repeat of ready two-candidate support counts.',labels='Independently verify every reference against native row by exact float32 confidence/class/frame -> unique raw detection -> unique noncrowd car GTIoU>=.5; independently verify every known current raw box and unique GT. Labels only measure same/different object. No interpolated GT, no model inference, no fr_hex decompression.',dedup='Unique video/reference_frame/current_frame/old_tid/raw_detection; firstUID frozen order. Current public owner is context only and not a training feature or eligibility rule.',readiness='STRICT_LAST has both own and foreign labels in every leave-one-video-out training fold. This is only mathematical trainability of a binary head with an unpenalized intercept, not a sample-size or generalization guarantee. Missing retinal array stops readiness. No relaxation after outcome.',next='Only on readiness: separately freeze per-channel feature restoration, fixed ridge-logistic objective/settings, video-separated evaluation and identical controls. At most two further diagnostics, then full native-ID intervention check or stop this version. No thresholds on the15inspected videos, no automatic expensive core training.',acceptance=[dict(criterion='Every retained label independently verified, sources unchanged, all exclusions/unknowns/cohorts retained; no fr_hex/model/score/tracker work.',met=None),dict(criterion='Both classes remain in every strict-last leave-one-video training fold with saved retinal input.',met=None)],sources={path.relative_to(ROOT).as_posix():sha(path) for path in sorted(set(files))})
    write('protocol.json',p);catpath=ROOT/'experiments/project_history.json';cat=read(catpath);assert not any(c['id']==OUT.name for c in cat['experiments']);cat['experiments'].append(dict(id=OUT.name,title='Обучаемое решение о принадлежности — прямые метки и причинная опора',area='flyvis',state='running',started='2026-11-09',question='Есть ли вне15просмотренныхвидео непосредственно подтверждённые пары для обучения решения по пространственным совпадениям каналов?',hypothesis='45 прежних условных пар могут не исчерпывать архив, но выбор ранней опоры по времени разметки и последний реальный шаблон являются разными режимами.',expected='Один аудит готовых метаданных69видео: отдельные кандидаты, прямыеGTметки, поддержка одного кандидата и строгая причинная опора. Ни обучение, ни новые карты здесь не разрешены.',acceptance=p['acceptance'],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Когорта, действие и критерии до новых подсчётов')]))
    staging=catpath.with_name('project_history.identity_gate_ready.json');staging.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8');staging.replace(catpath);journal('Зафиксирована первая проверка данных для обучаемого отказа',f'*Постановка (`{OUT.name}`).* После накопленной сводки и критики Gemini выбран один кандидат на правку: обучаемая регуляризованная оценка отдельных пространственных совпадений каналов вместо их равного среднего. Это лишь гипотеза. Первый из максимумтрёхдиагностическихшагов — прямые причинные метки вне15просмотренныхвидео. Используются69другихвидеопрежнегоархива, всеизвестныекандидаты, неизвестныенедодумываются. Последняяреальнаяопораотделенаотраннейопоры,выбраннойпоразметке. КонтрольбезFlyVisобязателен. Источник `runs/{OUT.name}/protocol.json`. Модели и сигналы не считаются.');print('Data gate registered',flush=True)
def hits(box,labels):
    objs=[x for x in labels if x['category']=='car' and not (x.get('attributes') or {}).get('crowd',False)]
    if not objs:return []
    b=np.array([[x['box2d'][k] for k in ['x1','y1','x2','y2']] for x in objs]);a=np.array(box);inter=np.maximum(0,np.minimum(a[2:],b[:,2:])-np.maximum(a[:2],b[:,:2])).prod(1);union=(a[2:]-a[:2]).prod()+(b[:,2:]-b[:,:2]).prod(1)-inter;return [str(o['id']) for o,v in zip(objs,inter/union) if v>=.5]
def run():
    p=read(OUT/'protocol.json');verify(p);inventory=read(DATA/'results.json');g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);edges=[];unknown=[];seen=set();checks=0;counts={};retina=True
    for i,(video,info) in enumerate(p['videos'].items()):
        rows=[r for r in inventory['records'] if r['video']==video and r.get('candidates') and r.get('reference_raw') and len(r.get('reference_hits',[]))==1]
        if not rows:continue
        with np.load(ROOT/info['data'],allow_pickle=False) as z:retina&='fr_hex' in z.files;meta={k:z[k] for k in ['ep_offset','ep_tid','ep_pert','fr_frame','fr_crop']}
        with np.load(ROOT/info['raw'],allow_pickle=False) as z:raw={k:z[k] for k in ['frame_idx','cls','conf','xyxy']}
        with np.load(ROOT/info['native'],allow_pickle=False) as z:native=z['outputs']
        temporal={r['uid']:r for r in read(ROOT/info['temporal'])['rows']};gt=read(ROOT/info['annotation']);align=read(ROOT/info['alignment'])['matches'];ann={int(a['video_frame_index']):f.get('labels') or [] for f,a in zip(gt,align)}
        for r in rows:
            t=temporal[r['uid']];e=int(t['episode_local']);off=int(meta['ep_offset'][e]);cur=int(t['current_array_index']);ref=int(r['reference_index']);assert not meta['ep_pert'][e] and meta['ep_tid'][e]==r['tid'] and meta['fr_frame'][cur]==r['frame'] and meta['fr_frame'][ref]==r['reference_frame'];assert len(set(map(int,meta['fr_frame'][off:ref+1])))>=2
            cohort='strict_last' if ref==off+3 else 'earlier_labelled';assert ref<=off+3;obs=r['reference_raw'];nr=native[(native[:,0]==r['reference_frame'])&(native[:,5]==r['tid'])];assert len(nr)==1;ids=np.flatnonzero((raw['frame_idx']==r['reference_frame'])&(raw['cls']==nr[0,7])&(raw['conf'].astype(np.float32)==np.float32(nr[0,6])));assert len(ids)==1 and ids[0]==obs['raw_id'] and np.array_equal(raw['xyxy'][ids[0]],obs['box']);ident=hits(obs['box'],ann[r['reference_frame']]);assert ident==[r['reference_hits'][0]['identity']];checks+=1
            rc=meta['fr_crop'][ref].astype(float);ct=meta['fr_crop'][cur].astype(float);b0=np.array(obs['state_box']);x=reader.points*(rc[2:]-rc[:2])/g['eye_size']+rc[:2]
            for k in r['candidates']:
                rid=k['raw_id'];assert raw['frame_idx'][rid]==r['frame'] and raw['cls'][rid]==2 and np.array_equal(raw['xyxy'][rid],k['box']);current=hits(k['box'],ann[r['frame']]);assert current==[v['identity'] for v in k['hits']];checks+=1
                if len(current)!=1:unknown.append(dict(video=video,uid=r['uid'],raw_id=rid,cohort=cohort,reason='current_GT_not_unique'));continue
                label='own' if current==ident else 'foreign';assert label==k['kind'];key=(video,r['reference_frame'],r['frame'],r['tid'],rid)
                if key in seen:counts['duplicate_edge']=counts.get('duplicate_edge',0)+1;continue
                seen.add(key);b=np.array(k['box']);assert min(*(b0[2:]-b0[:2]),*(b[2:]-b[:2]))>0;trial=b[:2]+(x-b0[:2])*(b[2:]-b[:2])/(b0[2:]-b0[:2]);trial=trial[:,None]+np.array(k['plan']['offsets']);q=(trial-ct[:2])/((ct[2:]-ct[:2])/g['eye_size']);mask=((x>=b0[:2])&(x<=b0[2:])).all(1)&((trial>=b[:2])&(trial<=b[2:])).all((1,2))&(reader.mesh.find_simplex(q)>=0).all(1);idx=np.flatnonzero(mask);supported=len(idx)>=3
                edges.append(dict(video=video,uid=r['uid'],frame=r['frame'],old=r['tid'],raw_id=rid,label=label,reference_identity=ident[0],current_identity=current[0],cohort=cohort,supported=supported,cells=idx,query=q[idx],reference_index=ref,current_index=cur,episode=e,sequence=[[int(meta['fr_frame'][j]),meta['fr_crop'][j].tolist()] for j in range(off,cur+1)],reference_relative=ref-off,reference_box=obs['state_box'],reference_raw_box=obs['box'],current_box=k['box'],original_cost=k['original_cost'],current_native_owners=k['current_native_owners']))
        status('metadata',videos_done=i+1,total=69,known_edges=len(edges),direct_checks=checks);print('LABELS',i+1,69,len(edges),checks,flush=True)
    table={};classes={}
    for cohort in ['strict_last','earlier_labelled']:
        rr=[r for r in edges if r['cohort']==cohort and r['supported']];table[cohort]=dict(known=sum(r['cohort']==cohort for r in edges),supported=len(rr),own=sum(r['label']=='own' for r in rr),foreign=sum(r['label']=='foreign' for r in rr),videos=len({r['video'] for r in rr}),own_videos=len({r['video'] for r in rr if r['label']=='own'}),foreign_videos=len({r['video'] for r in rr if r['label']=='foreign'}));classes[cohort]={video:{r['label'] for r in rr if r['video']!=video} for video in sorted({r['video'] for r in rr})}
    folds=classes['strict_last'];ready=bool(folds) and all(v=={'own','foreign'} for v in folds.values()) and retina;verify(p);write('results.json',dict(passed=True,table=table,edges=edges,unknowns=unknown,other_counts=counts,independent_direct_checks=checks,retinal_array_present=retina,strict_last_LOVO_training_classes=classes['strict_last'],training_readiness=ready,no_fr_hex_decompression=True,no_signal_or_model_or_tracker_calls=True));status('complete',training_readiness=ready,table=table);print('READY',ready,table,flush=True)
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
