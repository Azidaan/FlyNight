"""Read-only coverage gate: an older exactly annotated observed reference.

Annotation chooses evaluation timestamps only. This is not a deployable memory
policy, and a larger evaluable dataset does not demonstrate better tracking.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import ast,hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;MSK=timezone(timedelta(hours=3));DEADLINE=datetime(2026,11,9,6,40,tzinfo=MSK);STOP=DEADLINE-timedelta(minutes=5)
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(STOP-datetime.now(MSK)).total_seconds()
if remaining<=0:raise SystemExit('Scientific save/stop deadline reached; no worker started')
guard_entrypoint(timeout=min(900,remaining))
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.foveated_eyes import crop_box
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cast(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [cast(x) for x in v]
    return v
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for path,h in p['sources'].items():assert sha(ROOT/path)==h,path
def register():
    parent=ROOT/'runs/frozen_reader_transfer_20261109/external_selection_protocol.json';old=read(parent);verify(old)
    files=[Path(__file__),parent,ROOT/'runs/frozen_reader_transfer_20261109/external_inventory.py',ROOT/'runs/clipped_anchor_readout_20261109/study.py',ROOT/'runs/z13_local_peak_20261021/geometry.json',ROOT/'tracking/observed_spatial_reader.py',ROOT/'tracking/channel_balanced_correspondence.py',ROOT/'tracking/clipping_aware_correspondence.py',ROOT/'tracking/foveated_eyes.py']+[ROOT/p for p in old['sources']]
    p=dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),deadline=DEADLINE.isoformat(),scientific_save_stop=STOP.isoformat(),videos=old['videos'],selection='All517 stored singleton-car events from the46 other videos, same forward IoU>=0.2, native age<=6, same raw detections. For each actually annotated current frame, choose the latest earlier annotation-timestamp at which this old native track has a public observed row, no later than its last observation. Do not skip a candidate timestamp because identity/IoU is inconvenient. Then independently label both raw detections with exactly one same-class noncrowd GT IoU>=0.5. Ambiguous/unmatched/unannotated stay unknown. No score or identity-based selection of reference time.',
        memory_scope='Evaluation-timestamp construction ONLY. GT is not a runtime memory-selection input. A causal deployment must keep reference history without knowing which frames will be annotated. Older reference age, visibility, crop and neural-state history differ; none is presumed beneficial.',
        readiness='Compute frozen legacy and clipping-aware common19-shift support only. New reference crop is deterministically reconstructed from archived float32 native state; it is not falsely claimed to be a byte-exact old crop. Store up to4 prior public observations as a proposed causal reference recipe; do not execute core, render frames, compare features or train.',
        literature='STARK, Yan et al., ICCV2021, arxiv2103.17154, official stark_st.py lines42-54 and84-92 retains initial plus dynamic templates. This supports considering multiple memories, not selecting a template by GT or proving that an older FlyVis template is better.',
        acceptance='Every saved event accounted for; no score/model/image calls; exact raw/native provenance; actual annotation frames only; unknowns retained; immutable sources. Report direct own/foreign and clipped strata, support, reference age and potentially available causal recipes.',
        strongest_alternative='The last template is still the right object; the comparison follows background/common structure. Earlier annotated memory merely increases verification coverage and may worsen appearance correspondence.',sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))})
    write('protocol.json',p)
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=path.read_text(encoding='utf8');now=datetime.now(MSK);entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Проверка доступности более раннего размеченного наблюдения; срок06:40.\n\n*Постановка (`{OUT.name}`).* По новому поручению автора автономная работа до09.11.202606:40МСК, научное сохранение/остановка с06:35. Прежняя проверка517событий46видео обнаружила лишь2прямоподтверждённыесвоипары, поскольку последний кадр памяти почти всегда не размечен. Теперь дооценок выбирается последний более ранний размеченный по расписанию кадр, где старый нативный трек реально наблюдался; неудобныеID/IoUне дают права выбрать другой кадр. МеткиGTтолькооценочные, не входят в будущее правило памяти. Проверяется лишь покрытие, происхождение и фиксированная геометрия; сети, изображения, готовыеоценкинепересчитываются. Это не принятое улучшение памяти. Источник `runs/{OUT.name}/protocol.json`.\n\n';path.write_text(s.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8');print('Registered data-only gate, deadline',p['deadline'],flush=True)
def run():
    p=read(OUT/'protocol.json');verify(p)
    if (OUT/'results.json').exists():raise SystemExit('Ready data audit exists; no repetition')
    ns=dict(np=np);source=ROOT/'runs/clipped_anchor_readout_20261109/study.py';node=next(n for n in ast.parse(source.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='raw_observation');exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    source=ROOT/'runs/frozen_reader_transfer_20261109/external_inventory.py';node=next(n for n in ast.parse(source.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='match');exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    cfg=read(ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json')['native_config'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2]);records=[];counts={}
    for video,info in p['videos'].items():
        if datetime.now(MSK)>=STOP:raise RuntimeError('Save/stop deadline reached; retain protocol and logs')
        gt=read(ROOT/info['annotation']);alignment=read(ROOT/info['alignment'])['matches'];assert len(gt)==len(alignment);ann={int(a['video_frame_index']):f.get('labels') or [] for f,a in zip(gt,alignment)}
        with np.load(ROOT/info['native'],allow_pickle=False) as z:native=z['outputs']
        with np.load(ROOT/info['raw'],allow_pickle=False) as z:raw={n:z[n] for n in ['xyxy','cls','conf','frame_idx']}
        vc={}
        for event in read((ROOT/info['native']).with_suffix('.json'))['events']:
            for row in event['rows']:
                choices=[c for c in row['candidates'] if c['forward_iou']>=.2]
                if len(choices)!=1 or choices[0]['new_tid'] is None or int(row['row_data']['cls'])!=2:continue
                can=choices[0];t=row['row_data'];h=t['history'][-1];frame=event['frame'];r=dict(video=video,split=info['split'],frame=frame,old=row['old_tid'],new=can['new_tid'],last_observation=h['frame'],kind='unknown',status=None)
                if frame not in ann:r['status']='current_unannotated'
                else:
                    current=ns['raw_observation'](native,raw,can['new_tid'],frame,cfg);assert np.array_equal(current['box'],can['box']);hits=ns['match'](current['box'].tolist(),ann[frame]);r.update(current_raw=current,current_gt_hits=hits)
                    past=native[(native[:,5]==row['old_tid'])&(native[:,0]<=h['frame'])&(native[:,0]<frame)];times=[int(v) for v in past[:,0] if int(v) in ann]
                    if not times:r['status']='no_prior_annotated_observed_frame'
                    else:
                        anchor=max(times);old=ns['raw_observation'](native,raw,row['old_tid'],anchor,cfg);oldhits=ns['match'](old['box'].tolist(),ann[anchor]);r.update(reference_frame=anchor,reference_age=frame-anchor,reference_raw=old,reference_gt_hits=oldhits)
                        if len(hits)!=1 or len(oldhits)!=1:r['status']='not_both_unique'
                        else:
                            r['status']='both_unique';r['kind']='own' if hits[0]['identity']==oldhits[0]['identity'] else 'foreign';r.update(track=t,candidate=can)
                            def make_crop(b):return list(crop_box(((b[0]+b[2])/2,(b[1]+b[3])/2,b[2]-b[0],b[3]-b[1]),(1280,720)))
                            oldbox=np.array(old['state_box']);crop=make_crop(oldbox);prediction=np.array(t['mean'][:2])-(oldbox[:2]+oldbox[2:])/2;ct=np.array(t['current_crop']);scale=np.maximum((np.array(crop[2:])-crop[:2])/g['eye_size'],(ct[2:]-ct[:2])/g['eye_size']);offsets=(axes@np.array([[13.,6.5],[0.,13.]]))*scale
                            r['legacy_plan']=reader.plan(oldbox,crop,can['box'],ct,prediction,offsets,use_visible_edge=False);r['joint_plan']=reader.plan(oldbox,crop,can['box'],ct,prediction,offsets,reference_observation=old['box']);r['clipped']=any(b[0]<=0 or b[1]<=0 or b[2]>=1280 or b[3]>=720 for b in [old['box'],current['box']])
                            prefix=past[past[:,0]<=anchor][-4:];r['reference_recipe']=[[int(v[0]),make_crop(v[1:5])] for v in prefix];r['reference_key']=hashlib.sha256(json.dumps([video,r['reference_recipe']],separators=(',',':')).encode()).hexdigest()[:32];current_recipe=[[v['frame'],v['crop']] for v in t['history']]+t['search'];r['current_recipe']=current_recipe;r['current_key']=hashlib.sha256(json.dumps([video,current_recipe],separators=(',',':')).encode()).hexdigest()[:32]
                records.append(r);vc[r['status']]=vc.get(r['status'],0)+1
        counts[video]=vc;print('Inventory',video,vc,flush=True)
    assert len(records)==517
    good=[r for r in records if r['status']=='both_unique'];table={}
    for r in good:
        key=f'{r["kind"]}|clipped={r["clipped"]}|support={len(r["joint_plan"]["cells"])>=3}|prefix2={len(r["reference_recipe"])>=2}';table[key]=table.get(key,0)+1
    verify(p);write('results.json',dict(passed=True,records=records,counts=counts,unique_pairs=len(good),strata=table,no_model_calls=True,no_image_reads=True,no_score_calls=True,sources_stable=True));print('SUMMARY',len(good),table,flush=True)
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
