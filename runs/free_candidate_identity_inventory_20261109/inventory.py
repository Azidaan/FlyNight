"""Read-only gate: genuine same-query own/foreign alternatives in the free pool."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import ast,hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,6,35,tzinfo=MSK)
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(STOP-datetime.now(MSK)).total_seconds()
if remaining<=0:raise SystemExit('Scientific save/stop reached')
guard_entrypoint(timeout=min(900,remaining))
import numpy as np
from tracking.foveated_eyes import crop_box
from tracking.observed_spatial_reader import ObservedSpatialReader
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def cast(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [cast(x) for x in v]
    return v
def write(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for f,h in p['sources'].items():assert sha(ROOT/f)==h,f
def stamp():return datetime.now(MSK).isoformat()
def journal(title,body):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=p.read_text(encoding='utf8');now=datetime.now(MSK);p.write_text(s.replace('## Журнал изменений\n',f'## Журнал изменений\n\n**{now:%d.%m.%Y, %H:%M} MSK** — {title}.\n\n{body}\n\n',1),encoding='utf8')
def register():
    prior=ROOT/'runs/direct_memory_anchor_inventory_20261109/protocol.json';p=read(prior);verify(p)
    assert read(ROOT/'runs/common_state_memory_20261109/final_checks.json')['passed']
    sources=[Path(__file__),prior,ROOT/'runs/common_state_memory_20261109/summary.json',ROOT/'runs/common_state_memory_20261109/context_audit.json',ROOT/'runs/common_state_memory_20261109/candidate_audit.json',ROOT/'runs/common_state_memory_20261109/ownership_audit.json',ROOT/'runs/clipped_anchor_readout_20261109/study.py',ROOT/'runs/frozen_reader_transfer_20261109/external_inventory.py',ROOT/'runs/z13_local_peak_20261021/geometry.json',ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json',ROOT/'research_bbox_confirmation_inventory_20261108/export.py',ROOT/'tracking/observed_spatial_reader.py']
    write('protocol.json',dict(registered=stamp(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),scientific_save_stop=STOP.isoformat(),deadline='2026-11-09T06:40:00+03:00',videos=p['videos'],selection='All saved same-class car Lost rows on46 other videos with at least2 free high-confidence candidates at unchanged forwardIoU>=0.2, age<=6 and at least one native first-public candidate. The517 singleton events have already been audited and are not relabelled. Unannotated queries and ambiguous GT remain explicit.',reference='Latest earlier annotation-timestamp with a real public native old-track observation, no later than last observation. Never skip a timestamp because identity/IoU is inconvenient. Unique same-class noncrowd GT IoU>=0.5, no interpolation. GT selects an evaluation reference only, never a runtime policy.',readiness='A clean query has unique reference identity, at least one uniquely own and one uniquely foreign available free candidate, each with>=3 frozen19-shift cells. Report and exclude from clean-readiness old references that already have another public native ID of the same unique GT, or targets already carried at current time by a non-newly-confirmed native track. Labels define diagnostic coverage, not runtime vetoes.',gate='At least3 clean queries across at least3 videos and at least1 query where the minimum existing native cost does NOT select only a uniquely own candidate. If gate fails, no map recovery or new similarity is justified by this inventory. Do not retune the gate to available cases.',math='A within-query comparison needs both true and foreign objects actually present in the candidate set. A wrong-only residual pool cannot test choosing the true object. For the frozen unclipped prediction-aligned reader, the same reference point has the same current query regardless of candidate; intersecting supports across all candidates therefore produces identical score grids. Do not propose that intersection as an identity discriminator; report native support separately.',literature='Mayer et al., KeepTrack, ICCV2021, https://arxiv.org/pdf/2103.16556, sections3 and4.1, official keep_track.py: joint candidate identities differ from a pooled similarity threshold. Already reviewed locally25Oct; no claim to implement KeepTrack or that it universally solves a sole candidate.',scope='Data-only eligibility/provenance, no image/model/core/detector/tracker replay, no new similarity or decision threshold. Goal remains first problem of correct object continuation; full ID benefit remains open.',sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources}))
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath);assert not any(c['id']==OUT.name for c in cat['experiments']);cat['experiments'].append(dict(id=OUT.name,title='Своя и чужая рамки в одном запросе — доступность чистого сравнения',area='flyvis',state='running',started='2026-11-09',question='Есть ли в сохранённом свободном пуле запросы, где правильный и чужой варианты одновременно доступны и старыйIDне является дубликатом?',hypothesis='Плохой общий порядок сходства может смешивать отсутствие кандидата, дублированиеIDи реальные ошибки выбора. Нужен проверяемый набор выбора внутри одного запроса.',expected='Полный учёт всех новых многокандидатных запросов, прямыеGTопоры, строгие неизвестные, без новых карт и повторения517singleton-проверок.',acceptance=[dict(criterion='Все новые запросы учтены, источники стабильны, неизвестные и дубликаты отделены; no model/score calls.',met=None),dict(criterion='Не менее3чистыхзапросов/3видео и хотябы1ошибка исходнойстоимостибезFlyVisдляпродолжения к новымкартам.',met=None)],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Условия доступности и остановки до нового просмотра разметки')]))
    catpath.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8');journal('Проверка доступности честного выбора между своей и чужой рамками',f'*Постановка (`{OUT.name}`).* После отрицательного опыта ранней памяти проверяется только сохранённый многокандидатный контекст46видео. Внутри одного запроса должны одновременно существовать своя и чужая свободные рамки, без уже существующего другого сопровождения той жеGTмашины.517однокандидатныхсобытий не повторяются. ИсходныйIoU0,2/давность6, прямойGTIoU0,5 и19сдвигов неизменны. Минимум для дальнейшихкарт —3чистыхзапроса/3видео и хотябы1ошибка прежнейстоимости; иначе картыневосстанавливаются. Сейчас только метаданные, ядро, изображения и трекер не запускаются. Источник `runs/{OUT.name}/protocol.json`.');print('Registered clean-query inventory',flush=True)
def run():
    p=read(OUT/'protocol.json');verify(p)
    if (OUT/'results.json').exists():raise SystemExit('Ready inventory exists; no repetition')
    parent=read(ROOT/'runs/direct_memory_anchor_inventory_20261109/protocol.json');verify(parent);cfg=read(ROOT/'runs/bbox_confirmation_inventory_20261108/protocol.json')['native_config'];ns=dict(np=np)
    for path,name in [(ROOT/'runs/clipped_anchor_readout_20261109/study.py','raw_observation'),(ROOT/'runs/frozen_reader_transfer_20261109/external_inventory.py','match')]:
        node=next(n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2]);records=[];totals={};clean=[]
    for video,info in p['videos'].items():
        gt=read(ROOT/info['annotation']);align=read(ROOT/info['alignment'])['matches'];ann={int(a['video_frame_index']):f.get('labels') or [] for f,a in zip(gt,align)}
        with np.load(ROOT/info['native'],allow_pickle=False) as z:native=z['outputs']
        with np.load(ROOT/info['raw'],allow_pickle=False) as z:raw={n:z[n] for n in ['frame_idx','conf','cls','xyxy']}
        def observed(tid,frame):return ns['raw_observation'](native,raw,tid,frame,cfg)
        def same_owners(identity,frame):
            result=[]
            for v in native[native[:,0]==frame]:
                obs=observed(int(v[5]),frame);hits=ns['match'](obs['box'],ann[frame])
                if len(hits)==1 and hits[0]['identity']==identity:result.append(dict(tid=int(v[5]),raw_id=obs['raw_id']))
            return result
        for event in read((ROOT/info['native']).with_suffix('.json'))['events']:
            frame=event['frame']
            for row in event['rows']:
                t=row['row_data'];choices=[c for c in row['candidates'] if c['forward_iou']>=.2]
                if len(choices)<2 or int(t['cls'])!=2 or not any(c['new_tid'] is not None for c in choices) or row['age']>6:continue
                r=dict(video=video,frame=frame,old=row['old_tid'],candidate_count=len(choices),status='unknown',clean=False)
                if frame not in ann:r['status']='current_unannotated'
                else:
                    past=native[(native[:,5]==row['old_tid'])&(native[:,0]<=t['history'][-1]['frame'])&(native[:,0]<frame)];times=[int(f) for f in past[:,0] if int(f) in ann]
                    if not times:r['status']='no_prior_annotated_observation'
                    else:
                        anchor=max(times);old=observed(row['old_tid'],anchor);hits=ns['match'](old['box'],ann[anchor]);r.update(reference_frame=anchor,reference_raw=old,reference_hits=hits)
                        if len(hits)!=1:r['status']='reference_not_unique'
                        else:
                            identity=hits[0]['identity'];otherref=[o for o in same_owners(identity,anchor) if o['tid']!=row['old_tid']];working=[o for o in same_owners(identity,frame) if o['tid'] not in event['newly_confirmed']];r.update(other_public_same_gt_at_reference=otherref,working_public_same_gt_now=working,candidates=[])
                            def crop(b):return list(crop_box(((b[0]+b[2])/2,(b[1]+b[3])/2,b[2]-b[0],b[3]-b[1]),(1280,720)))
                            rc=crop(np.array(old['state_box']));ct=np.array(t['current_crop']);scale=np.maximum((np.array(t['history'][-1]['crop'][2:])-t['history'][-1]['crop'][:2])/g['eye_size'],(ct[2:]-ct[:2])/g['eye_size']);offsets=(axes@np.array([[13.,6.5],[0.,13.]]))*scale;pred=np.array(t['mean'][:2])-(np.array(old['state_box'][:2])+old['state_box'][2:])/2
                            for candidate in choices:
                                ch=ns['match'](candidate['box'],ann[frame]);kind='own' if len(ch)==1 and ch[0]['identity']==identity else ('foreign' if len(ch)==1 else 'unknown');plan=reader.plan(old['state_box'],rc,candidate['box'],ct,pred,offsets,reference_observation=old['box']);r['candidates'].append(dict(**candidate,kind=kind,hits=ch,plan=plan))
                            own=[c for c in r['candidates'] if c['kind']=='own' and len(c['plan']['cells'])>=3];foreign=[c for c in r['candidates'] if c['kind']=='foreign' and len(c['plan']['cells'])>=3]
                            r['status']='both_available' if own and foreign else 'not_both_supported';r['clean']=bool(own and foreign and not otherref and not working)
                            if r['clean']:
                                cost=min(c['cost'] for c in r['candidates']);winners=[c for c in r['candidates'] if c['cost']==cost];r['native_cost_selects_only_own']=all(c['kind']=='own' for c in winners);r['geometry_top_own']=max(c['forward_iou'] for c in own)>max(c['forward_iou'] for c in foreign)
                                prefix=past[past[:,0]<=anchor][-4:];recipe=[[int(v[0]),crop(v[1:5])] for v in prefix];known={v['frame']:v['crop'] for v in t['history']};tail=past[past[:,0]>anchor];r.update(reference_recipe=recipe,current_recipe=recipe+[[int(v[0]),known.get(int(v[0]),crop(v[1:5]))] for v in tail]+t['search'],track=t)
                                clean.append(r)
                records.append(r);totals[r['status']]=totals.get(r['status'],0)+1
        print('INVENTORY',video,'queries',len(records),'clean',len(clean),flush=True)
    video_count=len({r['video'] for r in clean});cost_errors=sum(not r['native_cost_selects_only_own'] for r in clean);gate=len(clean)>=3 and video_count>=3 and cost_errors>=1
    verify(p);verify(parent);write('results.json',dict(passed=True,records=records,totals=totals,clean_queries=len(clean),clean_videos=video_count,native_cost_error_or_unknown_queries=cost_errors,readiness_gate=gate,no_model_image_tracker_score_calls=True,singleton517_not_relabelled=True));print('RESULT',len(records),len(clean),video_count,cost_errors,gate,flush=True)
if __name__=='__main__':{'register':register,'run':run}[sys.argv[1]]()
