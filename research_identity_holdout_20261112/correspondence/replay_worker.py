"""Frozen bounded holdout C34 replay; output scores for separate full-matrix audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,json,gzip,hashlib,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np,cv2
from tracking.census_correspondence import register_translation
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear
from tracking.channel_agreement_correspondence import agreement_channels
from tracking.channel_balanced_correspondence import _center_pair
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cast(x):
    if isinstance(x,dict):return {k:cast(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [cast(v) for v in x]
    return x.tolist() if hasattr(x,'tolist') else x
def write(n,x):
    with (OUT/n).open('xb') as f:f.write((json.dumps(cast(x),ensure_ascii=False,indent=2,allow_nan=False).rstrip()+'\n').encode('utf8'))
p=read(OUT/'protocol.json');assert read(OUT/'replay_GO.json')['approved'] is True
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
core=read(OUT/'core_result.json');assert core['passed'];held={r['key']:r for r in core['records'] if r['mode']=='terminal_hold'}
for r in core['records']:assert sha(ROOT/r['path'])==r['sha256']
cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False);g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);heads=read(ROOT/'research_candidate_support_probe_20261112/scores.json')['heads']['C34'];start=time.monotonic();deadline=datetime.fromisoformat(p['deadline']);rows=[];registrations=[];samples={};pixels={};frame_hashes={};witness=[]
# Decode each complete source video once, keeping exactly required current/past frames.
for video in sorted({c['video'] for c in p['cases']}):
    cases=[c for c in p['cases'] if c['video']==video];wanted={f for c in cases for f in [c['endpoints'][0]['sequence'][-1][0],c['frame']]};cap=cv2.VideoCapture(str(ROOT/cases[0]['video_path']));assert cap.isOpened()
    for f in range(max(wanted)+1):
        ok,bgr=cap.read();assert ok,(video,f)
        if f in wanted:
            gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);pixels[(video,f)]=gray;frame_hashes[video+'|'+str(f)]=hashlib.sha256(bgr.tobytes()).hexdigest()
    cap.release()
    assert time.monotonic()-start<p['replay_timeout_seconds'] and datetime.now(ZoneInfo('Europe/Moscow'))<deadline
for video in sorted({c['video'] for c in p['cases']}):
    case_indices=[i for i,c in enumerate(p['cases']) if c['video']==video];archive=ROOT/p['cases'][case_indices[0]]['event_archive']
    with gzip.open(archive,'rt',encoding='utf8') as f:events=json.load(f)['events']
    byframe={e['frame']:e for e in events}
    for ci in case_indices:
        assert time.monotonic()-start<p['replay_timeout_seconds'] and datetime.now(ZoneInfo('Europe/Moscow'))<deadline
        c=p['cases'][ci];r=c['proposal_geometry'];event=byframe[c['frame']];proposal=next(q for q in event['proposals'] if q['old_id']==c['old_id'] and q['raw_id']==c['raw_id']);rc=np.array(r['reference_crop']);cc=np.array(r['current_crop']);axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2])@np.array([[13.,6.5],[0.,13.]]);offsets=axes*np.maximum((rc[2:]-rc[:2])/g['eye_size'],(cc[2:]-cc[:2])/g['eye_size']);plan=relative_template_plan(reader,r['reference_box'],rc,r['current_box'],cc,offsets);cells=plan['cells'];assert len(cells)==r['full_cells'];source=reader.points[cells]*(rc[2:]-rc[:2])/reader.eye_size+rc[:2];gray0=pixels[(video,c['endpoints'][0]['sequence'][-1][0])];gray1=pixels[(video,c['frame'])]
        registration=register_translation(gray0,gray1,r['reference_box'],rc,r['current_box'],cc,source,reader.mesh,reader.eye_size);registrations.append(dict(case_index=ci,case_key=c['case_key'],**registration));samples[f'{ci}_gray0']=gray0;samples[f'{ci}_gray1']=gray1;samples[f'{ci}_cells']=cells;samples[f'{ci}_source_world']=source
        a=np.load(ROOT/c['endpoints'][0]['maps'][0]['path'],allow_pickle=False).astype(float);b=np.load(ROOT/c['endpoints'][1]['maps'][0]['path'],allow_pickle=False).astype(float);h=np.load(ROOT/held[c['endpoints'][0]['key']]['path'],allow_pickle=False).astype(float);samples[f'{ci}_reference_index']=a;samples[f'{ci}_reference_terminal_hold']=h;samples[f'{ci}_current_index']=b;queries=dict(original=plan['query'][:,r['winner']])
        if registration['ready']:queries['census']=(source+registration['translation']-cc[:2])*reader.eye_size/(cc[2:]-cc[:2])
        for coordinate,q in queries.items():
            for reference_mode,Amap in [('index',a),('terminal_hold',h)]:
                A=Amap[cells];B=sample_linear(reader.mesh,b,q);key=f'{ci}_{coordinate}_{reference_mode}';samples[key+'_a']=A;samples[key+'_b']=B;samples[key+'_q']=q
                for head in ['native','agreement']:
                    if head=='agreement':ch=agreement_channels(A,B,mean_penalty=True)
                    else:
                        aa,bb,an,bn,active,activeb=_center_pair(A,B);ch=None if not active.any() or not activeb.any() else np.divide((aa*bb).sum(0),an*bn,out=np.zeros(34),where=an*bn>0)*34/active.sum()
                    if ch is None:
                        rows.append(dict(case_index=ci,case_key=c['case_key'],event_key=c['event_key'],video=video,frame=c['frame'],old_id=c['old_id'],raw_id=c['raw_id'],representation='C34',coordinate=coordinate,reference_mode=reference_mode,head=head,count=len(cells),channels=None,mean_similarity=None,score=None,accepted=None,reason='constant_patch_no_visual_evidence',patch_key=key))
                        continue
                    model=heads[head];features=np.r_[(np.array(r['geometry'])-model['geometry']['mean'])/model['geometry']['scale'],(ch-model['signal_mean'])/model['signal_scale']];theta=np.array(model['theta']);score=float(features@theta[:-1]+theta[-1])
                    if coordinate=='original' and reference_mode=='index' and head=='native':
                        error=float(abs(ch-proposal['channels']).max());score_error=abs(score-proposal['logit']);assert error<1e-9 and score_error<1e-9,(c['case_key'],error,score_error);witness.append(dict(case_key=c['case_key'],channel_error=error,score_error=score_error))
                    rows.append(dict(case_index=ci,case_key=c['case_key'],event_key=c['event_key'],video=video,frame=c['frame'],old_id=c['old_id'],raw_id=c['raw_id'],representation='C34',coordinate=coordinate,reference_mode=reference_mode,head=head,count=len(cells),channels=ch,mean_similarity=float(ch.mean()),score=score,accepted=bool(score>0),patch_key=key))
        (OUT/'replay_status.json').write_bytes((json.dumps(dict(stage='measured',case_key=c['case_key'],completed=len(witness),total=len(p['cases']),updated=datetime.now(ZoneInfo('Europe/Moscow')).isoformat()))+'\n').encode('utf8'))
        print('REPLAY_CASE',ci,c['case_key'],flush=True)
    del events,byframe
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
with (OUT/'samples.npz').open('xb') as f:np.savez_compressed(f,**samples)
write('result.json',dict(passed=True,records=rows,registrations=registrations,baseline_witness=witness,core_result_sha256=sha(OUT/'core_result.json'),samples_sha256=sha(OUT/'samples.npz'),frame_hashes=frame_hashes,elapsed_seconds=time.monotonic()-start,new_core_calls=0,new_fits=0,tracker_calls=0,GT_used_for_decision=False,scope='Head proposals only; complete appearance guard and actual matrix replay are separately mandatory. Unavailable Census is explicit missing proposal, not fabricated score.'))
