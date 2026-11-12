import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,json,hashlib,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
import numpy as np,cv2
from registration import register_translation
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear
from tracking.channel_agreement_correspondence import agreement_channels
from tracking.channel_balanced_correspondence import _center_pair
def read(p):return json.loads(p.read_text(encoding='utf8'))
def cast(x):
    if isinstance(x,dict):return {k:cast(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [cast(v) for v in x]
    return x.tolist() if hasattr(x,'tolist') else x
def write(n,x):
    data=json.dumps(cast(x),ensure_ascii=False,indent=2,allow_nan=False).rstrip()+'\n'
    with (OUT/n).open('xb') as f:f.write(data.encode('utf8'))
p=read(OUT/'proposal_protocol.json');assert read(OUT/'GO.json')['approved'] is True
for n,h in p['inputs'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False)
stage=read(ROOT/'research_stage_localization_20261112/protocol.json');old=read(ROOT/'research_remaining_causes_20261112/geometry832/result.json');frames=read(ROOT/'research_failure_localization_20261112/frame_manifest.json')['records'];heads=read(ROOT/'research_candidate_support_probe_20261112/scores.json')['heads'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);records=[];registrations=[];samples={};baselineerr=0.;start=time.monotonic();deadline=datetime.fromisoformat(p['deadline'])
with np.load(ROOT/'research_input_precision_probe_20261112/maps.npz') as maps,np.load(ROOT/'research_input_precision_probe_20261112/expanded_inputs.npz') as inputs,np.load(ROOT/'research_terminal_response_probe_20261112/maps.npz') as held:
    for ci,c in enumerate(stage['cases']):
        assert time.monotonic()-start<p['timeout_seconds'] and datetime.now(ZoneInfo('Europe/Moscow'))<deadline
        r=c['case'];plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets']);cells=plan['cells'];rc=np.array(r['reference_crop']);cc=np.array(r['current_crop']);source=reader.points[cells]*(rc[2:]-rc[:2])/reader.eye_size+rc[:2];endpoints=sorted([z for z in frames if z['old_id']==r['old_id']],key=lambda z:z['frame']);gray=[]
        for ei,z in enumerate(endpoints):
            im=cv2.imread(str(ROOT/'research_failure_localization_20261112'/z['frame_png']),cv2.IMREAD_GRAYSCALE);assert im is not None and im.shape==(720,1280);gray.append(im);samples[f'{ci}_gray{ei}']=im
        registration=register_translation(gray[0],gray[1],r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],source,reader.mesh,reader.eye_size);registrations.append(dict(case_index=ci,old_id=r['old_id'],**registration));samples[f'{ci}_cells']=cells;samples[f'{ci}_source_world']=source
        original=plan['query'][:,r['old_variants']['C34']['winner']];queries=dict(original=original)
        if registration['ready']:queries['census']=(source+registration['translation']-cc[:2])*reader.eye_size/(cc[2:]-cc[:2])
        for arm in ['C34','R134_temporal','retina32']:
            bank={}
            for ei,ep in enumerate(c['endpoints']):
                for mode in ['index','terminal_hold']:
                    if arm=='retina32':m=inputs[f'precision32_{ci}'][ep['length']-1,:,None].astype(float)
                    else:
                        suffix='C34' if arm=='C34' else 'R1';m=maps[f'index_{ci}_{ei}_{suffix}'].astype(float) if mode=='index' else held[f'terminal_hold_{ci}_{ei}_{suffix}'].astype(float)
                    bank[(ei,mode)]=m;samples[f'{ci}_{arm}_{ei}_{mode}']=m
            for coordinate,q in queries.items():
                for reference_mode in ['index','terminal_hold']:
                    A=bank[(0,reference_mode)][cells];B=sample_linear(reader.mesh,bank[(1,'index')],q);key=f'{ci}_{arm}_{coordinate}_{reference_mode}';samples[key+'_a']=A;samples[key+'_b']=B;samples[key+'_q']=q
                    for head in ['native','agreement']:
                        if head=='agreement':ch=agreement_channels(A,B,mean_penalty=True)
                        else:
                            a,b,an,bn,active,_=_center_pair(A,B);ch=np.divide((a*b).sum(0),an*bn,out=np.zeros(A.shape[1]),where=an*bn>0)*A.shape[1]/active.sum()
                        score=None
                        if arm!='retina32':
                            h=heads[arm][head];feat=np.r_[(np.array(r['geometry'])-h['geometry']['mean'])/h['geometry']['scale'],(ch-h['signal_mean'])/h['signal_scale']];theta=np.array(h['theta']);score=float(feat@theta[:-1]+theta[-1])
                        if coordinate=='original' and reference_mode=='index':
                            z=next(z for z in old['records'] if z['old_id']==r['old_id'] and z['representation']==arm and z['query']=='C34' and z['variant']=='detector_affine_saved_offset' and z['support']=='common' and z['head']==head);error=float(abs(ch-z['channels']).max());baselineerr=max(baselineerr,error);assert error<1e-9
                        records.append(dict(case_index=ci,old_id=r['old_id'],label=r['label'],representation=arm,coordinate=coordinate,reference_mode=reference_mode,head=head,count=len(cells),channels=ch,mean_similarity=float(ch.mean()),score=score,accepted=None if score is None else bool(score>0),action='accept' if score is not None and score>0 else 'defer',patch_key=key,map0_key=f'{ci}_{arm}_0_{reference_mode}',map1_key=f'{ci}_{arm}_1_index'))
        (OUT/'status.json').write_bytes((json.dumps(dict(stage='measured',old_id=r['old_id'],updated=datetime.now(ZoneInfo('Europe/Moscow')).isoformat()))+'\n').encode('utf8'))
for n,h in p['inputs'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
with (OUT/'samples.npz').open('xb') as f:np.savez_compressed(f,**samples)
write('result.json',dict(passed=True,registrations=registrations,records=records,geometry=[dict(old_id=c['case']['old_id'],geometry=c['case']['geometry']) for c in stage['cases']],max_baseline_error=baselineerr,elapsed_seconds=time.monotonic()-start,samples_sha256=hashlib.sha256((OUT/'samples.npz').read_bytes()).hexdigest(),core_calls=0,new_fits=0,tracker_calls=0,labels_used_for_registration=False))
print('REPAIR COMPLETE',len(records),flush=True)
