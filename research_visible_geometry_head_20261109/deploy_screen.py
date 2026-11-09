"""Four final fits and a fixed same-history screen; no actual ID claim."""
import os,sys,json,hashlib,gzip
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;TRAIN=ROOT/'research_domain_matched_reader_20261109';NATIVE=ROOT/'runs/domain_matched_native_20261109';OLD=ROOT/'runs/full_template_native_20261109'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)));guard_entrypoint(timeout=min(1200,(STOP-datetime.now(STOP.tzinfo)).total_seconds()))
import psutil,numpy as np
from scipy.special import expit
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cast(v):
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [cast(x) for x in v]
    return v.tolist() if hasattr(v,'tolist') else v
def save(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
p=read(OUT/'deployment_screen_protocol.json')
def verify():
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
verify();assert read(OUT/'audit.json')['passed'] and read(OUT/'summary.json')['gates']['C34__mask_only']
from tracking.channel_identity_head import fit_ridge_logistic,spatial_channel_features
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.geometric_template_correspondence import geometric_template_plan
from tracking.spatial_identity_assignment import geometric_match
rows=read(OUT/'features.json')['records'];oldrows=read(TRAIN/'features.json')['records'];baseline=read(NATIVE/'deployment.json');stats=baseline['geometry_normalization'];G=np.array([r['geometry'] for r in rows]);g=(G-stats['mean'])/stats['scale'];y=np.array([r['label']=='own' for r in rows],int)
models={'G7':baseline['models']['G7']};training_original={};errors={};scale_ratio=None
for arm in ['C34','R134_spatial','R134_temporal','C34_scale']:
    if arm=='C34_scale':
        raw=np.array([r['variants']['C34']['channels'] for r in oldrows]);target=np.array([r['variants']['C34']['mask_only']['channels'] for r in rows]);sd=raw.std(0);scale_ratio=np.divide(target.std(0),sd,out=np.ones_like(sd),where=sd>0);value=raw*scale_ratio
    else:value=np.array([r['variants'][arm]['mask_only']['channels'] for r in rows])
    X=np.c_[g,value];model=fit_ridge_logistic(X,y);theta=np.array(model['theta']);z=X@theta[:-1]+theta[-1];weights=len(y)/(2*np.bincount(y)[y]);residual=weights*(expit(z)-y);gradient=np.r_[X.T@residual/len(y)+theta[:-1],residual.mean()];error=float(abs(gradient).max());assert error<1e-7;errors[arm]=error
    training_original[arm]=model
    if arm=='C34_scale':
        model=dict(model);effective=theta.copy();effective[7:-1]*=scale_ratio;model['theta']=effective
        np.testing.assert_allclose(np.c_[g,raw]@effective[:-1]+effective[-1],z,atol=1e-12,rtol=0)
    models[arm]=model
save('deployment.json',dict(passed=True,geometry_normalization=stats,models=models,training_original=training_original,scale_ratio=scale_ratio,independent_gradient_max=errors,
    pairs=26,own=20,foreign=6,threshold=0,regularization=1,readout='mask_only: common visible reference support, ORIGINAL box coordinates andoldmeanwinner',
    deployment_screen_sha256=sha(OUT/'deployment_screen_protocol.json'),source_geometry_model_sha256=sha(NATIVE/'deployment.json'),actual_IDs_evaluated=False))
geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720]);axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2])@np.array([[13.,6.5],[0.,13.]])
poses={(r['video'],r['frame'],r['old_id'],r['raw_id']):r for r in read(ROOT/'research_geometry_consistency_20261109/coordinate_report.json')['records']}
cases=[];restored=0;reused=0
def seqkey(video,s):return hashlib.sha256(json.dumps([video,s],separators=(',',':')).encode()).hexdigest()[:32]
def decision(arm,geo,features):
    theta=np.array(models[arm]['theta']);x=geo if arm=='G7' else np.r_[geo,features];return float(x@theta[:-1]+theta[-1])
for case in p['cases']:
    video=case['video'];frame=case['frame']
    with gzip.open(NATIVE/'rollouts/C34'/(video+'.events.json.gz'),'rt',encoding='utf8') as f:events=json.load(f)['events']
    event=next(e for e in events if e['frame']==frame);q=next(q for q in event['proposals'] if q['old_id']==case['old_id'] and q['raw_id']==case['raw_id']);row=event['rows'][q['row']]
    with gzip.open(OLD/'rollouts/C34'/(video+'.events.json.gz'),'rt',encoding='utf8') as f:old_events=json.load(f)['events']
    previous_event=next(e for e in old_events if e['frame']==frame);previous_q=next(x for x in previous_event['proposals'] if x['raw_id']==q['raw_id']);previous_row=previous_event['rows'][previous_q['row']]
    equal=np.array_equal(row['mean'],previous_row['mean']) and np.array_equal(row['covariance'],previous_row['covariance']) and all(q[k]==previous_q[k] for k in ['reference_box','reference_crop','current_box','current_crop'])
    if equal:
        pose=poses[(video,frame,previous_q['old_id'],q['raw_id'])];scale=pose['scale_geom'];offset=pose['offset_reference_pixels'];reused+=1;pose_source='exact_previous_state_and_inputs_reused'
    else:
        pose=geometric_match(row['mean'],row['covariance'],row['history'],row['current_crop'],q['current_box'],geom);assert pose['valid'];scale=pose['scale'];offset=pose['offset'];restored+=1;pose_source='missing_current_state_geometry_restored_once'
    rc=np.array(q['reference_crop']);ct=np.array(q['current_crop']);offsets=axes*np.maximum((rc[2:]-rc[:2])/geom['eye_size'],(ct[2:]-ct[:2])/geom['eye_size'])
    old=relative_template_plan(reader,q['reference_box'],rc,q['current_box'],ct,offsets);visible=geometric_template_plan(reader,q['reference_box'],rc,q['current_box'],ct,offsets,scale,offset);cells=np.intersect1d(old['cells'],visible['cells']);indices=np.searchsorted(old['cells'],cells)
    reference=[[h['frame'],h['crop']] for h in q['history']];maps=[np.load(NATIVE/'features'/video/(seqkey(video,s)+'.C34.npy'),allow_pickle=False) for s in [reference,reference+q['search']]]
    geo=(np.array(q['geometry'])-stats['mean'])/stats['scale'];zg=decision('G7',geo,None);assert abs(zg-q['geometry_logit'])<1e-12
    if len(cells)<3:e=dict(valid=False,channels=None,winner=None)
    else:e=spatial_channel_features(reader.mesh,*maps,cells,old['query'][indices])
    score=decision('C34',geo,e['channels']) if e['valid'] else zg
    cases.append(dict(**case,old_logit=q['logit'],old_accepted=q['accepted'],new_logit=score,new_accepted=score>0,geometry_logit=zg,
        scale_control_logit=decision('C34_scale',geo,np.array(q['channels'])),common_cells=len(cells),full_cells=len(old['cells']),valid=e['valid'],winner=e['winner'],geometry_source=pose_source,
        scale=scale,offset=offset,not_used_for_fit=True,actual_IDs_evaluated=False))
old_errors=sum((r['old_accepted']!=(r['kind']=='own')) for r in cases);new_errors=sum((r['new_accepted']!=(r['kind']=='own')) for r in cases)
correct_retained=all(r['new_accepted']==r['old_accepted'] for r in cases if r['old_accepted']==(r['kind']=='own'))
gate=correct_retained and new_errors<old_errors
verify();save('deployment_screen.json',dict(passed=True,gate=gate,old_errors=old_errors,new_errors=new_errors,previous_correct_retained=correct_retained,records=cases,
    ready_geometry_reused=reused,missing_geometry_restored=restored,final_fits=4,independent_gradient_max=errors,no_core_or_video=True,old_models_not_refit=True,
    limits='Five predeclared already-viewed currentnativequeries. Same histories, not a new independent test. A corrected sign is not an emitted ID. No feature/window/threshold was chosen using these five outcomes.',goal_achieved=False))
print('MASK_DEPLOYMENT_SCREEN',gate,old_errors,new_errors,json.dumps(cast(cases)),flush=True)
