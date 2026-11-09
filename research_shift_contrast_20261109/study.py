"""New CPU readout features and folds on existing exact maps; no neural calls."""
import os,sys,json,hashlib,ast,gzip
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;TRAIN=ROOT/'research_domain_matched_reader_20261109';NATIVE=ROOT/'runs/domain_matched_native_20261109'
mode=sys.argv[1];os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)));guard_entrypoint(timeout=min(1800,(STOP-datetime.now(STOP.tzinfo)).total_seconds()))
import psutil,numpy as np
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cast(v):
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [cast(x) for x in v]
    return v.tolist() if hasattr(v,'tolist') else v
def save(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
p=read(OUT/'protocol.json')
def verify():
    for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
verify();names=['C34','R134_spatial','R134_temporal'];oldrows=read(TRAIN/'features.json')['records']
def seqkey(video,sequence):return hashlib.sha256(json.dumps([video,sequence],separators=(',',':')).encode()).hexdigest()[:32]
if mode=='extract':
    from tracking.observed_spatial_reader import ObservedSpatialReader
    from tracking.full_template_correspondence import relative_template_plan
    from tracking.z13_readout import HexGeometry
    from tracking.shift_profile_contrast import recover_missing_profiles
    from scripts.z13_target_model import neighbor_aug
    geometry=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geometry['points'],geometry['eye_size'],[1280,720]);nb=HexGeometry(np.array(geometry['points']),geometry['eye_size']).neighbors
    # Reuse only the pure saved-array resolver, never execute old restore/extract/fit.
    tree=ast.parse((TRAIN/'compute.py').read_text(encoding='utf8'));function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='arrays')
    namespace=dict(ROOT=ROOT,OUT=TRAIN,phase=read(TRAIN/'execution_protocol.json'),read=read,sha=sha,np=np,OTHER=[0,1,2,3]+list(range(12,34)))
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(TRAIN/'compute.py'),'exec'),namespace)
    records=[];maximum=0.
    for e,old in zip(read(TRAIN/'protocol_v2.json')['edges'],oldrows):
        assert (e['video'],e['frame'],e['old_id'],e['raw_id'])==(old['video'],old['frame'],old['old_id'],old['raw_id'])
        values=[namespace['arrays'](seqkey(e['video'],s)) for s in [e['sequence'][:4],e['sequence']]]
        plan=relative_template_plan(reader,e['reference_box'],e['reference_crop'],e['current_box'],e['current_crop'],e['offsets'])
        assert len(plan['cells'])==old['full_cells'];signals=[]
        for value in values:
            r=value['R1'].astype(float);signals.append(dict(C34=value['C34'].astype(float),R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T))
        variants={}
        for name in names:
            q=recover_missing_profiles(reader.mesh,signals[0][name],signals[1][name],plan['cells'],plan['query'],old['variants'][name]);maximum=max(maximum,q['parity_error']);variants[name]=q
        records.append(dict(video=old['video'],frame=old['frame'],old_id=old['old_id'],raw_id=old['raw_id'],label=old['label'],geometry=old['geometry'],variants=variants))
        print('PROFILE',len(records),'/26',flush=True)
    diagnostic=[]
    for case in p['diagnostic_cases']:
        video=case['video']
        with gzip.open(NATIVE/'rollouts/C34'/(video+'.events.json.gz'),'rt',encoding='utf8') as f:data=json.load(f)
        event=next(x for x in data['events'] if x['frame']==case['frame']);q=next(x for x in event['proposals'] if x['raw_id']==case['raw_id'] and x['old_id']==case['old_id'])
        sequence=[[h['frame'],h['crop']] for h in q['history']];arrays=[]
        for s in [sequence,sequence+q['search']]:arrays.append(np.load(NATIVE/'features'/video/(seqkey(video,s)+'.C34.npy'),allow_pickle=False))
        rc=np.array(q['reference_crop']);ct=np.array(q['current_crop']);axes=np.array(p['axes']);offsets=(axes@np.array([[13.,6.5],[0.,13.]]))*np.maximum((rc[2:]-rc[:2])/geometry['eye_size'],(ct[2:]-ct[:2])/geometry['eye_size'])
        plan=relative_template_plan(reader,q['reference_box'],rc,q['current_box'],ct,offsets)
        ready=dict(valid=q['evidence_valid'],winner=q['winner'],scores=q['scores'],channels=q['channels'])
        profile=recover_missing_profiles(reader.mesh,*arrays,plan['cells'],plan['query'],ready)
        diagnostic.append(dict(**case,profile=profile,old_logit=q['logit'],geometry=q['geometry'],not_used_for_fit_or_gate=True))
    verify();save('profiles.json',dict(passed=True,records=records,diagnostic=diagnostic,scalar_parity_max_error=maximum,no_core_or_video=True,ready_winning_vectors_reused=True,new_per_profile_vectors=18))
elif mode=='fit':
    from tracking.channel_identity_head import fit_ridge_logistic,decision_function
    rows=read(OUT/'profiles.json')['records'];oldpred=read(TRAIN/'predictions.json');groups=np.array([r['video'] for r in rows]);y=np.array([r['label']=='own' for r in rows],int);G=np.array([r['geometry'] for r in rows]);videos=sorted(set(groups))
    scores={n+s:np.full(len(rows),np.nan) for n in names for s in ['_contrast','_scale_only']};fits=[]
    norms={r['video']:r['normalization'] for r in oldpred['folds'] if r['model']=='G7'}
    for video in videos:
        tr=np.flatnonzero(groups!=video);te=np.flatnonzero(groups==video);assert set(y[tr])=={0,1};stats=norms[video];g=(G-np.array(stats['mean']))/np.array(stats['scale'])
        for name in names:
            delta=np.array([r['variants'][name]['contrast'] for r in rows]);raw=np.array([r['variants'][name]['channels'] for r in oldrows]);oldsd=raw[tr].std(0);newsd=delta[tr].std(0);ratio=np.divide(newsd,oldsd,out=np.ones_like(oldsd),where=oldsd>0)
            for suffix,features in [('_contrast',delta),('_scale_only',raw*ratio)]:
                matrix=np.c_[g,features];model=fit_ridge_logistic(matrix[tr],y[tr]);key=name+suffix;scores[key][te]=decision_function(model,matrix[te]);fits.append(dict(video=video,model=key,fit=model,normalization=stats,training_videos=sorted(set(groups[tr])),scale_ratio=ratio if suffix=='_scale_only' else None,unmatchable_constant_raw_channels=np.flatnonzero((oldsd==0)&(newsd>0)).tolist()))
        print('CONTRAST_FOLD',len(fits)//6,'/20',flush=True)
    def metric(z):
        own=y==1;yes=z>0;a=int((own&yes).sum());b=int((~own&yes).sum());return dict(own_accept=a,foreign_accept=b,balanced_accuracy=.5*(a/20+(6-b)/6))
    assert all(np.isfinite(z).all() for z in scores.values());metrics={n:metric(z) for n,z in scores.items()};baseline=read(TRAIN/'summary.json')['metrics'];gates={}
    for name in names:
        a=metrics[name+'_contrast'];others=[baseline[name],metrics[name+'_scale_only']]
        gates[name]=all(a['own_accept']>=b['own_accept'] and a['foreign_accept']<=b['foreign_accept'] and a['balanced_accuracy']>b['balanced_accuracy'] for b in others)
    verify();save('predictions.json',dict(passed=True,records=[dict(video=r['video'],frame=r['frame'],old_id=r['old_id'],raw_id=r['raw_id'],label=r['label'],scores={n:z[i] for n,z in scores.items()}) for i,r in enumerate(rows)],folds=fits))
    save('summary.json',dict(passed=True,metrics=metrics,ready_baselines=baseline,gates=gates,any_gate=any(gates.values()),fits=len(fits),old_models_not_refit=True,diagnostic_cases_not_in_gate=True,actual_IDs_evaluated=False,goal_achieved=False))
    print('SHIFT_CONTRAST_RESULT',json.dumps(metrics),json.dumps(gates),flush=True)
else:raise ValueError(mode)
