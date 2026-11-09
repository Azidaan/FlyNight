"""Paired mean/variance agreement on saved maps, at saved old winning shifts."""
import os,sys,json,hashlib,ast,inspect,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;TRAIN=ROOT/'research_domain_matched_reader_20261109';SCALE=ROOT/'research_signal_scaling_20261109'
mode=sys.argv[1];os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
stop=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)));guard_entrypoint(timeout=min(1800,(stop-datetime.now(stop.tzinfo)).total_seconds()))
import psutil,numpy as np
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cast(x):
    if isinstance(x,dict):return {k:cast(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [cast(v) for v in x]
    return x.tolist() if hasattr(x,'tolist') else x
def save(n,v):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(cast(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
p=read(OUT/'protocol.json')
def verify():
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
verify();assert read(OUT/'preflight.json')['passed'];arms=['C34','R134_spatial','R134_temporal'];oldrows=read(TRAIN/'features.json')['records']
if mode=='extract':
    from tracking.observed_spatial_reader import ObservedSpatialReader
    from tracking.full_template_correspondence import relative_template_plan
    from tracking.hex_linear_interpolation import sample_linear
    from tracking.channel_agreement_correspondence import agreement_channels
    from tracking.z13_readout import HexGeometry
    from scripts.z13_target_model import neighbor_aug
    geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720]);nb=HexGeometry(np.array(geom['points']),geom['eye_size']).neighbors
    function=next(n for n in ast.parse((TRAIN/'compute.py').read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='arrays')
    ns=dict(ROOT=ROOT,OUT=TRAIN,phase=read(TRAIN/'execution_protocol.json'),read=read,sha=sha,np=np,OTHER=[0,1,2,3]+list(range(12,34)));exec(compile(ast.Module(body=[function],type_ignores=[]),str(TRAIN/'compute.py'),'exec'),ns)
    records=[];undefined=[]
    for edge,old in zip(read(TRAIN/'protocol_v2.json')['edges'],oldrows):
        assert all(edge[k]==old[k] for k in ['video','frame','old_id','raw_id']);plan=relative_template_plan(reader,edge['reference_box'],edge['reference_crop'],edge['current_box'],edge['current_crop'],edge['offsets']);assert len(plan['cells'])==old['full_cells']
        maps=[]
        for seq in [edge['sequence'][:4],edge['sequence']]:
            key=hashlib.sha256(json.dumps([edge['video'],seq],separators=(',',':')).encode()).hexdigest()[:32];v=ns['arrays'](key);r=v['R1'].astype(float);maps.append(dict(C34=v['C34'].astype(float),R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T))
        variants={}
        for arm in arms:
            winner=old['variants'][arm]['winner'];a=maps[0][arm][plan['cells']];b=sample_linear(reader.mesh,maps[1][arm],plan['query'][:,winner]);values={k:agreement_channels(a,b,mean_penalty=k=='agreement') for k in ['agreement','variance_only']}
            if any(v is None for v in values.values()):undefined.append(dict(video=old['video'],frame=old['frame'],arm=arm))
            variants[arm]=dict(winner=winner,**values)
        records.append(dict(**{k:old[k] for k in ['video','frame','old_id','raw_id','label','geometry','full_cells']},variants=variants));print('AGREEMENT_EXTRACT',len(records),'/26',flush=True)
    verify();save('features.json',dict(passed=True,records=records,old_winners_and_support_reused=True,no_new_maps=True));save('readiness.json',dict(passed=not undefined,undefined=undefined,all26retained=len(records)==26,no_imputation=True))
elif mode=='fit':
    assert read(OUT/'readiness.json')['passed']
    import tracking.channel_identity_head as h
    # Same previously established precision termination; original production
    # source/objective stays unchanged and every fit is checkpointed.
    text=inspect.getsource(h.fit_ridge_logistic);old="    raise ArithmeticError('Fixed Newton iteration limit reached')"
    replacement="""    loss, gradient, _ = objective(theta,x,y,sw)
    residual=float(np.max(np.abs(gradient)))
    if residual<=1e-7:
        return dict(theta=theta,loss=loss,gradient_max=residual,iterations=100,class_counts=counts,threshold=0.,regularization=1.)
    raise ArithmeticError('Iteration limit outside registered gradient tolerance')"""
    assert text.count(old)==1;solver=dict(h.__dict__);exec(compile(text.replace(old,replacement),'agreement_precision_solver.py','exec'),solver)
    rows=read(OUT/'features.json')['records'];groups=np.array([r['video'] for r in rows]);y=np.array([r['label']=='own' for r in rows],int);G=np.array([r['geometry'] for r in rows]);norms={f['video']:f['normalization'] for f in read(TRAIN/'predictions.json')['folds'] if f['model']=='G7'}
    cache_path=OUT/'checkpoint_fits.json';cache=read(cache_path) if cache_path.exists() else {};fits=[];scores={a+'__'+k:np.full(len(rows),np.nan) for a in arms for k in ['agreement','variance_only']}
    for video in sorted(set(groups)):
        tr=np.flatnonzero(groups!=video);te=np.flatnonzero(groups==video);stats=norms[video];g=(G-stats['mean'])/stats['scale']
        for arm in arms:
            for kind in ['agreement','variance_only']:
                raw=np.array([r['variants'][arm][kind] for r in rows]);mu=raw[tr].mean(0);sd=raw[tr].std(0);constant=np.ptp(raw[tr],axis=0)==0;assert (~constant).any();rms=float(np.sqrt(np.mean(sd[~constant]**2)));scale=np.where(constant,1.,rms);X=np.c_[g,(raw-mu)/scale];key='|'.join([video,arm,kind]);fingerprint=hashlib.sha256(X[tr].tobytes()+y[tr].tobytes()).hexdigest()
                if key in cache:assert cache[key]['input_sha256']==fingerprint;model=cache[key]['fit']
                else:
                    model=solver['fit_ridge_logistic'](X[tr],y[tr]);cache[key]=dict(input_sha256=fingerprint,fit=cast(model));temp=cache_path.with_suffix('.pending');temp.write_text(json.dumps(cache,indent=2,allow_nan=False),encoding='utf8');temp.replace(cache_path)
                name=arm+'__'+kind;scores[name][te]=h.decision_function(model,X[te]);fits.append(dict(video=video,model=name,fit=model,normalization=stats,signal_mean=mu,signal_scale=scale,training_videos=sorted(set(groups[tr]))))
        print('AGREEMENT_FOLD',len(fits)//6,'/20',flush=True)
    def metric(z):
        yes=z>0;own=y==1;a=int((yes&own).sum());b=int((yes&~own).sum());return dict(own_accept=a,foreign_accept=b,balanced_accuracy=.5*(a/20+(6-b)/6))
    assert all(np.isfinite(z).all() for z in scores.values());metrics={n:metric(z) for n,z in scores.items()};baseline={a:read(SCALE/'summary.json')['metrics'][a+'__common'] for a in arms}
    def better(a,b):return a['own_accept']>=b['own_accept'] and a['foreign_accept']<=b['foreign_accept'] and a['balanced_accuracy']>b['balanced_accuracy']
    gates={a+'__'+k:better(metrics[a+'__'+k],baseline[a]) for a in arms for k in ['agreement','variance_only']};mean_specific={a:better(metrics[a+'__agreement'],metrics[a+'__variance_only']) for a in arms}
    selection={a:('agreement' if gates[a+'__agreement'] and (not gates[a+'__variance_only'] or mean_specific[a]) else ('variance_only' if gates[a+'__variance_only'] else None)) for a in arms}
    verify();save('predictions.json',dict(passed=True,records=[dict(**{k:r[k] for k in ['video','frame','old_id','raw_id','label']},scores={n:z[i] for n,z in scores.items()}) for i,r in enumerate(rows)],folds=fits));save('summary.json',dict(passed=True,metrics=metrics,ready_baselines=baseline,gates=gates,mean_specific=mean_specific,selection=selection,any_gate=any(gates.values()),fits=len(fits),actual_IDs_evaluated=False,goal_achieved=False));print('AGREEMENT_RESULT',json.dumps(metrics),json.dumps(selection),flush=True)
else:raise ValueError(mode)
