"""Fixed local-correspondence readout on the exact ready 26-pair cohort."""
import os,sys,json,hashlib,ast
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;TRAIN=ROOT/'research_domain_matched_reader_20261109'
mode=sys.argv[1];os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
STOP=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)));guard_entrypoint(timeout=min(1800,(STOP-datetime.now(STOP.tzinfo)).total_seconds()))
import psutil,numpy as np
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cast(x):
    if isinstance(x,dict):return {k:cast(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [cast(v) for v in x]
    return x.tolist() if hasattr(x,'tolist') else x
def save(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(cast(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
p=read(OUT/'protocol.json')
def verify():
    for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
verify();assert read(OUT/'preflight.json')['passed'];arms=['C34','R134_spatial','R134_temporal'];oldrows=read(TRAIN/'features.json')['records']
if mode=='extract':
    from tracking.observed_spatial_reader import ObservedSpatialReader
    from tracking.full_template_correspondence import relative_template_plan
    from tracking.partitioned_channel_correspondence import partitioned_channel_features,parts_at_match
    from tracking.hex_linear_interpolation import sample_linear
    from tracking.z13_readout import HexGeometry
    from scripts.z13_target_model import neighbor_aug
    geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720]);nb=HexGeometry(np.array(geom['points']),geom['eye_size']).neighbors
    tree=ast.parse((TRAIN/'compute.py').read_text(encoding='utf8'));function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='arrays')
    ns=dict(ROOT=ROOT,OUT=TRAIN,phase=read(TRAIN/'execution_protocol.json'),read=read,sha=sha,np=np,OTHER=[0,1,2,3]+list(range(12,34)))
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(TRAIN/'compute.py'),'exec'),ns)
    records=[];undefined=[]
    for edge,old in zip(read(TRAIN/'protocol_v2.json')['edges'],oldrows):
        assert all(edge[k]==old[k] for k in ['video','frame','old_id','raw_id'])
        plan=relative_template_plan(reader,edge['reference_box'],edge['reference_crop'],edge['current_box'],edge['current_crop'],edge['offsets']);assert len(plan['cells'])==old['full_cells']
        snapshots=[]
        for sequence in [edge['sequence'][:4],edge['sequence']]:
            key=hashlib.sha256(json.dumps([edge['video'],sequence],separators=(',',':')).encode()).hexdigest()[:32];value=ns['arrays'](key);r=value['R1'].astype(float)
            snapshots.append(dict(C34=value['C34'].astype(float),R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T))
        variants={}
        for arm in arms:
            baseline=old['variants'][arm];value=partitioned_channel_features(reader.mesh,snapshots[0][arm],snapshots[1][arm],plan['cells'],plan['query'],edge['reference_box'],baseline)
            if not value['valid']:undefined.append(dict(video=old['video'],frame=old['frame'],arm=arm,reason=value['reason']))
            else:
                assert value['winner']==baseline['winner'];np.testing.assert_allclose(value['channels'][:34]*np.sqrt(3),baseline['channels'],atol=1e-14,rtol=0)
                assert sorted(np.concatenate(value['groups']).tolist())==sorted(plan['cells'].tolist())
            permutation=np.random.default_rng(20261109).permutation(len(plan['cells']));middle=len(permutation)//2;mixed_groups=[permutation[:middle],permutation[middle:]]
            if len(plan['cells'])<6:mixed=dict(valid=False,reason='fewer_than_six_cells',channels=None)
            else:mixed=parts_at_match(snapshots[0][arm][plan['cells']],sample_linear(reader.mesh,snapshots[1][arm],plan['query'][:,baseline['winner']]),mixed_groups,baseline['channels'])
            if not mixed['valid']:undefined.append(dict(video=old['video'],frame=old['frame'],arm=arm,reason='mixed_control_'+mixed['reason']))
            value['mixed']=mixed;value['mixed_groups']=[plan['cells'][g] for g in mixed_groups];variants[arm]=value
        records.append(dict(**{k:old[k] for k in ['video','frame','old_id','raw_id','label','geometry','full_cells']},variants=variants))
        print('LOCAL_PARTS',len(records),'/26',flush=True)
    verify();save('features.json',dict(passed=True,records=records,global_features_and_winners_reused=78,new_maps=0,new_core_calls=0))
    save('readiness.json',dict(passed=not undefined,undefined=undefined,all26retained=len(records)==26,no_zero_imputation=True,minimum_full_cells=min(r['full_cells'] for r in records),minimum_part_cells=min(len(g) for r in records for v in r['variants'].values() for g in v.get('groups',[]))))
elif mode=='fit':
    assert read(OUT/'readiness.json')['passed']
    from tracking.channel_identity_head import fit_ridge_logistic,decision_function
    rows=read(OUT/'features.json')['records'];groups=np.array([r['video'] for r in rows]);videos=sorted(set(groups));y=np.array([r['label']=='own' for r in rows],int);G=np.array([r['geometry'] for r in rows]);oldpred=read(TRAIN/'predictions.json');norms={f['video']:f['normalization'] for f in oldpred['folds'] if f['model']=='G7'}
    scores={arm+'__'+kind:np.full(len(rows),np.nan) for arm in arms for kind in ['parts','mixed_parts','scale_only']};fits=[]
    for video in videos:
        tr=np.flatnonzero(groups!=video);te=np.flatnonzero(groups==video);assert set(y[tr])=={0,1};stats=norms[video];g=(G-stats['mean'])/stats['scale']
        for arm in arms:
            full=np.array([r['variants'][arm]['channels'] for r in oldrows]);new=np.array([r['variants'][arm]['channels'] for r in rows]);sd=full[tr].std(0);target=np.sqrt((new[tr].std(0).reshape(3,34)**2).sum(0));ratio=np.divide(target,sd,out=np.ones_like(sd),where=sd>0)
            mixed=np.array([r['variants'][arm]['mixed']['channels'] for r in rows])
            for kind,value in [('parts',new),('mixed_parts',mixed),('scale_only',full*ratio)]:
                x=np.c_[g,value];model=fit_ridge_logistic(x[tr],y[tr]);name=arm+'__'+kind;scores[name][te]=decision_function(model,x[te]);fits.append(dict(video=video,model=name,fit=model,normalization=stats,training_videos=sorted(set(groups[tr])),scale_ratio=ratio if kind=='scale_only' else None,constant_global_local_variance=np.flatnonzero((sd==0)&(target>0))))
        print('LOCAL_FOLD',len(fits)//9,'/20',flush=True)
    def metric(z):
        own=y==1;yes=z>0;a=int((own&yes).sum());b=int((~own&yes).sum());return dict(own_accept=a,foreign_accept=b,balanced_accuracy=.5*(a/20+(6-b)/6))
    assert all(np.isfinite(v).all() for v in scores.values());metrics={n:metric(v) for n,v in scores.items()};baselines=read(TRAIN/'summary.json')['metrics']
    def better(a,b):return a['own_accept']>=b['own_accept'] and a['foreign_accept']<=b['foreign_accept'] and a['balanced_accuracy']>b['balanced_accuracy']
    gates={arm:all(better(metrics[arm+'__parts'],control) for control in [baselines[arm],metrics[arm+'__scale_only'],metrics[arm+'__mixed_parts']]) for arm in arms}
    verify();save('predictions.json',dict(passed=True,records=[dict(**{k:r[k] for k in ['video','frame','old_id','raw_id','label']},scores={n:v[i] for n,v in scores.items()}) for i,r in enumerate(rows)],folds=fits))
    save('summary.json',dict(passed=True,metrics=metrics,ready_baselines=baselines,gates=gates,any_gate=any(gates.values()),fits=len(fits),duplicate_control_reused_by_proof=True,actual_IDs_evaluated=False,goal_achieved=False))
    print('LOCAL_PART_RESULT',json.dumps(metrics),json.dumps(gates),flush=True)
else:raise ValueError(mode)
