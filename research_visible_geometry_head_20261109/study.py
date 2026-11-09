"""Matched training of visible support and full-object coordinates on ready maps."""
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
    from tracking.geometric_template_correspondence import geometric_template_plan
    from tracking.channel_identity_head import spatial_channel_features
    from tracking.z13_readout import HexGeometry
    from scripts.z13_target_model import neighbor_aug
    geometry=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geometry['points'],geometry['eye_size'],[1280,720]);nb=HexGeometry(np.array(geometry['points']),geometry['eye_size']).neighbors
    tree=ast.parse((TRAIN/'compute.py').read_text(encoding='utf8'));function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='arrays')
    namespace=dict(ROOT=ROOT,OUT=TRAIN,phase=read(TRAIN/'execution_protocol.json'),read=read,sha=sha,np=np,OTHER=[0,1,2,3]+list(range(12,34)))
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(TRAIN/'compute.py'),'exec'),namespace)
    records=[];unsupported=[];reused=0
    for e,old in zip(read(TRAIN/'protocol_v2.json')['edges'],oldrows):
        assert (e['video'],e['frame'],e['old_id'],e['raw_id'])==(old['video'],old['frame'],old['old_id'],old['raw_id']) and e['geometry']['valid']
        full=relative_template_plan(reader,e['reference_box'],e['reference_crop'],e['current_box'],e['current_crop'],e['offsets'])
        aligned=geometric_template_plan(reader,e['reference_box'],e['reference_crop'],e['current_box'],e['current_crop'],e['offsets'],e['geometry']['scale'],e['geometry']['offset'])
        assert len(full['cells'])==old['full_cells'] and len(aligned['cells'])==e['aligned_cells']
        cells=np.intersect1d(full['cells'],aligned['cells']);oi=np.searchsorted(full['cells'],cells);ai=np.searchsorted(aligned['cells'],cells)
        assert np.array_equal(full['cells'][oi],cells) and np.array_equal(aligned['cells'][ai],cells)
        rec=dict(video=old['video'],frame=old['frame'],old_id=old['old_id'],raw_id=old['raw_id'],label=old['label'],geometry=old['geometry'],old_full_cells=len(full['cells']),visible_cells=len(aligned['cells']),common_cells=cells,variants={})
        if len(cells)<3:unsupported.append(dict(video=e['video'],frame=e['frame'],reason='less_than3commoncells'));records.append(rec);continue
        values=[namespace['arrays'](seqkey(e['video'],s)) for s in [e['sequence'][:4],e['sequence']]];signals=[]
        for value in values:
            r=value['R1'].astype(float);signals.append(dict(C34=value['C34'].astype(float),R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T))
        for name in names:
            if np.array_equal(cells,full['cells']):mask=old['variants'][name];reused+=1
            else:mask=spatial_channel_features(reader.mesh,signals[0][name],signals[1][name],cells,full['query'][oi])
            if np.array_equal(full['query'][oi],aligned['query'][ai]):changed=mask;reused+=1
            else:changed=spatial_channel_features(reader.mesh,signals[0][name],signals[1][name],cells,aligned['query'][ai])
            for kind,value in [('mask_only',mask),('aligned',changed)]:
                if not value['valid']:unsupported.append(dict(video=e['video'],frame=e['frame'],arm=name,kind=kind,reason='undefined_spatial_profile'))
            rec['variants'][name]=dict(mask_only=mask,aligned=changed)
        records.append(rec);print('VISIBLE_COORDINATES',len(records),'/26','cells',len(cells),flush=True)
    verify();save('features.json',dict(passed=True,records=records,old_features_reused=reused,common_support_fixed_across19shifts=True,no_core_or_video=True))
    save('readiness.json',dict(passed=not unsupported,unsupported=unsupported,all26retained=True,no_zero_imputation=True,minimum_common_cells=min(len(r['common_cells']) for r in records)))
elif mode=='fit':
    assert read(OUT/'readiness.json')['passed']
    from tracking.channel_identity_head import fit_ridge_logistic,decision_function
    rows=read(OUT/'features.json')['records'];groups=np.array([r['video'] for r in rows]);y=np.array([r['label']=='own' for r in rows],int);G=np.array([r['geometry'] for r in rows]);videos=sorted(set(groups));oldpred=read(TRAIN/'predictions.json');norms={r['video']:r['normalization'] for r in oldpred['folds'] if r['model']=='G7'}
    kinds=['mask_only','aligned','mask_scale_only','aligned_scale_only'];scores={n+'__'+k:np.full(len(rows),np.nan) for n in names for k in kinds};fits=[]
    for video in videos:
        tr=np.flatnonzero(groups!=video);te=np.flatnonzero(groups==video);assert set(y[tr])=={0,1};stats=norms[video];g=(G-np.array(stats['mean']))/np.array(stats['scale'])
        for name in names:
            raw=np.array([r['variants'][name]['channels'] for r in oldrows]);oldsd=raw[tr].std(0)
            for kind in kinds:
                target='mask_only' if kind.startswith('mask') else 'aligned';value=np.array([r['variants'][name][target]['channels'] for r in rows]);ratio=None
                if kind.endswith('scale_only'):
                    sd=value[tr].std(0);ratio=np.divide(sd,oldsd,out=np.ones_like(oldsd),where=oldsd>0);value=raw*ratio
                matrix=np.c_[g,value];model=fit_ridge_logistic(matrix[tr],y[tr]);key=name+'__'+kind;scores[key][te]=decision_function(model,matrix[te]);fits.append(dict(video=video,model=key,fit=model,normalization=stats,training_videos=sorted(set(groups[tr])),scale_ratio=ratio))
        print('GEOMETRY_FOLD',len(fits)//12,'/20',flush=True)
    def metric(z):
        own=y==1;yes=z>0;a=int((own&yes).sum());b=int((~own&yes).sum());return dict(own_accept=a,foreign_accept=b,balanced_accuracy=.5*(a/20+(6-b)/6))
    assert all(np.isfinite(z).all() for z in scores.values());metrics={n:metric(z) for n,z in scores.items()};baseline=read(TRAIN/'summary.json')['metrics'];gates={};warp_added={}
    def better(a,b):return a['own_accept']>=b['own_accept'] and a['foreign_accept']<=b['foreign_accept'] and a['balanced_accuracy']>b['balanced_accuracy']
    for name in names:
        for kind,scale in [('mask_only','mask_scale_only'),('aligned','aligned_scale_only')]:gates[name+'__'+kind]=better(metrics[name+'__'+kind],baseline[name]) and better(metrics[name+'__'+kind],metrics[name+'__'+scale])
        warp_added[name]=better(metrics[name+'__aligned'],metrics[name+'__mask_only'])
    verify();save('predictions.json',dict(passed=True,records=[dict(video=r['video'],frame=r['frame'],old_id=r['old_id'],raw_id=r['raw_id'],label=r['label'],scores={n:z[i] for n,z in scores.items()}) for i,r in enumerate(rows)],folds=fits))
    save('summary.json',dict(passed=True,metrics=metrics,ready_baselines=baseline,gates=gates,warp_added=warp_added,any_gate=any(gates.values()),fits=len(fits),old_models_not_refit=True,actual_IDs_evaluated=False,goal_achieved=False))
    print('VISIBLE_GEOMETRY_RESULT',json.dumps(metrics),json.dumps(gates),flush=True)
else:raise ValueError(mode)
