"""Bound execution passport, missing snapshots and one fixed matched comparison."""
import os,sys,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,23,54,tzinfo=MSK)
mode=sys.argv[1]
if mode!='restore':os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=min(3600,(STOP-datetime.now(MSK)).total_seconds()))
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cast(v):
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [cast(x) for x in v]
    if hasattr(v,'tolist'):return v.tolist()
    return v
def save(name,obj):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(cast(obj),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
def seqkey(video,seq):return hashlib.sha256(json.dumps([video,seq],separators=(',',':')).encode()).hexdigest()[:32]
p=read(OUT/'protocol_v2.json')
if mode=='register':
    verify(p);inputs=read(OUT/'inputs_complete.json');assert inputs['passed']
    inventory=read(OUT/'input_inventory.json')
    files=[Path(__file__),OUT/'protocol_v2.json',OUT/'inputs_complete.json',OUT/'input_recovery_protocol.json',OUT/'input_inventory.json',OUT/'protocol.json',
        ROOT/'tracking/z13_readout.py',ROOT/'tracking/flyvis_adapter.py',ROOT/'tracking/flyvis_motion.py',ROOT/'tracking/flyvis_activity.py',ROOT/'tracking/flyvis_checkpoint.py']
    files += list((ROOT/'tools/flyvis-runtime/flyvis/network').glob('*.py'))
    sources=dict(p['sources']);sources.update({f.relative_to(ROOT).as_posix():sha(f) for f in files})
    for key,item in inputs['snapshots'].items():assert sha(ROOT/item['path'])==item['sha256'];sources[item['path']]=item['sha256']
    for name,digest in inventory['sources'].items():assert sha(ROOT/name)==digest;sources[name]=digest
    save('execution_protocol.json',dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),sources=sources,
        inputs=inputs['snapshots'],inventory_requests=inventory['requests'],
        restore='Reuse ready C34/R1 snapshots exactly. For each case missing any core map, run one continuous frozen-core prefix and capture warm/end states. Compare any already-ready snapshot at the same causal prefix at tolerance1e-4; never substitute a different history. Restore only missing fields, no core training.',
        controls='Normalize raw34C and34R1 exactly as native. R134spatial =17EMAs plus their6neighbor means; R134temporal =17EMAs plus17transitions. Same support and fixed old shift selection for all.',
        mathematics='Recheck a stored ordinary snapshot from exact restored prefix before core use; after features, mean channel vector must equal old equal-channel score and shift; no new mathematical model.',
        gate=p['primary'],fit=p['fit'],deadline=STOP.isoformat()))
    print('EXECUTION_REGISTERED',len(inputs['snapshots']),flush=True);raise SystemExit(0)
phase=read(OUT/'execution_protocol.json');verify(phase)
import numpy as np
OTHER=[0,1,2,3]+list(range(12,34))
def arrays(key):
    values={}
    for item in phase['inventory_requests'].get(key,{}).get('ready_files',[]):
        with np.load(ROOT/item['path'],allow_pickle=False) as z:value=z[item['array']].copy()
        slot=item['slot']
        if slot in values:np.testing.assert_array_equal(values[slot],value)
        values[slot]=value
    path=OUT/'maps'/(key+'.npz')
    if path.exists():
        assert read(path.with_suffix('.json'))['sha256']==sha(path)
        with np.load(path,allow_pickle=False) as z:
            for slot in z.files:assert slot not in values;values[slot]=z[slot].copy()
    if 'C34' not in values and 'C8' in values and 'C_OTHER' in values:
        c=np.empty((721,34),np.float32);c[:,4:12]=values['C8'];c[:,OTHER]=values['C_OTHER'];values['C34']=c
    return values
if mode=='restore':
    import torch
    from tracking.gpu_lock import GpuLock
    from tracking.flyvis_adapter import activate_flyvis
    from scripts.z13_target_model import build_arm,receptor_features_rich
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    norm={name:read(ROOT/f'runs/z13_target_center_dev_20261018/{name}/normalization.json') for name in ['C','R1']}
    cm,cs=[np.array(norm['C'][n])[:34] for n in ['mean','sd']];rm,rs=[np.array(norm['R1'][n])[:34] for n in ['mean','sd']]
    # Before GPU/core work, one independent exact archived R1 witness from the recovered input.
    witness=next(k for k in phase['inputs'] if 'R1' in arrays(k))
    hx=np.load(ROOT/phase['inputs'][witness]['path'],allow_pickle=False)
    r=((receptor_features_rich(hx[None].astype(float))[-1][0].T-rm)/rs).astype(np.float32)
    np.testing.assert_array_equal(r,arrays(witness)['R1'])
    save('input_math_witness.json',dict(passed=True,key=witness,ordinary_snapshot_exact=True,ready_input_sha256=phase['inputs'][witness]['sha256'],no_core_calls_before_witness=True))
    new=0;parity=[];bb=None;weights=None
    with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        for index,e in enumerate(p['edges']):
            video=e['video'];seq=e['sequence'];keys=[seqkey(video,seq[:4]),seqkey(video,seq)]
            pending=[k for k in keys if not {'C34','R1'}<=set(arrays(k))]
            if not pending:continue
            hx=np.load(ROOT/phase['inputs'][keys[1]]['path'],allow_pickle=False)
            assert np.array_equal(hx[:4],np.load(ROOT/phase['inputs'][keys[0]]['path'],allow_pickle=False))
            rich=receptor_features_rich(hx[None].astype(float));needcore=any('C34' not in arrays(k) for k in pending)
            if needcore:
                if bb is None:
                    activate_flyvis(device='cuda');_,_,bb=build_arm('C',torch,None)
                    assert bb.dev.type=='cuda' and not bb.net.training and bb.steps==2 and abs(bb.dt-1/60)<1e-12
                    weights={k:v.detach().clone() for k,v in bb.net.named_parameters()}
                x=torch.from_numpy(hx[None]).to(bb.dev);state=bb.init(x[:,:1])
            ends={3:keys[0],len(seq)-1:keys[1]}
            for t in range(1,len(seq)):
                if needcore:state,feature=bb.step(state,x[:,t-1],x[:,t])
                if t not in ends:continue
                key=ends[t];old=arrays(key);values={};errors={}
                ordinary=((rich[t][0].T-rm)/rs).astype(np.float32)
                if 'R1' in old:np.testing.assert_array_equal(ordinary,old['R1'])
                else:values['R1']=ordinary
                if needcore:
                    core=((feature[0].cpu().numpy().T.astype(float)-cm)/cs).astype(np.float32)
                    if 'C34' in old:errors['C34']=float(abs(core-old['C34']).max());assert errors['C34']<=1e-4
                    elif 'C8' in old:
                        errors['C8']=float(abs(core[:,4:12]-old['C8']).max());assert errors['C8']<=1e-4;values['C_OTHER']=core[:,OTHER]
                    else:values['C34']=core
                if values:
                    dest=OUT/'maps'/(key+'.npz');dest.parent.mkdir(exist_ok=True);assert not dest.exists()
                    with dest.open('xb') as f:np.savez_compressed(f,**values)
                    save('maps/'+key+'.json',dict(sha256=sha(dest),execution_sha256=sha(OUT/'execution_protocol.json'),errors=errors,video=video,sequence=seq[:t+1]))
                    new+=1
                parity.append(dict(key=key,errors=errors))
            if needcore:del x,state,feature
            print('MAP_RESTORE',index+1,'/26','new',new,flush=True)
        if bb is not None:assert all(torch.equal(v,weights[k]) for k,v in bb.net.named_parameters())
    for key in phase['inputs']:assert {'C34','R1'}<=set(arrays(key)),key
    verify(phase)
    files={path.relative_to(ROOT).as_posix():sha(path) for path in (OUT/'maps').glob('*') if path.is_file()}
    save('maps_complete.json',dict(passed=True,sources=files,total_snapshots=len(phase['inputs']),new_files=new,weights_unchanged=True,parity=parity,continuous_prefix_snapshots=True))
    print('MAPS_COMPLETE',new,flush=True)
elif mode=='extract':
    verify(read(OUT/'maps_complete.json'))
    from tracking.observed_spatial_reader import ObservedSpatialReader
    from tracking.full_template_correspondence import relative_template_plan
    from tracking.channel_identity_head import spatial_channel_features,causal_geometry
    from tracking.z13_readout import HexGeometry
    from scripts.z13_target_model import neighbor_aug
    geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720]);nb=HexGeometry(np.array(geom['points']),geom['eye_size']).neighbors
    records=[];parity=0.
    for index,e in enumerate(p['edges']):
        keys=[seqkey(e['video'],seq) for seq in [e['sequence'][:4],e['sequence']]];values=[arrays(k) for k in keys]
        full=relative_template_plan(reader,e['reference_box'],e['reference_crop'],e['current_box'],e['current_crop'],e['offsets'])
        old=relative_template_plan(reader,e['reference_box'],e['reference_crop'],e['current_box'],e['current_crop'],e['offsets'],clip_to_candidate=True)
        assert len(old['cells'])==e['old_cells'] and len(full['cells'])>=3
        signals=[]
        for value in values:
            r=value['R1'].astype(float);signals.append(dict(C34=value['C34'].astype(float),R134_temporal=r,R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T))
        variants={}
        for name in ['C34','R134_spatial','R134_temporal']:
            v=spatial_channel_features(reader.mesh,signals[0][name],signals[1][name],full['cells'],full['query'])
            witness=reader.compare(signals[0][name],signals[1][name],full)
            assert v['winner']==witness['winner'] and v['scores']==witness['scores']
            if v['valid']:parity=max(parity,abs(np.mean(v['channels'])-witness['score']))
            variants[name]=v
        geometry=causal_geometry(e['reference_box'],e['current_box'],e['original_cost'],e['frame']-e['reference_frame'],e['old_cells'])
        records.append(dict(video=e['video'],frame=e['frame'],old_id=e['old_id'],raw_id=e['raw_id'],label=e['kind'],geometry=geometry,variants=variants,full_cells=len(full['cells'])))
        print('EXTRACT',index+1,'/26',flush=True)
    assert parity<1e-12;verify(phase)
    save('features.json',dict(passed=True,records=records,scalar_and_shift_parity_max_error=parity,one_fixed_reader=True,no_core_changes=True))
elif mode=='fit':
    from tracking.channel_identity_head import geometry_transform,fit_ridge_logistic,decision_function
    rows=read(OUT/'features.json')['records'];groups=np.array([r['video'] for r in rows]);y=np.array([r['label']=='own' for r in rows],int);G=np.array([r['geometry'] for r in rows]);videos=sorted(set(groups));scores={n:np.full(len(rows),np.nan) for n in p['models']};fits=[]
    for video in videos:
        train=np.flatnonzero(groups!=video);test=np.flatnonzero(groups==video);assert set(y[train])=={0,1}
        gt,ge,stats=geometry_transform(G[train],G[test]);base=fit_ridge_logistic(gt,y[train]);z=decision_function(base,ge);scores['G7'][test]=z
        fits.append(dict(video=video,model='G7',fit=base,normalization=stats,training_videos=sorted(set(groups[train]))))
        for name in p['models'][1:]:
            valid=np.array([r['variants'][name]['valid'] for r in rows]);tr=train[valid[train]];te=test[valid[test]];scores[name][test]=z
            if not len(tr) or set(y[tr])!={0,1}:fits.append(dict(video=video,model=name,fallback='one_class_or_no_features'));continue
            x=np.column_stack([gt[valid[train]],np.array([rows[i]['variants'][name]['channels'] for i in tr])]);model=fit_ridge_logistic(x,y[tr])
            if len(te):scores[name][te]=decision_function(model,np.column_stack([ge[valid[test]],np.array([rows[i]['variants'][name]['channels'] for i in te])]))
            fits.append(dict(video=video,model=name,fit=model,training_videos=sorted(set(groups[tr])),fallbacks=int((~valid[test]).sum())))
        print('DOMAIN_FOLD',len({f['video'] for f in fits}),'/20',flush=True)
    assert all(np.isfinite(z).all() for z in scores.values())
    def metric(labels,z):
        own=labels==1;accepted=z>0;tp=int((own&accepted).sum());fp=int((~own&accepted).sum());pos=int(own.sum());neg=int((~own).sum())
        return dict(own_accept=tp,own_reject=pos-tp,foreign_accept=fp,foreign_reject=neg-fp,balanced_accuracy=.5*(tp/pos+(neg-fp)/neg) if pos and neg else None)
    metrics={n:metric(y,z) for n,z in scores.items()}
    a=metrics['C34'];b=metrics['G7'];gate=(a['own_accept']>=b['own_accept'] and a['foreign_accept']<=b['foreign_accept'] and a['balanced_accuracy']>b['balanced_accuracy'])
    rng=np.random.default_rng(20261109);byvideo=[np.flatnonzero(groups==v) for v in videos];contrasts={n:[] for n in ['G7','R134_spatial','R134_temporal']}
    for _ in range(4000):
        ii=np.concatenate([byvideo[k] for k in rng.integers(len(videos),size=len(videos))])
        if len(set(y[ii]))<2:continue
        target=metric(y[ii],scores['C34'][ii])['balanced_accuracy']
        for n in contrasts:contrasts[n].append(target-metric(y[ii],scores[n][ii])['balanced_accuracy'])
    intervals={n:dict(point=a['balanced_accuracy']-metrics[n]['balanced_accuracy'],ci95=np.percentile(v,[2.5,97.5]),draws=len(v)) for n,v in contrasts.items()}
    verify(phase)
    save('predictions.json',dict(passed=True,records=[dict(video=r['video'],frame=r['frame'],old_id=r['old_id'],raw_id=r['raw_id'],label=r['label'],scores={n:z[i] for n,z in scores.items()}) for i,r in enumerate(rows)],folds=fits))
    save('summary.json',dict(passed=True,gate=gate,edges=26,own=20,foreign=6,videos=20,metrics=metrics,paired_video_intervals=intervals,zero_threshold_tuning=True,actual_IDs_evaluated=False,
        interpretation='Fixed development leave-video-out admission only. No native trajectory gain or unique FlyVis benefit established. Stop this version if gatefalse.'))
    print('DOMAIN_MATCHED_RESULT',gate,json.dumps(metrics),flush=True)
elif mode=='deploy':
    assert read(OUT/'summary.json')['gate'] and read(OUT/'audit.json')['passed']
    from tracking.channel_identity_head import geometry_transform,fit_ridge_logistic
    from scipy.special import expit
    rows=read(OUT/'features.json')['records'];G=np.array([r['geometry'] for r in rows]);y=np.array([r['label']=='own' for r in rows],int)
    g,_,normalization=geometry_transform(G,G);models={};errors={}
    for name in p['models']:
        if name=='G7':x=g
        else:
            valid=np.array([r['variants'][name]['valid'] for r in rows]);assert valid.all(),'Unexpected undefined map: preserve cohort and resolve explicit deployment fallback before proceeding'
            x=np.c_[g,np.array([r['variants'][name]['channels'] for r in rows])]
        model=fit_ridge_logistic(x,y);theta=np.array(model['theta']);z=x@theta[:-1]+theta[-1];sw=len(y)/(2*np.bincount(y)[y]);residual=sw*(expit(z)-y)
        gradient=np.r_[x.T@residual/len(y)+theta[:-1],residual.mean()];error=float(abs(gradient).max());assert error<1e-7
        models[name]=model;errors[name]=error
    verify(phase)
    save('deployment.json',dict(passed=True,geometry_normalization=normalization,models=models,pairs=26,videos=20,own=20,foreign=6,
        threshold=0,regularization=1,independent_gradient_max=errors,execution_sha256=sha(OUT/'execution_protocol.json'),
        no_validation_labels=True,no_hyperparameter_selection=True,actual_IDs_evaluated=False))
    print('DOMAIN_DEPLOYMENT_READY',errors,flush=True)
else:raise ValueError(mode)
