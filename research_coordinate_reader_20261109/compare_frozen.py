"""One fixed geometric reader intervention with unchanged deployed heads."""
import os,sys,json,hashlib,gzip,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
NATIVE=ROOT/'runs/full_template_native_20261109';LEGACY=ROOT/'runs/identity_assignment_20261108/full_precision'
MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,23,54,tzinfo=MSK)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=min(1200,(STOP-datetime.now(MSK)).total_seconds()))
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(p):
    for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
def sequence_key(video,sequence):return hashlib.sha256(json.dumps([video,sequence],separators=(',',':')).encode()).hexdigest()[:32]
def view_key(frame,crop):return hashlib.sha256(json.dumps([int(frame),list(crop)],separators=(',',':')).encode()).hexdigest()[:32]
def evidence_key(video,row):
    value=[video,row['reference_box'],row['reference_crop'],row['current_box'],row['current_crop'],
           [[v['frame'],v['crop']] for v in row['history']],row['search']]
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()
mode=sys.argv[1]
if mode=='register':
    assert read(ROOT/'research_geometry_consistency_20261109/implementation_check.json')['passed']
    files=[Path(__file__),NATIVE/'deployment.json',NATIVE/'execution_protocol.json',NATIVE/'id_fate.json',
        ROOT/'research_geometry_consistency_20261109/coordinate_report.json',ROOT/'research_geometry_consistency_20261109/training_states_complete.json',
        ROOT/'tracking/geometric_template_correspondence.py',ROOT/'tracking/full_template_correspondence.py',ROOT/'tracking/channel_identity_head.py',
        ROOT/'tracking/channel_balanced_correspondence.py',ROOT/'tracking/hex_linear_interpolation.py',ROOT/'scripts/z13_target_model.py',
        ROOT/'runs/z13_local_peak_20261021/geometry.json',ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json']
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in files};queries=[];requests={};ready_old={}
    known_features={};known_views={}
    for folder in (NATIVE/'rollouts',LEGACY/'rollouts'):
        for receipt in folder.rglob('*.receipt.json'):
            if folder==LEGACY/'rollouts' and receipt.parent.name not in ['G_R1','G_FLYVIS','G_PERM']:continue
            sources[receipt.relative_to(ROOT).as_posix()]=sha(receipt)
            for name,digest in read(receipt).get('feature_files',{}).items():
                if name in known_features:assert known_features[name]==digest,name
                known_features[name]=digest
    for arm in ['C34','R134_spatial','R134_temporal']:
        for path in sorted((NATIVE/'rollouts'/arm).glob('*.events.json.gz')):
            sources[path.relative_to(ROOT).as_posix()]=sha(path);video=path.name.split('.events')[0]
            with gzip.open(path,'rt',encoding='utf8') as f:events=json.load(f)['events']
            for event in events:
                for row in event['proposals']:
                    if row['reason'] not in ['head_accept','head_refuse']:continue
                    ready_old.setdefault(arm,{})[evidence_key(video,row)]=dict(valid=row.get('evidence_valid',False),channels=row.get('channels'),winner=row.get('winner'))
                    if arm!='C34':continue
                    queries.append(dict(video=video,frame=event['frame'],row=row))
                    refseq=[[v['frame'],v['crop']] for v in row['history']]
                    for seq in [refseq,refseq+row['search']]:
                        key=sequence_key(video,seq);core=NATIVE/'features'/video/(key+'.C34.npy');meta=core.with_suffix('.json')
                        assert known_features[core.relative_to(ROOT).as_posix()]==sha(core)
                        m=read(meta);assert m['sha256']==sha(core) and m['sequence']==seq
                        sources[core.relative_to(ROOT).as_posix()]=sha(core);sources[meta.relative_to(ROOT).as_posix()]=sha(meta)
                        for name,digest in m['retinal_sources'].items():
                            if name in known_views:assert known_views[name]==digest,name
                            known_views[name]=digest
                        ordinary=next((path for path in [NATIVE/'features'/video/(key+'.R1.npy'),LEGACY/'features'/video/(key+'.R1.npy')] if path.exists()),None)
                        item=dict(video=video,sequence=seq,core=core.relative_to(ROOT).as_posix(),ordinary=None)
                        if ordinary is not None:
                            name=ordinary.relative_to(ROOT).as_posix();assert known_features[name]==sha(ordinary),name
                            item['ordinary']=name;sources[name]=sha(ordinary)
                        requests[key]=item
    assert len(queries)==213
    for key,item in requests.items():
        if item['ordinary'] is not None:continue
        views=[]
        for frame,crop in item['sequence']:
            name=view_key(frame,crop)+'.npy'
            path=next(p for p in [NATIVE/'retinal'/item['video']/name,LEGACY/'retinal'/item['video']/name] if p.exists())
            name=path.relative_to(ROOT).as_posix();assert known_views[name]==sha(path),name
            sources[name]=sha(path);views.append(name)
        item['views']=views
    save('protocol.json',dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),sources=sources,
        action='Replace only candidate-relative box stretching/support by the existing admission full-scale/offset and nominally visible fixed support. Freeze all deployed coefficients, G7features,19physicalshifts,normalization,lambda history,zero decision threshold and cost correction. No new training or core inference.',
        distinction='The missing gated foreign training class prevents a justified newly fitted conditional head. It does not make this fixed, unlearned correspondence intervention mathematically impossible. Test it explicitly with the already frozen heads; do not mix these two questions or claim the data gap caused the previous failure.',
        population='All213scoredrequests from the closed C34nativepilot, same past/current input histories for allthree representations. Already viewed20developmentvalvideos; post-result mechanism screen, not independent validation.',
        baselines='Reuse logged C34old channels and any exactly matching logged ordinary-input features. Compute missing matched-control old features only for previously unscored histories/plans. Never repeat ready core maps or old rollout/metric calculations.',
        ordinary='Use closed normalizedR1maps when available; otherwise compute ONLY the missing ordinary input features from the already saved exact retinal arrays. New arrays saved under this experiment with input hashes. No video decoding, retina, model/network inference.',
        controls=['G7_ready','C34_old','C34_aligned','R134_spatial_old','R134_spatial_aligned','R134_temporal_old','R134_temporal_aligned'],
        fallback='Aligned support<3 or undefined feature uses the unchanged G7logit in every signal arm. No candidate/pose/shift chosen by GT.',
        gate='Admit this frozen C34reader to an actual native pilot only if all originally accepted known own links remain accepted, at least one of the2knownownfalse-refusals becomes accepted, and known foreign accepts do not increase. This is an exploratory safety/admission screen; it is not proof of generalization or preservedID. If it fails stop this frozen-reader version without changing weights,pose rules,shifts or threshold.',
        future='If admitted, native full20video matched controls and original harm/uncertainty gates remain required; independent new final videos only after a useful actual-tracker pilot.',
        zero_witness='Before new aligned scores, reproduce old full-plan C34channelvectors and logits on first3queries in frozen file order, maxerror1e-10; ready disabled/empty native controls are reused because solver/lifecycle/G7remain unchanged. No generic replay of40readyzero runs.',
        deadline=STOP.isoformat(),queries=queries,requests=requests,ready_old=ready_old))
    print('REGISTERED_FROZEN_COORDINATE_READER',len(queries),len(requests),flush=True);raise SystemExit(0)
assert mode=='run';p=read(OUT/'protocol.json');verify(p)
import numpy as np,torch
torch.set_num_threads(2)
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.geometric_template_correspondence import geometric_template_plan
from tracking.full_template_correspondence import relative_template_plan
from tracking.channel_identity_head import spatial_channel_features,decision_function
from tracking.z13_readout import HexGeometry
from scripts.z13_target_model import receptor_features_rich,neighbor_aug
g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720])
neighbors=HexGeometry(np.array(g['points']),g['eye_size']).neighbors
axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2])@np.array([[13.,6.5],[0.,13.]])
deployed=read(NATIVE/'deployment.json');stats=deployed['geometry_normalization'];models=deployed['models']
norm=read(ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json');mean=np.array(norm['mean'])[:34];sd=np.array(norm['sd'])[:34]
poses={(r['video'],r['frame'],r['old_id'],r['raw_id']):r for r in read(ROOT/'research_geometry_consistency_20261109/coordinate_report.json')['records']}
fate={(r['video'],r['frame'],r['old_id'],r['raw_id']):r for r in read(NATIVE/'id_fate.json')['records'] if r['arm']=='C34'}
cache={};new_ordinary=0;reused_features=0;parity=[];records=[]
def maps(key,arm):
    global new_ordinary
    item=p['requests'][key]
    if (key,arm) in cache:return cache[key,arm]
    if arm=='C34':value=np.load(ROOT/item['core'],allow_pickle=False).astype(float)
    else:
        if item['ordinary'] is not None:value=np.load(ROOT/item['ordinary'],allow_pickle=False).astype(float)
        elif (key,'R1') in cache:value=cache[key,'R1']
        else:
            hx=np.stack([np.load(ROOT/name,allow_pickle=False) for name in item['views']]).astype(np.float32)
            raw=receptor_features_rich(hx[None].astype(np.float64))[-1][0].T
            value=((raw-mean)/sd).astype(np.float32).astype(float);new_ordinary+=1
            path=OUT/'ordinary_maps'/(key+'.npy');path.parent.mkdir(exist_ok=True)
            with path.open('xb') as f:np.save(f,value.astype(np.float32),allow_pickle=False)
            save('ordinary_maps/'+key+'.json',dict(sha256=sha(path),protocol_sha256=sha(OUT/'protocol.json'),views=item['views']))
        cache[key,'R1']=value
        if arm=='R134_spatial':value=neighbor_aug(value[:,:17].T[None],neighbors)[0].T
    cache[key,arm]=value;return value
def logit(arm,geo,channels):return float(decision_function(models[arm],np.r_[geo,channels][None])[0])
for index,item in enumerate(p['queries']):
    row=item['row'];video=item['video'];key=(video,item['frame'],row['old_id'],row['raw_id']);pose=poses[key]
    refseq=[[v['frame'],v['crop']] for v in row['history']];keys=[sequence_key(video,seq) for seq in [refseq,refseq+row['search']]]
    rc=np.array(row['reference_crop']);ct=np.array(row['current_crop'])
    shifts=axes*np.maximum((rc[2:]-rc[:2])/g['eye_size'],(ct[2:]-ct[:2])/g['eye_size'])
    old=relative_template_plan(reader,row['reference_box'],rc,row['current_box'],ct,shifts)
    aligned=geometric_template_plan(reader,row['reference_box'],rc,row['current_box'],ct,shifts,pose['scale_geom'],pose['offset_reference_pixels'])
    assert aligned['cells'].tolist()==pose['aligned_cells'] and len(old['cells'])==row['full_cells']
    geo=(np.array(row['geometry'])-stats['mean'])/stats['scale'];zg=float(row['geometry_logit'])
    scores={'G7_ready':zg};details={}
    for arm in ['C34','R134_spatial','R134_temporal']:
        a,b=[maps(k,arm) for k in keys]
        cached=p['ready_old'].get(arm,{}).get(evidence_key(video,row))
        if cached is not None:
            old_e=cached;reused_features+=1
        else:
            e=spatial_channel_features(reader.mesh,a,b,old['cells'],old['query'])
            old_e=dict(valid=e['valid'],channels=None if not e['valid'] else e['channels'].tolist(),winner=e['winner'])
        zold=logit(arm,geo,old_e['channels']) if old_e['valid'] else zg
        if arm=='C34':assert abs(zold-row['logit'])<1e-12
        if index<3 and arm=='C34':
            witness=spatial_channel_features(reader.mesh,a,b,old['cells'],old['query'])
            error=float(abs(witness['channels']-np.array(old_e['channels'])).max());assert error<1e-10
            assert witness['winner']==old_e['winner'];parity.append(error)
        if len(aligned['cells'])<3:e=dict(valid=False,winner=None,channels=None)
        else:e=spatial_channel_features(reader.mesh,a,b,aligned['cells'],aligned['query'])
        znew=logit(arm,geo,e['channels']) if e['valid'] else zg
        scores[arm+'_old']=zold;scores[arm+'_aligned']=znew
        details[arm]=dict(old_winner=old_e['winner'],aligned_winner=e['winner'],fallback=not e['valid'],
            aligned_channels=None if not e['valid'] else e['channels'].tolist())
    records.append(dict(video=video,frame=item['frame'],old_id=row['old_id'],raw_id=row['raw_id'],
        old_cells=len(old['cells']),aligned_cells=len(aligned['cells']),scores=scores,details=details))
# Evaluation labels are attached after all deterministic feature decisions.
for row in records:row['kind']=fate[(row['video'],row['frame'],row['old_id'],row['raw_id'])]['kind']
known=[r for r in records if r['kind']!='unknown'];assert len(known)==6
counts={arm:{kind:sum(r['kind']==kind and r['scores'][arm]>0 for r in known) for kind in ['own','foreign']} for arm in records[0]['scores']}
preserved=all(r['scores']['C34_aligned']>0 for r in known if r['kind']=='own' and r['scores']['C34_old']>0)
recovered=sum(r['kind']=='own' and r['scores']['C34_old']<=0<r['scores']['C34_aligned'] for r in known)
gate=preserved and recovered>=1 and counts['C34_aligned']['foreign']<=counts['C34_old']['foreign']
verify(p);assert not torch.cuda.is_initialized() and not any(x=='flyvis' or x.startswith('flyvis.') for x in sys.modules)
save('comparison.json',dict(passed=True,gate=gate,old_map_feature_parity=parity,new_ordinary_maps=new_ordinary,reused_old_feature_vectors=reused_features,
    records=records,known=known,accepted_known=counts,previous_accepted_own_preserved=preserved,old_own_false_refusals_recovered=recovered,
    unknown=len(records)-len(known),no_core_fits_threshold_search_or_rollouts=True,
    limit='Fixed-vector post-result screen, not actual IDs or independent evidence. CommonC34histories differ from separately closed native ordinary-input trajectories.'))
print('FROZEN_COORDINATE_SCREEN',gate,counts,'recovered',recovered,flush=True)
