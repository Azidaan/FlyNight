"""Apply 80 existing held-out-video models; never retrain or restore a map."""
import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent;TRAIN=ROOT/'research_domain_matched_reader_20261109'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
stop=datetime(2026,11,9,23,54,tzinfo=timezone(timedelta(hours=3)))
guard_entrypoint(timeout=min(600,(stop-datetime.now(stop.tzinfo)).total_seconds()))
import psutil,numpy as np
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.channel_identity_head import spatial_channel_features
from tracking.z13_readout import HexGeometry
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cast(x):
    if isinstance(x,dict):return {k:cast(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [cast(v) for v in x]
    return x.tolist() if hasattr(x,'tolist') else x
def verify():
    for name,digest in protocol['sources'].items():assert sha(ROOT/name)==digest,name
protocol=read(OUT/'protocol.json');verify();assert read(OUT/'preflight.json')['passed']
inventory=read(OUT/'inventory.json');assert inventory['ready']
old=read(TRAIN/'predictions.json');rows=read(TRAIN/'features.json')['records'];deployment=read(ROOT/'runs/domain_matched_native_20261109/deployment.json')
folds={(r['video'],r['model']):r for r in old['folds']};geometry=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
reader=ObservedSpatialReader(geometry['points'],geometry['eye_size'],[1280,720]);nb=HexGeometry(np.array(geometry['points']),geometry['eye_size']).neighbors
axes=np.array([(u,v) for u in range(-2,3) for v in range(-2,3) if max(abs(u),abs(v),abs(u+v))<=2])@np.array([[13.,6.5],[0.,13.]])
arms=['G7','C34','R134_spatial','R134_temporal']
def score(model,stats,g,channels=None):
    normalized=(np.array(g)-stats['mean'])/stats['scale'];x=normalized if channels is None else np.r_[normalized,channels]
    t=np.array(model['theta']);return float(x@t[:-1]+t[-1])
# Exact saved-prediction witness verifies indexing and saved fold normalizations.
parity=0.
for row,record in zip(rows,old['records']):
    assert all(row[k]==record[k] for k in ['video','frame','old_id','raw_id'])
    for arm in arms:
        fold=folds[(row['video'],arm)];stats=folds[(row['video'],'G7')]['normalization']
        channels=None if arm=='G7' else row['variants'][arm]['channels']
        parity=max(parity,abs(score(fold['fit'],stats,row['geometry'],channels)-record['scores'][arm]))
assert parity<1e-12,parity
cases=[]
for item in inventory['records']:
    assert item['video'] not in {r['video'] for r in rows}
    rc=np.array(item['reference_crop']);ct=np.array(item['current_crop']);offsets=axes*np.maximum((rc[2:]-rc[:2])/geometry['eye_size'],(ct[2:]-ct[:2])/geometry['eye_size'])
    plan=relative_template_plan(reader,item['reference_box'],rc,item['current_box'],ct,offsets)
    values=[]
    for request in item['maps']:
        maps=[np.load(ROOT/m['path'],allow_pickle=False) for m in request['matches']]
        for value in maps[1:]:np.testing.assert_array_equal(maps[0],value)
        values.append(maps[0].astype(float))
    channels={'C34':np.array(item['channels'])};features={}
    for arm in ['R134_spatial','R134_temporal']:
        maps=[np.concatenate([v[:,:17],v[nb[:,1:],:17].mean(1)],1) if arm=='R134_spatial' else v for v in values]
        feature=spatial_channel_features(reader.mesh,*maps,plan['cells'],plan['query']);assert feature['valid']
        channels[arm]=feature['channels'];features[arm]=dict(winner=feature['winner'],scalar=feature['scalar'],channels=feature['channels'])
    baseline={arm:score(deployment['models'][arm],deployment['geometry_normalization'],item['geometry'],channels.get(arm)) for arm in arms}
    assert abs(baseline['C34']-item['logit'])<1e-12
    evaluations=[]
    for video in sorted({r['video'] for r in rows}):
        removed=[r for r in rows if r['video']==video];scores={}
        for arm in arms:
            fold=folds[(video,arm)];assert video not in fold['training_videos'] and item['video'] not in fold['training_videos']
            scores[arm]=score(fold['fit'],folds[(video,'G7')]['normalization'],item['geometry'],channels.get(arm))
        evaluations.append(dict(removed_video=video,removed_own=sum(r['label']=='own' for r in removed),removed_foreign=sum(r['label']=='foreign' for r in removed),scores=scores))
    cases.append(dict(**{k:item[k] for k in ['video','frame','old_id','raw_id','kind']},baseline=baseline,fold_predictions=evaluations,new_image_features=features,geometry=item['geometry'],ready_C34_channels=item['channels']))
summary={}
for arm in arms:
    records=[]
    for c in cases:
        label=c['kind']=='own';base=c['baseline'][arm];variants=[dict(**e,score=e['scores'][arm]) for e in c['fold_predictions']]
        changes=[e for e in variants if (e['score']>0)!=(base>0)]
        records.append(dict(video=c['video'],frame=c['frame'],old_id=c['old_id'],kind=c['kind'],baseline=base,min_score=min(e['score'] for e in variants),max_score=max(e['score'] for e in variants),
            baseline_correct=(base>0)==label,sign_changes=[dict(video=e['removed_video'],own=e['removed_own'],foreign=e['removed_foreign'],score=e['score'],correct=(e['score']>0)==label) for e in changes]))
    summary[arm]=records
verify()
with (OUT/'comparison.json').open('x',encoding='utf8') as f:json.dump(cast(dict(passed=True,cases=cases,summary=summary,old_prediction_parity_max=parity,reused_models=80,new_fits=0,no_core_or_video=True,actual_IDs_evaluated=False,goal_achieved=False)),f,ensure_ascii=False,indent=2)
print('READY_MODEL_STABILITY_DONE',json.dumps(cast(summary)),flush=True)
