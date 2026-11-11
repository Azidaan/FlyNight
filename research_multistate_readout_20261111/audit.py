"""Independent correlations/metrics from saved maps and frozen outputs."""
from common import ROOT,OUT,OLD,read,write,sha,key,verify,start,status
start()
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import rankdata
from scripts.z13_target_model import neighbor_aug
from tracking.z13_readout import HexGeometry
from common import arrays

verify();p=read(OUT/'predictions.json');rows=p['records'];edges=read(OLD/'protocol_v2.json')['edges']
geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');points=np.array(geom['points'],dtype=np.float32).astype(float)
nb=HexGeometry(np.array(geom['points']),geom['eye_size']).neighbors
old=read(OLD/'features.json')['records']; oldpred=read(OLD/'predictions.json')
maxerr=0.; checked=0
for row,e,saved in zip(rows,edges,old):
    mids=[];last=[]
    for seq in [e['sequence'][:4],e['sequence']]:
        k=key(e['video'],seq);end=arrays(k)
        with np.load(OUT/'maps'/(k+'.npz'),allow_pickle=False) as z: mid={n:z[n].copy().astype(float) for n in z.files}
        def shape(c,r):return dict(C34=c.astype(float),R134_temporal=r.astype(float),R134_spatial=neighbor_aug(r[:,:17].T[None],nb)[0].T.astype(float))
        mids.append(shape(mid['C1'],mid['R1_mid']));last.append(shape(end['C34'],end['R1']))
    for name,item in row['variants'].items():
        cells=np.array(row['cells']);q=np.array(row['query']);a=mids[0][name][cells]
        patches=LinearNDInterpolator(points,mids[1][name])(q)
        def independent(a,b):
            active=np.ptp(a,axis=0)!=0
            if not active.any() or not (np.ptp(b,axis=0)!=0).any():return None
            x=a-a.mean(0);y=b-b.mean(0);x[:,~active]=0;y[:,np.ptp(b,axis=0)==0]=0
            den=np.sqrt((x*x).sum(0)*(y*y).sum(0));r=np.zeros(a.shape[1]);ok=den>0;r[ok]=(x[:,ok]*y[:,ok]).sum(0)/den[ok]
            return r*len(r)/sum(active)
        for k in range(q.shape[1]):
            v=independent(a,patches[:,k]);recorded=item['first_vectors'][k]
            assert (v is None)==(recorded is None)
            if v is not None:maxerr=max(maxerr,float(np.max(abs(v-recorded))))
            checked+=1
        if item['valid']:
            k=item['winner'];b=LinearNDInterpolator(points,last[1][name])(q[:,k]);v=independent(last[0][name][cells],b)
            maxerr=max(maxerr,float(np.max(abs(v-item['last_vector_at_joint']))))
            assert k==max(range(len(item['scores'])),key=lambda j:item['scores'][j])
    assert np.array_equal(row['geometry'],saved['geometry'])
assert maxerr<1e-10,(maxerr,'independent interpolation/correlation disagreement')
labels=np.array([r['label']=='own' for r in rows]);videos=sorted(set(r['video'] for r in rows))
def metric(ix,name,which):
    y=labels[ix];z=np.array([rows[i]['variants'][name][which] for i in ix]);accepted=z>0
    own=int((accepted&y).sum());foreign=int((accepted&~y).sum())
    ba=.5*(own/y.sum()+1-foreign/(~y).sum()) if y.any() and (~y).any() else None
    ranks=rankdata(z);auc=(ranks[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()) if ba is not None else None
    return dict(own_accept=own,foreign_accept=foreign,own_reject=int(y.sum())-own,foreign_reject=int((~y).sum())-foreign,balanced_accuracy=ba,auc=auc)
ix=np.arange(len(rows));metrics={}
for n in ['C34','R134_temporal','R134_spatial']:
    metrics[n]=dict(last=metric(ix,n,'last_decision_score'),joint=metric(ix,n,'joint_decision_score'))
    assert all(metrics[n]['last'][k]==read(OLD/'summary.json')['metrics'][n][k] for k in ['own_accept','foreign_accept','balanced_accuracy'])
rng=np.random.default_rng(20261111);groups=[np.flatnonzero([r['video']==v for r in rows]) for v in videos]
contrasts={n:[] for n in ['last_C34','joint_R134_temporal','joint_R134_spatial']}
for _ in range(4000):
    ii=np.concatenate([groups[k] for k in rng.integers(len(groups),size=len(groups))])
    if not labels[ii].any() or labels[ii].all():continue
    c=metric(ii,'C34','joint_decision_score')['balanced_accuracy']
    contrasts['last_C34'].append(c-metric(ii,'C34','last_decision_score')['balanced_accuracy'])
    for n in ['R134_temporal','R134_spatial']:contrasts['joint_'+n].append(c-metric(ii,n,'joint_decision_score')['balanced_accuracy'])
intervals={k:dict(ci95=np.percentile(v,[2.5,97.5]),draws=len(v)) for k,v in contrasts.items()}
a=metrics['C34']['last'];b=metrics['C34']['joint'];gate=b['own_accept']>=a['own_accept'] and b['foreign_accept']<=a['foreign_accept'] and (b['own_accept']>a['own_accept'] or b['foreign_accept']<a['foreign_accept'])
changes=[dict(video=r['video'],frame=r['frame'],old_id=r['old_id'],raw_id=r['raw_id'],label=r['label'],signal=n,
    old=v['last_decision_score'],joint=v['joint_decision_score'],corrected=bool((v['joint_decision_score']>0)==(r['label']=='own')))
    for r in rows for n,v in r['variants'].items() if v['decision_changed']]
write('summary.json',dict(passed=True,edges=26,own=20,foreign=6,videos=len(videos),metrics=metrics,
    paired_video_intervals=intervals,preliminary_C34_gate=gate,decision_changes=changes,
    independent_correlation_max_error=maxerr,checked_first_shift_vectors=checked,
    final_state_parity=read(OUT/'maps_complete.json')['final_parity_max_error'],
    threshold_tuning=False,new_training=False,actual_IDs_evaluated=False,
    scope='Previously viewed development cohort; frozen-head equal fusion only. Not independent tracker validation or rejection of all multi-state readers.'))
write('audit.json',dict(passed=True,independent_correlation_max_error=maxerr,checked_first_shift_vectors=checked,
    prior_scores_reused=True,old_metrics_reconciled=True,geometry_unchanged=True,source_fingerprints_verified=True,
    shared_components=['archived-map loader','coordinate plan saved by comparator','neighbor_aug for ordinary spatial representation'],
    independent_components=['LinearNDInterpolator rather than project interpolator','centered correlations','acceptance counts','paired-video uncertainty']))
verify();status('completed',completed=26,total=26,gate=gate)
print('MULTISTATE_RESULT',gate,metrics)
