"""Independent matrix algebra and fold membership audit; no refitting."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=300)
import numpy as np
from scipy.special import expit
def read(name):return json.loads((OUT/name).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
p=read('protocol_v2.json');features=read('features.json')['records'];pred=read('predictions.json');summary=read('summary.json')
assert len(features)==26 and len(pred['records'])==26
identity=lambda r:(r['video'],r['frame'],r['old_id'],r['raw_id'])
assert [identity(r) for r in features]==[identity(r) for r in pred['records']]
groups=np.array([r['video'] for r in features]);labels=np.array([r['label']=='own' for r in features],int);G=np.array([r['geometry'] for r in features])
fits={(f['video'],f['model']):f for f in pred['folds']};maxgrad=maxscore=maxloss=0.;audited=0
for video in sorted(set(groups)):
    train=np.flatnonzero(groups!=video);test=np.flatnonzero(groups==video);geo=fits[video,'G7']['normalization']
    mean=G[train].mean(0);sd=G[train].std(0);sd[np.ptp(G[train],axis=0)==0]=1
    np.testing.assert_array_equal(mean,geo['mean']);np.testing.assert_array_equal(sd,geo['scale']);X=(G-mean)/sd
    for name in p['models']:
        fit=fits[video,name];assert video not in fit.get('training_videos',[])
        valid=np.ones(len(features),bool) if name=='G7' else np.array([r['variants'][name]['valid'] for r in features])
        tr=train[valid[train]];te=test[valid[test]]
        if 'fallback' in fit:
            assert not len(tr) or set(labels[tr])!={0,1}
            assert all(pred['records'][i]['scores'][name]==pred['records'][i]['scores']['G7'] for i in test);continue
        assert fit['training_videos']==sorted(set(groups[tr])) and set(labels[tr])=={0,1}
        xt=X[tr];xe=X[te]
        if name!='G7':
            xt=np.c_[xt,np.array([features[i]['variants'][name]['channels'] for i in tr])]
            xe=np.c_[xe,np.array([features[i]['variants'][name]['channels'] for i in te])] if len(te) else np.empty((0,xt.shape[1]))
        theta=np.array(fit['fit']['theta']);weights,bias=theta[:-1],theta[-1];z=xt@weights+bias
        sw=len(tr)/(2*np.bincount(labels[tr],minlength=2)[labels[tr]])
        residual=sw*(expit(z)-labels[tr]);gradient=np.r_[xt.T@residual/len(tr)+weights,residual.mean()]
        loss=float(np.mean(sw*(np.logaddexp(0,z)-labels[tr]*z))+.5*(weights@weights))
        maxgrad=max(maxgrad,float(abs(gradient).max()));maxloss=max(maxloss,abs(loss-fit['fit']['loss']))
        expected=np.array([pred['records'][i]['scores'][name] for i in te]);observed=xe@weights+bias
        if len(te):maxscore=max(maxscore,float(abs(expected-observed).max()))
        for i in test[~valid[test]]:assert pred['records'][i]['scores'][name]==pred['records'][i]['scores']['G7']
        audited+=1
assert maxgrad<1e-7 and maxscore<1e-12 and maxloss<1e-10
counts={}
for name in p['models']:
    accepted=np.array([r['scores'][name]>0 for r in pred['records']]);own=labels==1
    row=dict(own_accept=int((own&accepted).sum()),own_reject=int((own&~accepted).sum()),foreign_accept=int((~own&accepted).sum()),foreign_reject=int((~own&~accepted).sum()))
    row['balanced_accuracy']=.5*(row['own_accept']/20+row['foreign_reject']/6)
    assert row==summary['metrics'][name];counts[name]=row
gate=(counts['C34']['own_accept']>=counts['G7']['own_accept'] and counts['C34']['foreign_accept']<=counts['G7']['foreign_accept'] and counts['C34']['balanced_accuracy']>counts['G7']['balanced_accuracy'])
assert gate==summary['gate']
result=dict(passed=True,audited_fits=audited,all20heldoutvideofolds_checked=True,independent_gradient_max=maxgrad,
    independent_loss_max_error=maxloss,prediction_max_error=maxscore,metrics=counts,gate=gate,
    sources={name:sha(OUT/name) for name in ['protocol_v2.json','execution_protocol.json','features.json','predictions.json','summary.json']},
    source_sha256=sha(Path(__file__)),zero_refits=True,zero_core_or_tracker_calls=True,goal_achieved=False)
with (OUT/'audit.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('INDEPENDENT_DOMAIN_AUDIT',gate,'fits',audited,flush=True)
