from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
p=read(OUT/'protocol.json');q=read(OUT/'geometry_intervention_protocol.json');m=np.load(OUT/'models.npz',allow_pickle=False);x=np.load(OUT/'features.npz',allow_pickle=False);pred=np.load(OUT/'predictions.npz',allow_pickle=False);rows=[];out={}
for n,h in q['inputs'].items():assert sha(ROOT/n)==h,n
for r in [r for r in p['records'] if r['split']=='test']:
 key='all__FlyVis34_G6';raw=x[r['id']+'__FlyVis34_G6'].astype(float);original=raw.copy();z=(raw-m[key+'__mu'])/m[key+'__sd'];score=z@m[key+'__coef']+m[key+'__intercept'];baseline=score.argmax(1);assert np.max(abs(score-pred[key+'__'+r['id']]))<1e-12
 candidates={'identity':z.copy(),'mean_FlyVis':z.copy(),'mean_geometry':z.copy(),'permute_FlyVis':z.copy()};candidates['mean_FlyVis'][:,:34]=0;candidates['mean_geometry'][:,34:]=0;candidates['permute_FlyVis'][:,:34]=z[np.random.default_rng(42).permutation(721),:34]
 assert np.array_equal(candidates['mean_FlyVis'][:,34:],z[:,34:]);assert np.array_equal(candidates['mean_geometry'][:,:34],z[:,:34]);assert np.array_equal(raw,original)
 for arm,v in candidates.items():
  s=v@m[key+'__coef']+m[key+'__intercept'];lab=s.argmax(1);out[arm+'__'+r['id']]=s
  rows.append(dict(id=r['id'],arm=arm,changed_cells=int((lab!=baseline).sum()),total_cells=721,predicted_positive_cells=int(lab.sum()),equals_supplied_box_membership=bool(np.array_equal(lab,raw[:,39].astype(int)))))
with (OUT/'geometry_intervention_predictions.npz').open('xb') as h:np.savez_compressed(h,**out)
for n,h in q['inputs'].items():assert sha(ROOT/n)==h,n
save('geometry_intervention.json',dict(passed=all(r['changed_cells']==0 for r in rows if r['arm']=='identity'),records=rows,neutralize_flyvis_changed_cells=sum(r['changed_cells'] for r in rows if r['arm']=='mean_FlyVis'),neutralize_geometry_changed_cells=sum(r['changed_cells'] for r in rows if r['arm']=='mean_geometry'),permute_flyvis_changed_cells=sum(r['changed_cells'] for r in rows if r['arm']=='permute_FlyVis'),baseline_equals_box_all=all(r['equals_supplied_box_membership'] for r in rows if r['arm']=='identity'),weights_and_features_unchanged=True,source_sha256=sha(__file__),scope='Post-result causal diagnostic of fixed trained readout, not another training or validation round. Zero normalized channels = training mean, not zero neural recurrent state.'))
print('GEOMETRY_INTERVENTION_DONE',flush=True)
