import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
import math
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear
from tracking.channel_agreement_correspondence import agreement_channels
from tracking.z13_readout import HexGeometry
p=verify();rep=read(OUT/'replay.json');assert rep['passed'] and sha(OUT/'maps.npz')==rep['maps_sha256'] and sha(OUT/'expanded_inputs.npz')==rep['expanded_inputs_sha256']
pred=read(ROOT/'research_agreement_readout_20261112/predictions.json');geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720]);nb=HexGeometry(np.array(geom['points']),geom['eye_size']).neighbors
rows=[];maxfeature=maxinterp=maxscore=0.;parity=[]
with np.load(OUT/'maps.npz',allow_pickle=False) as maps,np.load(OUT/'expanded_inputs.npz',allow_pickle=False) as expanded:
    for ci,case in enumerate(p['cases']):
        r=case['case'];plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets']);cells=plan['cells']
        for mode in p['modes']:
            sch=next(s for s in rep['schedules'] if s['case']==ci and s['mode']==mode);hx=expanded[f'{mode}_{ci}'];original=np.stack([np.load(ROOT/n,allow_pickle=False) for n in case['views']]).astype(np.float32);assert np.array_equal(hx[sch['positions']],original)
            # Independently reconstruct the full intended schedule, including duplicates.
            target=[original[0]];pos=[0]
            for j in range(1,len(original)):
                delta=case['sequence'][j][0]-case['sequence'][j-1][0]
                count=delta if mode in ('elapsed_linear','elapsed_hold') else max(1,delta) if mode in ('gaps_linear','gaps_hold') else int(delta!=0) if mode=='dedup_only' else 1
                for k in range(1,count+1):target.append(original[j] if k==count else original[j-1] if mode.endswith('hold') else original[j-1]+np.float32(k/count)*(original[j]-original[j-1]))
                pos.append(len(target)-1)
            assert np.array_equal(np.stack(target),hx) and pos==sch['positions']
            for arm in ARMS:
                arrays=[]
                for ei in [0,1]:
                    if arm=='C34':m=maps[f'{mode}_{ci}_{ei}_C34'].astype(float)
                    else:
                        rr=maps[f'{mode}_{ci}_{ei}_R1'].astype(float);m=rr if arm=='R134_temporal' else np.c_[rr[:,:17],rr[nb[:,1:],:17].mean(1)]
                    arrays.append(m)
                query=plan['query'][:,r['old_variants'][arm]['winner']];a=arrays[0][cells];b=sample_linear(reader.mesh,arrays[1],query);q=agreement_channels(a,b,mean_penalty=True);assert q is not None
                alternate=LinearNDInterpolator(reader.mesh,arrays[1])(query);maxinterp=max(maxinterp,float(abs(alternate-b).max()));assert maxinterp<1e-9
                va=np.var(a,axis=0);vb=np.var(b,axis=0);active=np.ptp(a,axis=0)!=0;va[~active]=0;vb[np.ptp(b,axis=0)==0]=0;den=va+vb+(a.mean(0)-b.mean(0))**2;ind=np.divide(den-np.mean((a-b)**2,axis=0),den,out=np.zeros(34),where=den>0);ind[~active]=0;ind*=34/active.sum();maxfeature=max(maxfeature,float(abs(ind-q).max()));assert maxfeature<1e-9
                fit=next(f for f in pred['fits'] if f['video']=='FINAL' and f['model']==arm+'__agreement');g=(np.array(r['geometry'])-fit['normalization']['mean'])/fit['normalization']['scale'];theta=np.array(fit['fit']['theta']);x=np.r_[g,(q-fit['signal_mean'])/fit['signal_scale']];score=float(x@theta[:-1]+theta[-1]);xx=np.r_[g,(ind-fit['signal_mean'])/fit['signal_scale']];indscore=math.fsum(xx*theta[:-1])+theta[-1];maxscore=max(maxscore,abs(score-indscore));assert maxscore<1e-9 and (score>0)==(indscore>0)
                if mode=='index':
                    old=next(t for t in pred['screen'] if t['old_id']==r['old_id'])['scores'][arm+'__agreement'];parity.append(dict(arm=arm,old_id=r['old_id'],error=abs(score-old)));assert abs(score-old)<1e-3
                rows.append(dict(old_id=r['old_id'],label=r['label'],representation=arm,mode=mode,score=score,accepted=bool(score>0),channels=q))
for row in rows:
    if row['mode']=='identity':assert row['score']==next(r['score'] for r in rows if r['mode']=='index' and r['old_id']==row['old_id'] and r['representation']==row['representation'])
gates={};counts={}
for mode in p['modes']:
    for arm in ARMS:
        subset={r['old_id']:r for r in rows if r['mode']==mode and r['representation']==arm};gates[mode+'|'+arm]=bool(all(subset[i]['accepted'] for i in [70,1102]) and any(subset[i]['accepted'] for i in [232,832]) and not subset[574]['accepted']);counts[mode+'|'+arm]=dict(own_accept=sum(r['accepted'] for r in subset.values() if r['label']=='own'),foreign_accept=sum(r['accepted'] for r in subset.values() if r['label']=='foreign'))
verify();save('scores.json',dict(passed=True,records=rows,counts=counts,gates=gates));save('audit.json',dict(passed=True,records=len(rows),max_interpolation_error=maxinterp,max_feature_error=maxfeature,max_score_error=maxscore,original_score_parity=parity,identity_exact=True,inputs_and_weights_unchanged=True,independence='Independent schedule expansion, SciPy interpolation, agreement through MSE and scalar sum. Shared saved inputs, heads, geometry and maps disclosed. Core dynamics not independently reimplemented.',primary_any_elapsed_gate=any(gates[m+'|C34'] for m in ['elapsed_linear','elapsed_hold'])))
for r in rows:
    if r['representation']=='C34':print(r['mode'],r['old_id'],round(r['score'],6))
