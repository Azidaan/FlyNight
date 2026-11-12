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
rows=[];maxfeature=maxinterp=maxscore=0.;parity=[];map_changes=[]
with np.load(OUT/'maps.npz',allow_pickle=False) as maps,np.load(OUT/'expanded_inputs.npz',allow_pickle=False) as expanded:
    for ci,case in enumerate(p['cases']):
        original=np.stack([np.load(ROOT/n,allow_pickle=False) for n in case['views']]).astype(np.float32)
        for mode in p['modes']:
            for ei,ep in enumerate(case['endpoints']):
                length=ep['length'];hx=expanded[f'{mode}_{ci}_{ei}'];prefix=original[:length]
                if mode=='terminal_hold':prefix=np.tile(prefix[-1:],(length,1))
                target=np.vstack([prefix,np.tile(prefix[-1:],(30,1))]) if mode.endswith('_hold') else prefix
                assert np.array_equal(hx,target) and np.array_equal(hx[-1],original[length-1])
                rec=next(r for r in rep['records'] if r['case']==ci and r['endpoint']==ei and r['mode']==mode)
                assert rec['update_calls']==len(target)-1 and rec['terminal_input_unchanged']
                for arm in ['C34','R1']:
                    diff=maps[f'{mode}_{ci}_{ei}_{arm}']-maps[f'index_{ci}_{ei}_{arm}'];map_changes.append(dict(old_id=case['case']['old_id'],mode=mode,endpoint=ei,representation=arm,max_abs=float(abs(diff).max()),rms=float(np.sqrt(np.mean(diff.astype(float)**2)))))
                    if mode=='identity':assert np.count_nonzero(diff)==0
        r=case['case'];plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets']);cells=plan['cells']
        for pair in p['pairs']:
            for arm in ARMS:
                arrays=[]
                for ei,mode in enumerate(pair):
                    if arm=='C34':m=maps[f'{mode}_{ci}_{ei}_C34'].astype(float)
                    else:
                        rr=maps[f'{mode}_{ci}_{ei}_R1'].astype(float);m=rr if arm=='R134_temporal' else np.c_[rr[:,:17],rr[nb[:,1:],:17].mean(1)]
                    arrays.append(m)
                query=plan['query'][:,r['old_variants'][arm]['winner']];a=arrays[0][cells];b=sample_linear(reader.mesh,arrays[1],query);q=agreement_channels(a,b,mean_penalty=True);assert q is not None
                alternate=LinearNDInterpolator(reader.mesh,arrays[1])(query);maxinterp=max(maxinterp,float(abs(alternate-b).max()));assert maxinterp<1e-9
                va=np.var(a,axis=0);vb=np.var(b,axis=0);active=np.ptp(a,axis=0)!=0;va[~active]=0;vb[np.ptp(b,axis=0)==0]=0;den=va+vb+(a.mean(0)-b.mean(0))**2;ind=np.divide(den-np.mean((a-b)**2,axis=0),den,out=np.zeros(34),where=den>0);ind[~active]=0;ind*=34/active.sum();maxfeature=max(maxfeature,float(abs(ind-q).max()));assert maxfeature<1e-9
                fit=next(f for f in pred['fits'] if f['video']=='FINAL' and f['model']==arm+'__agreement');g=(np.array(r['geometry'])-fit['normalization']['mean'])/fit['normalization']['scale'];theta=np.array(fit['fit']['theta']);x=np.r_[g,(q-fit['signal_mean'])/fit['signal_scale']];score=float(x@theta[:-1]+theta[-1]);xx=np.r_[g,(ind-fit['signal_mean'])/fit['signal_scale']];indscore=math.fsum(xx*theta[:-1])+theta[-1];maxscore=max(maxscore,abs(score-indscore));assert maxscore<1e-9 and (score>0)==(indscore>0)
                if pair==['index','index']:
                    old=next(t for t in pred['screen'] if t['old_id']==r['old_id'])['scores'][arm+'__agreement'];parity.append(dict(arm=arm,old_id=r['old_id'],error=abs(score-old)));assert abs(score-old)<1e-3
                rows.append(dict(old_id=r['old_id'],label=r['label'],representation=arm,mode='|'.join(pair),score=score,accepted=bool(score>0),channels=q))
for row in rows:
    if row['mode']=='identity|identity':assert row['score']==next(r['score'] for r in rows if r['mode']=='index|index' and r['old_id']==row['old_id'] and r['representation']==row['representation'])
gates={};counts={};decision_changes=[]
for pair in p['pairs']:
    mode='|'.join(pair)
    for arm in ARMS:
        subset={r['old_id']:r for r in rows if r['mode']==mode and r['representation']==arm};gates[mode+'|'+arm]=bool(all(subset[i]['accepted'] for i in [232,832,70,1102]) and not subset[574]['accepted']);counts[mode+'|'+arm]=dict(own_accept=sum(r['accepted'] for r in subset.values() if r['label']=='own'),foreign_accept=sum(r['accepted'] for r in subset.values() if r['label']=='foreign'))
        for old_id,r in subset.items():
            before=next(x for x in rows if x['mode']=='index|index' and x['old_id']==old_id and x['representation']==arm)
            if r['accepted']!=before['accepted']:decision_changes.append(dict(old_id=old_id,representation=arm,mode=mode,before=before['score'],after=r['score'],label=r['label'],improved=r['accepted']==(r['label']=='own')))
verify();save('scores.json',dict(passed=True,records=rows,counts=counts,gates=gates,decision_changes=decision_changes))
primary={}
for mode in ['history_hold','terminal_hold']:
    subset={r['old_id']:r for r in rows if r['mode']==mode+'|'+mode and r['representation']=='C34'};primary[mode]=all(subset[i]['accepted'] for i in [232,70,1102])
save('audit.json',dict(passed=True,records=len(rows),max_interpolation_error=maxinterp,max_feature_error=maxfeature,max_score_error=maxscore,original_score_parity=parity,identity_exact=True,terminal_inputs_exact=True,inputs_and_weights_unchanged=True,map_changes=map_changes,independence='Independent constant-input schedule reconstruction; alternative SciPy interpolation, agreement via MSE and scalar sum. Shared rendered views, maps, labels, geometry and fitted heads. Core dynamics not independently reimplemented.',primary_232_gates=primary,primary_any_both_gate=any(primary.values()),secondary_joint_gates={m:gates[m+'|'+m+'|C34'] for m in ['history_hold','terminal_hold']}))
for row in rows:
    if row['representation']=='C34':print(row['old_id'],row['mode'],round(row['score'],6))
print('DECISION_CHANGES',json.dumps(decision_changes))
