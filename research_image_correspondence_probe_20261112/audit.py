from common import *
import math,numpy as np
from scipy.interpolate import LinearNDInterpolator,RegularGridInterpolator
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear
from tracking.channel_agreement_correspondence import agreement_channels
from tracking.z13_readout import HexGeometry
p=verify();flow=read(OUT/'correspondence.json');assert flow['passed'] and sha(OUT/'flows.npz')==flow['flows_sha256']
pred=read(ROOT/'research_agreement_readout_20261112/predictions.json');geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720]);nb=HexGeometry(np.array(geom['points']),geom['eye_size']).neighbors
rows=[];checks=[];errors=[];maxinterp=maxfeature=maxscore=maxflow=0.
for phase in ['baseline','intervention']:
    with np.load(OUT/'flows.npz',allow_pickle=False) as fields:
        for ci,case in enumerate(p['cases']):
            r=case['case'];f=flow['records'][ci];assert r['old_id']==f['old_id'];source=np.array(f['source']);lk=np.array(f['lk']);dp=np.array(f['dis']);mask=np.array(f['mask']);cells=np.array(f['cells']);cc=np.array(r['current_crop']);rc=np.array(r['reference_crop'])
            plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets']);assert np.array_equal(cells,plan['cells'])
            source2=reader.points[cells]*(rc[2:]-rc[:2])/reader.eye_size+rc[:2];assert np.array_equal(source,source2)
            def sample(field,pts):return RegularGridInterpolator((np.arange(720),np.arange(1280)),field)(pts[:,::-1])
            dp2=source+sample(fields[f'{ci}_01'],source);db2=dp2+sample(fields[f'{ci}_10'],dp2)
            maxflow=max(maxflow,float(abs(dp2-dp).max()),float(abs(db2-np.array(f['dis_back'])).max()));assert maxflow<1e-9
            cb=np.array(r['current_box']);inside=lambda pts:((pts>=cb[:2])&(pts<=cb[2:])).all(1)
            q={m:(np.array(f[m])-cc[:2])*reader.eye_size/(cc[2:]-cc[:2]) for m in ['lk','dis','opposite_lk']}
            rebuilt=(np.array(f['lk_status'])>0)&(np.array(f['lk_back_status'])>0)&(np.linalg.norm(np.array(f['lk_back'])-source,axis=1)<=1)&(np.linalg.norm(db2-source,axis=1)<=1)&(np.linalg.norm(lk-dp2,axis=1)<=1)&inside(lk)&inside(dp2)
            for query in q.values():rebuilt&=reader.mesh.find_simplex(query)>=0
            assert np.array_equal(rebuilt,mask);rank=int(np.linalg.matrix_rank(source[mask]-source[mask].mean(0))) if mask.any() else 0
            assert f['ready']==bool(mask.sum()>=10 and rank==2)
            if phase=='baseline':checks.append(dict(old_id=r['old_id'],ready=f['ready'],retained=int(mask.sum()),total=len(mask),C34_mapping_discrepancy_median=f['median_C34_displacement_discrepancy']))
            for arm in ARMS:
                maps=[]
                for ep in case['endpoints']:
                    if arm=='C34':m=np.load(ROOT/ep['paths']['C34'],allow_pickle=False).astype(float)
                    else:
                        rr=np.load(ROOT/ep['paths']['R1'],allow_pickle=False).astype(float);m=rr if arm=='R134_temporal' else np.c_[rr[:,:17],rr[nb[:,1:],:17].mean(1)]
                    maps.append(m)
                fit=next(x for x in pred['fits'] if x['video']=='FINAL' and x['model']==arm+'__agreement');theta=np.array(fit['fit']['theta']);g=(np.array(r['geometry'])-fit['normalization']['mean'])/fit['normalization']['scale']
                for mode in (['original_full'] if phase=='baseline' else p['modes'][1:]):
                    if mode!='original_full' and not f['ready']:continue
                    take=np.ones(len(cells),dtype=bool) if mode=='original_full' else mask
                    query=plan['query'][:,r['old_variants'][arm]['winner']] if mode.startswith('original') or mode=='identity_subset' else q[mode]
                    aa=maps[0][cells[take]];bb=sample_linear(reader.mesh,maps[1].copy() if mode=='identity_subset' else maps[1],query[take]);value=agreement_channels(aa,bb,mean_penalty=True);assert value is not None
                    alternate=LinearNDInterpolator(reader.mesh,maps[1])(query[take]);maxinterp=max(maxinterp,float(abs(alternate-bb).max()));assert maxinterp<1e-9
                    va=np.var(aa,axis=0);vb=np.var(bb,axis=0);active=np.ptp(aa,axis=0)!=0;va[~active]=0;vb[np.ptp(bb,axis=0)==0]=0;den=va+vb+(aa.mean(0)-bb.mean(0))**2;ind=np.divide(den-np.mean((aa-bb)**2,axis=0),den,out=np.zeros(34),where=den>0);ind[~active]=0;ind*=34/active.sum();maxfeature=max(maxfeature,float(abs(ind-value).max()));assert maxfeature<1e-9
                    x=np.r_[g,(value-fit['signal_mean'])/fit['signal_scale']];score=float(x@theta[:-1]+theta[-1]);xx=np.r_[g,(ind-fit['signal_mean'])/fit['signal_scale']];other=math.fsum(xx*theta[:-1])+theta[-1];maxscore=max(maxscore,abs(score-other));assert maxscore<1e-9 and (score>0)==(other>0)
                    if mode=='original_full':
                        old=next(z for z in pred['screen'] if z['old_id']==r['old_id'])['scores'][arm+'__agreement'];errors.append(abs(old-score));assert abs(old-score)<1e-9
                    if mode=='identity_subset':assert score==next(z['score'] for z in rows if z['old_id']==r['old_id'] and z['representation']==arm and z['mode']=='original_subset')
                    rows.append(dict(old_id=r['old_id'],label=r['label'],representation=arm,mode=mode,cells=int(take.sum()),score=score,accepted=bool(score>0),channels=value))
            print('SCORED',r['old_id'],f['ready'],flush=True)
    if phase=='baseline':save('baseline_gate.json',dict(passed=True,records=rows.copy(),all15_original_scores_before_changed_comparisons=True))
def get(i,mode):return next((x for x in rows if x['old_id']==i and x['mode']==mode and x['representation']=='C34'),None)
primary=all(get(i,'lk') is not None for i in [232,70,1102])
if primary:primary=(not get(232,'original_subset')['accepted'] and get(232,'lk')['accepted'] and get(232,'dis')['accepted'] and not get(232,'opposite_lk')['accepted'] and all(get(i,m)['accepted'] for i in [70,1102] for m in ['lk','dis']))
secondary=bool(primary and all(c['ready'] for c in checks) and all(get(832,m)['accepted'] and not get(574,m)['accepted'] for m in ['lk','dis']))
verify();save('scores.json',dict(passed=True,records=rows));save('audit.json',dict(passed=True,checks=checks,records=len(rows),max_original_error=max(errors),max_flow_interpolation_error=maxflow,max_map_interpolation_error=maxinterp,max_feature_error=maxfeature,max_score_error=maxscore,primary_met=bool(primary),secondary_met=secondary,independence='Raw flow generation and image decode shared. Independent RegularGridInterpolator for fields, common-mask reconstruction, SciPy map interpolation, MSE concordance and scalar score sum. No independent physical landmarks.'))
for r in rows:
    if r['representation']=='C34':print(r['old_id'],r['mode'],round(r['score'],6))
