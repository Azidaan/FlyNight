"""Independent audit: direct Census bits, entire argmin, SciPy, scalar heads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,hashlib,math
from pathlib import Path
import numpy as np,cv2
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
p=read(OUT/'proposal_protocol.json');res=read(OUT/'result.json');stage=read(ROOT/'research_stage_localization_20261112/protocol.json');heads=read(ROOT/'research_candidate_support_probe_20261112/scores.json')['heads'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');mesh=Delaunay(np.array(g['points']));geometry={x['old_id']:x['geometry'] for x in res['geometry']};errors=dict(census_cost=0.,coordinate=0.,source=0.,interpolation=0.,channels=0.,score=0.);registration_checks=[]
for n,h in p['inputs'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
assert hashlib.sha256((OUT/'samples.npz').read_bytes()).hexdigest()==res['samples_sha256']
offsets=np.array([(dx,dy) for dy in range(-2,3) for dx in range(-2,3) if dx or dy])
with np.load(OUT/'samples.npz') as data:
    for rr in res['registrations']:
        ci=rr['case_index'];r=stage['cases'][ci]['case'];source=data[f'{ci}_source_world'];rc=r['reference_crop'];cc=r['current_crop'];rb=r['reference_box'];cb=r['current_box'];a=data[f'{ci}_gray0'];b=data[f'{ci}_gray1'];xmin=max(math.ceil(rb[0]),rc[0]+2,2);xmax=min(math.floor(rb[2]),rc[2]-3,a.shape[1]-3);ymin=max(math.ceil(rb[1]),rc[1]+2,2);ymax=min(math.floor(rb[3]),rc[3]-3,a.shape[0]-3);pixels=np.array([(x,y) for y in range(ymin,ymax+1) for x in range(xmin,xmax+1)],int)
        assert rr['ready'],'Missing registration is an explicit failure, not acceptance'
        assert np.array_equal(pixels,rr['source_pixels'])
        dxlo=max(cc[0]+2,2)-xmin;dxhi=min(cc[2]-3,b.shape[1]-3)-xmax;dylo=max(cc[1]+2,2)-ymin;dyhi=min(cc[3]-3,b.shape[0]-3)-ymax
        assert [dxlo,dylo,dxhi,dyhi]==rr['integer_rectangle']
        center0=a[pixels[:,1],pixels[:,0]];bits0=a[pixels[:,1,None]+offsets[None,:,1],pixels[:,0,None]+offsets[None,:,0]]<center0[:,None];expected=[];cost=[];prior=[(cb[j]+cb[j+2]-rb[j]-rb[j+2])/2 for j in [0,1]]
        for dx in range(dxlo,dxhi+1):
            for dy in range(dylo,dyhi+1):
                world=source+[dx,dy];q=(world-np.array(cc[:2]))*g['eye_size']/(np.array(cc[2:])-cc[:2])
                if (mesh.find_simplex(q)<0).any():continue
                xy=pixels+[dx,dy];center1=b[xy[:,1],xy[:,0]];bits1=b[xy[:,1,None]+offsets[None,:,1],xy[:,0,None]+offsets[None,:,0]]<center1[:,None];hamming=np.count_nonzero(bits0!=bits1,axis=1);ordered=sorted(int(x) for x in hamming);N=len(ordered);median=ordered[N//2] if N%2 else (ordered[N//2-1]+ordered[N//2])/2;value=(float(median),sum(ordered),sum((z-w)**2 for z,w in zip([dx,dy],prior)),dx,dy);expected.append((dx,dy));cost.append(value)
        assert expected==[tuple(x) for x in rr['candidate_translations']]
        for x,y in zip(cost,rr['cost_tuples']):errors['census_cost']=max(errors['census_cost'],max(abs(v-w) for v,w in zip(x,y)))
        winner=min(range(len(cost)),key=lambda i:cost[i]);assert winner==rr['winner'] and list(expected[winner])==rr['translation'];registration_checks.append(dict(case_index=ci,old_id=r['old_id'],translations=len(expected),pixels=len(pixels),translation=rr['translation'],global_argmin_verified=True))
    for row in res['records']:
        ci=row['case_index'];k=row['patch_key'];A=data[k+'_a'];B=data[k+'_b'];q=data[k+'_q'];r=stage['cases'][ci]['case'];cells=data[f'{ci}_cells'];m0=data[row['map0_key']];m1=data[row['map1_key']];errors['source']=max(errors['source'],float(abs(A-m0[cells]).max()));other=LinearNDInterpolator(np.array(g['points']),m1)(q);errors['interpolation']=max(errors['interpolation'],float(abs(B-other).max()))
        if row['coordinate']=='census':
            shift=res['registrations'][ci]['translation'];cc=np.array(r['current_crop']);expectedq=(data[f'{ci}_source_world']+shift-cc[:2])*g['eye_size']/(cc[2:]-cc[:2]);errors['coordinate']=max(errors['coordinate'],float(abs(q-expectedq).max()))
        active=sum(max(A[:,j])!=min(A[:,j]) for j in range(A.shape[1]));Q=[]
        for j in range(A.shape[1]):
            aa=[float(x) for x in A[:,j]];bb=[float(x) for x in B[:,j]];ma=math.fsum(aa)/len(aa);mb=math.fsum(bb)/len(bb);va=math.fsum((x-ma)**2 for x in aa)/len(aa);vb=math.fsum((x-mb)**2 for x in bb)/len(bb);cov=math.fsum((x-ma)*(y-mb) for x,y in zip(aa,bb))/len(aa);v=0. if max(aa)==min(aa) or max(bb)==min(bb) else cov/math.sqrt(va*vb) if row['head']=='native' else 2*cov/(va+vb+(ma-mb)**2);Q.append(v*A.shape[1]/active)
        errors['channels']=max(errors['channels'],max(abs(x-y) for x,y in zip(Q,row['channels'])))
        if row['score'] is not None:
            h=heads[row['representation']][row['head']];feat=[(x-m)/s for x,m,s in zip(geometry[row['old_id']],h['geometry']['mean'],h['geometry']['scale'])]+[(x-m)/s for x,m,s in zip(Q,h['signal_mean'],h['signal_scale'])];score=math.fsum(x*y for x,y in zip(feat,h['theta'][:-1]))+h['theta'][-1];errors['score']=max(errors['score'],abs(score-row['score']));assert (score>0)==row['accepted']
assert max(errors.values())<1e-9,errors
summary={}
for coordinate in ['original','census']:
    for mode in ['index','terminal_hold']:
        rows=[r for r in res['records'] if r['coordinate']==coordinate and r['reference_mode']==mode and r['representation']=='C34' and r['head']=='native'];summary[coordinate+'_'+mode]=dict(count=len(rows),correct=sum(r['accepted']==(r['label']=='own') for r in rows),decisions={str(r['old_id']):r['accepted'] for r in rows})
out=dict(passed=True,records=len(res['records']),registration_checks=registration_checks,max_errors=errors,primary_decisions=summary,independence='No production registration/scoring helpers imported; direct Boolean Census comparisons and independently enumerated entire search domain; SciPy interpolation plus scalar moments/head scoring. Same saved gray inputs and cached map generator remain shared.',labels_used_only_for_evaluation=True)
with (OUT/'audit.json').open('xb') as f:f.write((json.dumps(out,ensure_ascii=False,indent=2).rstrip()+'\n').encode('utf8'))
print('REPAIR AUDIT',len(res['records']),errors,flush=True)
