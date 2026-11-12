"""Independent holdout readout audit, direct Census costs and scalar heads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,hashlib,math
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(OUT/'protocol.json');res=read(OUT/'result.json');assert res['passed'];heads=read(ROOT/'research_candidate_support_probe_20261112/scores.json')['heads']['C34'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');mesh=Delaunay(np.array(g['points']));errors=dict(census_cost=0.,coordinates=0.,source=0.,interpolation=0.,channels=0.,head=0.);checks=[]
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
assert sha(OUT/'samples.npz')==res['samples_sha256'] and sha(OUT/'core_result.json')==res['core_result_sha256']
offsets=np.array([(dx,dy) for dy in range(-2,3) for dx in range(-2,3) if dx or dy])
with np.load(OUT/'samples.npz') as data:
    for reg in res['registrations']:
        ci=reg['case_index'];c=p['cases'][ci];r=c['proposal_geometry'];rb=r['reference_box'];cb=r['current_box'];rc=r['reference_crop'];cc=r['current_crop'];source=data[f'{ci}_source_world'];a=data[f'{ci}_gray0'];b=data[f'{ci}_gray1'];xmin=max(math.ceil(rb[0]),rc[0]+2,2);xmax=min(math.floor(rb[2]),rc[2]-3,a.shape[1]-3);ymin=max(math.ceil(rb[1]),rc[1]+2,2);ymax=min(math.floor(rb[3]),rc[3]-3,a.shape[0]-3);pixels=np.array([(x,y) for y in range(ymin,ymax+1) for x in range(xmin,xmax+1)],int).reshape(-1,2)
        if not len(pixels):assert not reg['ready'] and reg['reason']=='empty_source_pixel_template';checks.append(dict(case_key=c['case_key'],ready=False,reason=reg['reason']));continue
        dxlo=max(cc[0]+2,2)-xmin;dxhi=min(cc[2]-3,b.shape[1]-3)-xmax;dylo=max(cc[1]+2,2)-ymin;dyhi=min(cc[3]-3,b.shape[0]-3)-ymax;center0=a[pixels[:,1],pixels[:,0]];bits0=a[pixels[:,1,None]+offsets[None,:,1],pixels[:,0,None]+offsets[None,:,0]]<center0[:,None];expected=[];cost=[];prior=[(cb[j]+cb[j+2]-rb[j]-rb[j+2])/2 for j in [0,1]]
        for dx in range(dxlo,dxhi+1):
            for dy in range(dylo,dyhi+1):
                q=(source+[dx,dy]-np.array(cc[:2]))*g['eye_size']/(np.array(cc[2:])-cc[:2])
                if (mesh.find_simplex(q)<0).any():continue
                xy=pixels+[dx,dy];bits1=b[xy[:,1,None]+offsets[None,:,1],xy[:,0,None]+offsets[None,:,0]]<b[xy[:,1],xy[:,0],None];hamming=np.count_nonzero(bits0!=bits1,axis=1);ordered=sorted(int(x) for x in hamming);N=len(ordered);median=float(ordered[N//2] if N%2 else (ordered[N//2-1]+ordered[N//2])/2);expected.append((dx,dy));cost.append((median,sum(ordered),sum((v-w)**2 for v,w in zip([dx,dy],prior)),dx,dy))
        if not expected:
            assert not reg['ready'] and reg['reason']=='no_full_support_translation';checks.append(dict(case_key=c['case_key'],ready=False,reason=reg['reason']));continue
        assert reg['ready'] and expected==[tuple(x) for x in reg['candidate_translations']]
        assert np.array_equal(pixels,reg['source_pixels'])
        for x,y in zip(cost,reg['cost_tuples']):errors['census_cost']=max(errors['census_cost'],max(abs(v-w) for v,w in zip(x,y)))
        winner=min(range(len(cost)),key=lambda i:cost[i]);assert winner==reg['winner'] and list(expected[winner])==reg['translation'];checks.append(dict(case_key=c['case_key'],ready=True,candidates=len(expected),pixels=len(pixels),global_argmin_verified=True))
    for row in res['records']:
        ci=row['case_index'];c=p['cases'][ci];k=row['patch_key'];A=data[k+'_a'];B=data[k+'_b'];q=data[k+'_q'];a=data[f'{ci}_reference_'+row['reference_mode']];b=data[f'{ci}_current_index'];errors['source']=max(errors['source'],float(abs(A-a[data[f'{ci}_cells']]).max()));other=LinearNDInterpolator(np.array(g['points']),b)(q);errors['interpolation']=max(errors['interpolation'],float(abs(B-other).max()))
        if row['coordinate']=='census':
            reg=next(z for z in res['registrations'] if z['case_index']==ci);cc=np.array(c['proposal_geometry']['current_crop']);expected=(data[f'{ci}_source_world']+reg['translation']-cc[:2])*g['eye_size']/(cc[2:]-cc[:2]);errors['coordinates']=max(errors['coordinates'],float(abs(q-expected).max()))
        active=sum(max(A[:,j])!=min(A[:,j]) for j in range(34));activeb=sum(max(B[:,j])!=min(B[:,j]) for j in range(34));Q=[]
        if not active or not activeb:
            assert row['score'] is None and row['channels'] is None and row['accepted'] is None and row['reason']=='constant_patch_no_visual_evidence'
            continue
        for j in range(34):
            aa=[float(x) for x in A[:,j]];bb=[float(x) for x in B[:,j]];ma=math.fsum(aa)/len(aa);mb=math.fsum(bb)/len(bb);va=math.fsum((x-ma)**2 for x in aa)/len(aa);vb=math.fsum((x-mb)**2 for x in bb)/len(bb);cov=math.fsum((x-ma)*(y-mb) for x,y in zip(aa,bb))/len(aa);v=0. if max(aa)==min(aa) or max(bb)==min(bb) else cov/math.sqrt(va*vb) if row['head']=='native' else 2*cov/(va+vb+(ma-mb)**2);Q.append(v*34/active)
        errors['channels']=max(errors['channels'],max(abs(x-y) for x,y in zip(Q,row['channels'])));h=heads[row['head']];features=[(x-m)/s for x,m,s in zip(c['proposal_geometry']['geometry'],h['geometry']['mean'],h['geometry']['scale'])]+[(x-m)/s for x,m,s in zip(Q,h['signal_mean'],h['signal_scale'])];score=math.fsum(x*y for x,y in zip(features,h['theta'][:-1]))+h['theta'][-1];errors['head']=max(errors['head'],abs(score-row['score']));assert (score>0)==row['accepted']
assert max(errors.values())<1e-9,errors
out=dict(passed=True,records=len(res['records']),registration_checks=checks,max_errors=errors,selection_sha256=p['selection_sha256'],independence='Independent domain enumeration and direct Boolean24bit comparisons; SciPy interpolation; scalar channel/head arithmetic. Shared raw gray preprocessing, maps and core solver remain explicit.',scope='No new fits or outcome-based changes. GT identity and complete assignment evaluation remain separate parent tasks; missing Census stays missing.')
with (OUT/'audit.json').open('xb') as f:f.write((json.dumps(out,ensure_ascii=False,indent=2).rstrip()+'\n').encode('utf8'))
print('HOLDOUT_CORRESPONDENCE_AUDIT',len(res['records']),errors,flush=True)
