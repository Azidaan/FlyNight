import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
remaining=(datetime.fromisoformat('2026-11-12T07:40:00+03:00')-datetime.now(ZoneInfo('Europe/Moscow'))).total_seconds();assert remaining>0
observation=guard_entrypoint(timeout=remaining)
import cv2,numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
p=verify();cv2.setNumThreads(2);cv2.setRNGSeed(0);cv2.ocl.setUseOpenCL(False)
save('execution_start.json',dict(created=now(),supervisor=observation.read(),cv2_version=cv2.__version__,cpu_only=True))
geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720])
manifest=read(ROOT/'research_failure_localization_20261112/frame_manifest.json')['records'];rows=[];fields={}
def sample(field,points):
    x,y=np.array(points,dtype=float).T;ix=np.floor(x).astype(int);iy=np.floor(y).astype(int)
    assert (ix>=0).all() and (iy>=0).all() and (ix+1<field.shape[1]).all() and (iy+1<field.shape[0]).all()
    ax=(x-ix)[:,None];ay=(y-iy)[:,None]
    return field[iy,ix]*(1-ax)*(1-ay)+field[iy,ix+1]*ax*(1-ay)+field[iy+1,ix]*(1-ax)*ay+field[iy+1,ix+1]*ax*ay
for ci,case in enumerate(p['cases']):
    r=case['case'];old_id=r['old_id'];(OUT/'status.json').write_text(json.dumps(dict(stage='image_motion',old_id=old_id,updated=now())),encoding='utf8')
    ends=sorted([z for z in manifest if z['old_id']==old_id],key=lambda z:z['frame']);assert len(ends)==2
    imgs=[cv2.imread(str(ROOT/'research_failure_localization_20261112'/z['frame_png']),cv2.IMREAD_GRAYSCALE) for z in ends]
    assert all(x.shape==(720,1280) for x in imgs)
    plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets']);cells=plan['cells']
    rc=np.array(r['reference_crop']);cc=np.array(r['current_crop']);source=np.array(reader.points)[cells]*(rc[2:]-rc[:2])/reader.eye_size+rc[:2]
    originals={arm:plan['query'][:,r['old_variants'][arm]['winner']]*(cc[2:]-cc[:2])/reader.eye_size+cc[:2] for arm in ARMS}
    lk_args=dict(winSize=(11,11),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
    lk,s1,e1=cv2.calcOpticalFlowPyrLK(imgs[0],imgs[1],source.astype(np.float32)[:,None],None,**lk_args);back,s2,e2=cv2.calcOpticalFlowPyrLK(imgs[1],imgs[0],lk,None,**lk_args)
    lk=lk[:,0].astype(float);back=back[:,0].astype(float);assert np.isfinite(lk).all() and np.isfinite(back).all()
    flows=[]
    for a,b in [(0,1),(1,0)]:
        dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);flow=dis.calc(imgs[a],imgs[b],None);assert np.isfinite(flow).all();flows.append(flow);fields[f'{ci}_{a}{b}']=flow
    dp=source+sample(flows[0],source)
    assert ((dp>=[0,0])&(dp<[1278,718])).all()
    db=dp+sample(flows[1],dp);opposite=2*source-lk
    lkfb=np.linalg.norm(back-source,axis=1);disfb=np.linalg.norm(db-source,axis=1);agreement=np.linalg.norm(lk-dp,axis=1)
    cb=np.array(r['current_box']);inside=lambda pts:((pts>=cb[:2])&(pts<=cb[2:])).all(1)
    qc={mode:(pts-cc[:2])*reader.eye_size/(cc[2:]-cc[:2]) for mode,pts in [('lk',lk),('dis',dp),('opposite_lk',opposite)]}
    mask=(s1[:,0]>0)&(s2[:,0]>0)&(lkfb<=1)&(disfb<=1)&(agreement<=1)&inside(lk)&inside(dp)
    hull=np.ones(len(cells),dtype=bool)
    for query in qc.values():hull&=reader.mesh.find_simplex(query)>=0
    mask&=hull;rank=int(np.linalg.matrix_rank(source[mask]-source[mask].mean(0))) if mask.sum()>0 else 0;ready=bool(mask.sum()>=10 and rank==2)
    rows.append(dict(case=ci,old_id=old_id,cells=cells,source=source,originals=originals,lk=lk,lk_back=back,lk_status=s1[:,0],lk_back_status=s2[:,0],lk_error=e1[:,0],dis=dp,dis_back=db,opposite_lk=opposite,lk_fb=lkfb,dis_fb=disfb,method_disagreement=agreement,mask=mask,rank=rank,ready=ready,retained=int(mask.sum()),total=len(mask),median_C34_displacement_discrepancy=float(np.median(np.linalg.norm(lk[mask]-originals['C34'][mask],axis=1))) if mask.any() else None,scope='Algorithmically consistent correspondences; not independent physical landmarks.'))
    print('FLOW',old_id,'retained',int(mask.sum()),'of',len(mask),'ready',ready,flush=True)
verify()
with (OUT/'flows.npz').open('xb') as f:np.savez_compressed(f,**fields)
save('correspondence.json',dict(passed=True,created=now(),records=rows,flows_sha256=sha(OUT/'flows.npz'),new_neural_maps=0,core_calls=0,tracker_calls=0))
(OUT/'status.json').write_text(json.dumps(dict(stage='motion_complete',updated=now())),encoding='utf8')
