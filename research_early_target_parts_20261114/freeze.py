from common import *
import cv2,numpy as np
front=[[(343,286),(448,272),(519,257),(540,249),(542,269),(527,300),(517,310),(418,310),(383,300)],[(449,251),(486,225),(530,207),(567,198),(542,218),(511,247),(466,256)],[(546,246),(568,215),(589,204),(611,200),(592,248),(556,257)]]
rear=[[(697,179),(857,177),(902,183),(939,211),(921,219),(739,221),(706,208)],[(746,231),(927,223),(952,219),(980,231),(995,253),(976,265),(936,251),(811,258),(764,271)],[(806,292),(840,279),(921,276),(962,292),(984,311),(961,316),(937,299),(876,294),(828,310)],[(859,347),(894,344),(910,363),(910,388),(896,403),(865,412),(847,397),(844,370)]]
def draw(polys):
 m=np.zeros((720,1280),'uint8')
 for xyz in polys:cv2.fillPoly(m,[np.rint(np.array(xyz)/3+[900,200]).astype('int32')],1)
 return m.astype(bool)
f,r=draw(front),draw(rear);assert not (f&r).any()
def core(m,n):
 yy,xx=np.where(m);dist=(yy-yy.mean())**2+(xx-xx.mean())**2;i=np.lexsort((xx,yy,dist))[:n];a=np.zeros_like(m);a[yy[i],xx[i]]=1;return a
n=min(f.sum(),r.sum());a,b=core(f,n),core(r,n)
with (OUT/'prior_parts.npz').open('xb') as file:np.savez_compressed(file,visible=f|r,front_equal=a,rear_equal=b)
im=cv2.imread(str(OUT/'reference_1141.png'));im[f]=(im[f]*.45+np.array([60,215,255])*.55).astype('uint8');im[r]=(im[r]*.45+np.array([50,225,65])*.55).astype('uint8');assert cv2.imwrite(str(OUT/'past_reference_review.png'),cv2.resize(im[200:450,900:1280],(1140,750)))
save('parts.json',dict(frame=1141,annotator='single visual reviewer before SAM predictions, conservative visible cores only; not an independent instance segmentation',polygons_on_3x_image=dict(front=front,rear=rear),pixels=dict(front=int(f.sum()),rear=int(r.sum()),each_matched=int(n),union=int((f|r).sum())),coords='full=(900+x/3,200+y/3) rounded to nearest integer',review=sha(OUT/'past_reference_review.png')))
old=ROOT/'research_mask_propagation_20261114';prior=read(old/'protocol.json');design=read(OUT/'design.json');inputs={k:h for k,h in prior['inputs'].items() if k.startswith(('weights/','tools/'))}
paths=[OUT/x for x in ['common.py','freeze.py','worker.py','evaluate.py','prior_parts.npz','parts.json','design.json','frames.json']]+[old/x for x in ['protocol.json','predictions.npz','memory_predictions.npz','evaluation_truth.npz']]+[ROOT/prior['initial_mask']]
inputs.update({rel(x):sha(x) for x in paths})
for folder in [OUT,old]:
 for v in read(folder/'frames.json')['records']:
  name=v.get('path') or rel(folder/'frames'/('%05d.jpg'%(v['frame']-(1141 if folder==OUT else 1159))));inputs[name]=v['sha256']
for name,h in inputs.items():assert sha(ROOT/name)==h,name
save('protocol.json',dict(registered=now(),queries=design['queries'],inputs=inputs,weights=prior['weights'],rules=design['acceptance'],note='Primary fixed at registration. Earlier vs late manipulates appearance, occlusion and temporal lag together. Area matched front/rear holds time and size fixed.'))
save('queue_run.json',dict(runner_sha256=sha(OUT/'run_queue.py'),deadline=design['deadline'],inputs={rel(OUT/n):sha(OUT/n) for n in ['worker.py','protocol.json','evaluate.py']},jobs=[dict(entry=rel(OUT/'worker.py'),timeout=900),dict(entry=rel(OUT/'evaluate.py'),timeout=300)]))
