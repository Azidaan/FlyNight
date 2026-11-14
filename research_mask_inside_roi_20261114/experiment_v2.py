from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,cv2,torch,time,subprocess,shutil
from scipy.spatial import cKDTree,ConvexHull
from PIL import Image,ImageDraw,ImageFont
torch.set_num_threads(2);cv2.setNumThreads(2)
MODE=sys.argv[1];points=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int);px,py=points.T
near=cKDTree(points).query(points,k=7)[1]
def sampled(im):return cv2.boxFilter(cv2.resize(im.astype(np.float32),(391,391),interpolation=cv2.INTER_AREA),-1,(13,13),normalize=True,borderType=cv2.BORDER_CONSTANT)[py,px]
def polygon(poly):return cv2.fillPoly(np.zeros((391,391),np.uint8),[np.array(poly,np.int32)],1).astype(bool)
if MODE=='prepare':
 records=[];inputs={};old=read(ROOT/'research_yolo26seg_20261114/flyvis_probe/protocol.json');case=read(ROOT/'research_yolo26seg_20261114/case2179.json')['events'][0];manualold=ROOT/'research_visible_part_2179_20261114/mask_review/masks_reviewed.npz'
 for r in old['records']:
  r=dict(r,video=old['video'],id=f"old2179_{r['frame']}",split='train' if r['frame']==1159 else 'diagnostic',image=f"research_visible_part_2179_20261114/frames/{r['frame']}.png",mask=rel(manualold),keys=[f"{r['frame']}_background",f"{r['frame']}_target",f"{r['frame']}_neighbor"],occluder=f"{r['frame']}_occluder")
  r['box']=case['roles'][0 if r['frame']==1159 else -1]['box'];records.append(r)
 for e in read(ROOT/'research_yolo26seg_20261114/holdout/manifest.json')['events']:
  if e['id']=='B_dusk_queue':continue
  for role in ['target_ref','query']:
   q=next(r for r in e['roles'] if r['role']==role);matches=[]
   for f in (ROOT/'runs/domain_matched_native_20261109/features'/e['video']).glob('*.C34.json'):
    m=read(f)
    if m['sequence'][-1][0]==q['frame']:matches.append((f,m))
   assert len(matches)==1,(e['id'],q['frame'],len(matches));path,m=matches[0];ret=[]
   for fr,crop in m['sequence']:
    key=hashlib.sha256(json.dumps([fr,crop],separators=(',',':')).encode()).hexdigest()[:32];n=next(n for n in m['retinal_sources'] if Path(n).stem==key);assert sha(ROOT/n)==m['retinal_sources'][n];ret.append(dict(path=n))
   records.append(dict(id=e['id']+'_'+role,video=e['video'],frame=q['frame'],split='train',image=q['image'],box=q['box'],crop=m['sequence'][-1][1],map=rel(path.with_suffix('.npy')),metadata=rel(path),retinal=ret,mask='research_yolo26seg_20261114/holdout/manual_masks.npz',keys=[None,q['manual_key'],q['other_manual_key']],uncertain=q['uncertain_key']))
 # Manual visible cores frozen before predictions. Unknown everywhere else;
 # explicit neighboring vehicle and background regions are separately scored.
 polys=[dict(target=[[120,150],[155,135],[218,135],[228,153],[202,173],[124,184]],neighbor=[[241,170],[324,167],[333,245],[300,294],[213,285],[207,221]],background=[[281,128],[292,129],[292,155],[282,155]]),dict(target=[[151,167],[187,156],[243,166],[264,186],[253,211],[176,207],[150,191]],neighbor=[[136,259],[300,254],[350,300],[342,380],[134,379],[115,310]],background=[[102,148],[122,148],[122,179],[102,179]]),dict(target=[[275,183],[349,183],[358,206],[343,261],[348,304],[322,314],[301,286],[298,244],[279,222]],neighbor=[[191,248],[264,248],[280,279],[278,335],[215,349],[194,317]],background=[[232,155],[247,155],[247,174],[232,174]])]
 for r,pol in zip(read(OUT/'test_selection.json')['records'],polys):
  r=dict(r,split='test',polygons=pol);records.append(r)
 for r in records:
  for n in [r['image'],r['map'],r['metadata']]+[t['path'] for t in r['retinal']]+([r['mask']] if 'mask' in r else []):inputs[n]=sha(ROOT/n)
 inputs[rel(Path(__file__))]=sha(__file__)
 criteria=['No train/testvideo overlap; known2179 remains diagnostic.','Target recall>=0.5 and false positive fraction among known neighbor/background INSIDE ROI<=0.1 on each new video.','FlyVis must improve average targetIoU over identical retinal-history reader, and win on at least2of3newvideos.','Do not select seed, epochs, thresholds or labels from test predictions; source hashes unchanged.']
 save('protocol.json',dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),records=records,inputs=inputs,acceptance=criteria,method='Frozen maps; 34self+34meanof7nearestcells+2relativeROIcoordinates; MLP70-32-3, ReLU, Adam lr0.003,200epochs,seeds42/43/44 averaged. No box-inside bit. Matched retinal-history34 and geometry-only controls. Training uses existing reviewed manual cores and original/1.6expanded boxes; class-balanced loss, background split equally inside/outsideROI when both available. Classes other,target,neighbor. Masks are visible cores not full ground truth.',stop='Missing/ambiguous map, nonfinite feature, fewer than3cells any trainingclass, or overlap of training/testvideos: stop. No core or tracker runs.',scope='Newtestvideos excluded from previous mask-reader fits and previous4videoevaluation; previously present in archived feature/teacher preparation. Agent manual test cores; difficult adjacent/occluding vehicles; no global unseen-data claim.',budget='CPU small readout<=15minutes; no downloads or core forward',literature=['https://openaccess.thecvf.com/content_ICCV_2019/html/Bhat_Learning_Discriminative_Model_Prediction_for_Tracking_ICCV_2019_paper.html','https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html']))
 (OUT/'before').mkdir()
 for n in ['project_history.json','system_problems.json','obsidian_experiment_topics.json','useful_literature.json']:shutil.copyfile(ROOT/'experiments'/n,OUT/'before'/n)
 shutil.copyfile(ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md',OUT/'before/ЕДИНЫЙ_ПЛАН.md')
 cat=read(ROOT/'experiments/project_history.json');cat['experiments'].append(dict(id='mask_inside_roi_20261114',title='Различение машины и помех внутри рамки — нелинейный считыватель',area='own_tracker',state='running',started='2026-11-14',question='Может ли считыватель отделить машину от соседа и фона внутри смешанной рамки?',hypothesis='Явная разметка помех и нелинейное локальное считывание могут устранить заполнение рамки; вклад FlyVis требует равного контроля.',expected='Маски на старом2179 и трёх новых для обучения видео.',acceptance=[dict(criterion=x,met=None) for x in criteria],evidence=[dict(kind='file',ref=rel(OUT/'protocol.json'))]));(ROOT/'experiments/project_history.json').write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 journal('Постановка mask_inside_roi_20261114: ручные метки целевой машины, соседа и фона; убран прямой признак попадания в рамку, добавлены локальное окружение и нелинейный считыватель. Исходный2179показан отдельно от новых видео. Протокол '+rel(OUT/'protocol.json')+'. Гипотеза глубины предложена автором отдельно; пока не рассчитана.')
 print('REGISTERED',len(records));raise SystemExit
p=read(OUT/'protocol_v2.json')
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
trainvid={r['video'] for r in p['records'] if r['split']=='train'};testvid={r['video'] for r in p['records'] if r['split']=='test'};assert trainvid.isdisjoint(testvid)
features={};truth={};examples=[];originals={}
for r in p['records']:
 x1,y1,x2,y2=r['crop'];patch=cv2.resize(cv2.imread(str(ROOT/r['image']))[y1:y2,x1:x2],(391,391));originals[r['id']]=patch
 if 'mask' in r:
  z=np.load(ROOT/r['mask'],allow_pickle=False);pos=z[r['keys'][1]].astype(bool);neg=z[r['keys'][2]].astype(bool)
  bg=z[r['keys'][0]].astype(bool)|z[r['occluder']].astype(bool) if r['keys'][0] else ~(pos|neg|z[r['uncertain']].astype(bool))
  raw=[a[y1:y2,x1:x2] for a in [bg,pos,neg]];raster=[cv2.resize(a.astype(np.uint8),(391,391),interpolation=cv2.INTER_NEAREST).astype(bool) for a in raw];cov=np.stack([sampled(a) for a in raw],1)
 else:
  raster=[polygon(r['polygons'][name]) for name in ['background','target','neighbor']];cov=np.stack([sampled(a) for a in raster],1)
 lab=np.full((391,391),-1,int)
 for k,a in enumerate(raster):lab[a]=k
 truth[r['id']]=lab;y=np.where(cov.max(1)>=.9,cov.argmax(1),-1)
 fly=np.load(ROOT/r['map'],allow_pickle=False).astype(float);hx=np.stack([np.load(ROOT/t['path'],allow_pickle=False) for t in r['retinal']]).astype(float);ema=np.repeat(hx[:1],17,axis=0)
 for cur in hx:ema+=(cur[None]-ema)/np.geomspace(1.5,32,17)[:,None]
 retina=np.concatenate([ema,cur[None]-ema]).T
 for scale in ([1.,1.6] if r['split']=='train' else [1.]):
  b=np.array(r['box']);center=(b[:2]+b[2:])/2;wh=(b[2:]-b[:2])*scale;xy=np.c_[x1+(px+.5)*(x2-x1)/391-.5,y1+(py+.5)*(y2-y1)/391-.5];g=(xy-center)/wh
  fs={'FlyVis':np.c_[fly,fly[near].mean(1),g],'Control':np.c_[retina,retina[near].mean(1),g],'Geometry':g}
  if scale==1:features[r['id']]=fs
  if r['split']=='train':
   valid=y>=0;counts=np.bincount(y[valid],minlength=3);assert counts.min()>=3,(r['id'],counts)
   w=np.zeros(721);inside=(abs(g)<=.5).all(1)
   for k in range(3):
    groups=[(y==k)&inside,(y==k)&~inside] if k==0 else [y==k];groups=[m for m in groups if m.any()]
    for m in groups:w[m]=1/(3*len(groups)*m.sum())
   examples.append((fs,y,w))
pred={};weights={};losses=[];start=time.perf_counter()
for arm in ['FlyVis','Control','Geometry']:
 xx=np.concatenate([e[0][arm][e[1]>=0] for e in examples]);yy=np.concatenate([e[1][e[1]>=0] for e in examples]);ww=np.concatenate([e[2][e[1]>=0]/len(examples) for e in examples]);mu=xx.mean(0);sd=xx.std(0);sd[sd<1e-8]=1;x=torch.tensor((xx-mu)/sd,dtype=torch.float32);y=torch.tensor(yy,dtype=torch.long);w=torch.tensor(ww,dtype=torch.float32);scores={r['id']:[] for r in p['records']}
 for seed in [42,43,44]:
  torch.manual_seed(seed);model=torch.nn.Sequential(torch.nn.Linear(xx.shape[1],32),torch.nn.ReLU(),torch.nn.Linear(32,3));opt=torch.optim.Adam(model.parameters(),lr=.003)
  for epoch in range(200):
   opt.zero_grad();loss=(torch.nn.functional.cross_entropy(model(x),y,reduction='none')*w).sum();loss.backward();assert torch.isfinite(loss);opt.step()
  losses.append(dict(arm=arm,seed=seed,loss=float(loss.detach())))
  for name,t in model.state_dict().items():weights[f'{arm}_{seed}_{name}']=t.detach().numpy()
  weights[arm+'_mu']=mu;weights[arm+'_sd']=sd
  with torch.no_grad():
   for r in p['records']:scores[r['id']].append(torch.softmax(model(torch.tensor((features[r['id']][arm]-mu)/sd,dtype=torch.float32)),1).numpy())
 for rid,s in scores.items():pred[arm+'__'+rid]=np.mean(s,0)
with (OUT/'models.npz').open('xb') as f:np.savez_compressed(f,**weights)
with (OUT/'predictions.npz').open('xb') as f:np.savez_compressed(f,**pred)
# Test annotations are held in preparation memory for rendering but never used
# in normalization, fitting, stopping or parameter selection. Fixed epoch count.
grid=np.stack(np.meshgrid(np.arange(391),np.arange(391)),axis=-1);nearest=cKDTree(points).query(grid.reshape(-1,2))[1].reshape(391,391);support=cv2.fillConvexPoly(np.zeros((391,391),np.uint8),points[ConvexHull(points).vertices],1).astype(bool);results=[];rasters={};font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
for r in [r for r in p['records'] if r['split']!='train']:
 truthim=truth[r['id']];known=(truthim>=0)&support;gx=(np.arange(391)+.5)*(r['crop'][2]-r['crop'][0])/391+r['crop'][0]-.5;gy=(np.arange(391)+.5)*(r['crop'][3]-r['crop'][1])/391+r['crop'][1]-.5;b=r['box'];inside=(gx[None]>=b[0])&(gx[None]<=b[2])&(gy[:,None]>=b[1])&(gy[:,None]<=b[3]);sheet=Image.new('RGB',(391*4,425),'#222222');draw=ImageDraw.Draw(sheet)
 for j,arm in enumerate(['Original','FlyVis','Control','Geometry']):
  im=originals[r['id']].copy()
  if arm!='Original':
   labels=pred[arm+'__'+r['id']].argmax(1)[nearest];target=labels==1;tp=int((target&(truthim==1)&known).sum());fp=int((target&(truthim!=1)&known).sum());fn=int((~target&(truthim==1)&known).sum());negative=known&inside&(truthim!=1);results.append(dict(id=r['id'],split=r['split'],arm=arm,tp=tp,fp=fp,fn=fn,iou=tp/max(1,tp+fp+fn),recall=tp/max(1,tp+fn),inside_negative_pixels=int(negative.sum()),inside_false_positive=int((target&negative).sum()),inside_fpr=float((target&negative).sum()/max(1,negative.sum()))));rasters[arm+'__'+r['id']]=labels
   for cls,color in [(1,[0,220,0]),(2,[30,70,240])]:m=(labels==cls)&support;im[m]=(im[m]*.45+np.array(color)*.55).astype(np.uint8)
  sheet.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),(j*391,34));draw.text((j*391+4,4),r['id']+' '+arm,font=font,fill='white')
 sheet.save(OUT/(r['id']+'_masks.png'))
with (OUT/'evaluation_arrays.npz').open('xb') as f:np.savez_compressed(f,**{**{'truth__'+k:v for k,v in truth.items()},**rasters,'support':support})
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
save('result.json',dict(records=results,losses=losses,training_frames=4,training_videos=len(trainvid),test_videos=len(testvid),seconds=time.perf_counter()-start,source_unchanged=True,labels='manual conservative cores, not complete silhouettes',test_labels_used_in_fitting=False,models_sha256=sha(OUT/'models.npz'),predictions_sha256=sha(OUT/'predictions.npz')))
print('DONE',json.dumps(results))
