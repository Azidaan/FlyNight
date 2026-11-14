from official import *
from scipy.spatial import cKDTree,ConvexHull
p=read(OUT/'protocol.json');lab=zipfile.ZipFile(ROOT/p['labelzip']);imgs=zipfile.ZipFile(ROOT/p['imagezip']);pts=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int);px,py=pts.T;neighbors=cKDTree(pts).query(pts,k=7)[1][:,1:];norm=read(ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json');mu,sd=np.array(norm['mean'])[:34],np.array(norm['sd'])[:34];cache={};seen=set();requests=[];excluded=[];arr={};qa={}
def sample(a):return cv2.boxFilter(cv2.resize(a.astype(np.float32),(391,391),interpolation=cv2.INTER_AREA),-1,(13,13),normalize=True,borderType=cv2.BORDER_CONSTANT)[py,px]
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
for r in p['train_maps']+p['test_maps']:
 v=r['video'];split=r['split'];fr=r['frame'];x,y,x2,y2=r['crop'];c=np.array([(x+x2)/2,(y+y2)/2]);sz=np.array([x2-x,y2-y]);k=(v,split)
 if k not in cache:cache[k]={g['frameIndex']:g for g in json.loads(lab.read(PREFIX+'rles/'+split+'/'+v+'.json'))['frames']}
 frame=cache[k][r['index']];assert frame['name']==r['name'];name=frame['name'];im=cv2.imdecode(np.frombuffer(imgs.read('bdd100k/images/seg_track_20/'+split+'/'+v+'/'+name),'uint8'),1);assert im is not None
 pix=np.asarray(Image.open(io.BytesIO(lab.read(PREFIX+'bitmasks/'+split+'/'+v+'/'+name.replace('.jpg','.png')))));valid=(pix[:,:,1]&3)==0;vehicles=[o for o in frame['labels'] if o['category'] in ['car','truck','bus'] and not any((o['attributes'] or {}).get(k,False) for k in ['crowd','ignore'])];allvehicle=np.zeros(valid.shape,bool)
 for o in vehicles:allvehicle|=rle(o)
 if split=='train':
  flyfile=np.load(ROOT/r['fly']);retfile=np.load(ROOT/r['retina']);assert int(flyfile['uid'][r['row']])==int(r['id'])==int(retfile['uid'][r['row']]);fly=flyfile['full'][r['row'],:,:34];ret=retfile['full'][r['row'],:,:34];bs=r['boxes'];ids=r['raw_ids']
 else:
  fly=np.load(ROOT/r['fly']);hx=np.stack([np.load(ROOT/n).astype('float64') for n in r['retinal']]);ema=np.repeat(hx[:1],17,axis=0);tau=np.geomspace(1.5,32,17)
  for step in hx:ema+=(step-ema)/tau[:,None]
  ret=((np.concatenate([ema,hx[-1]-ema]).T-mu)/sd).astype('float32')
  if r['retina'] is not None:assert np.max(abs(ret-np.load(ROOT/r['retina'])))<2e-5,('retina replay',r['id'],np.max(abs(ret-np.load(ROOT/r['retina']))))
  det=np.load(ROOT/r['detections']);ii=np.flatnonzero((det['frame_idx']==fr)&(det['cls']==2)&(det['conf']>=.25));bb=det['xyxy'][ii];cc=(bb[:,:2]+bb[:,2:])/2;indices=[j for j in range(len(ii)) if x<=cc[j,0]<x2 and y<=cc[j,1]<y2];indices.sort(key=lambda j:(np.sum(((cc[j]-c)/sz)**2),ii[j]));ids=ii[indices[:2]].tolist();bs=bb[indices[:2]].tolist()
 rgb=sample(cv2.cvtColor(im[y:y2,x:x2],cv2.COLOR_BGR2RGB)/255.);assert fly.shape==ret.shape==(721,34)
 for bi,(b,raw_id) in enumerate(zip(bs,ids)):
  b=np.array(b,float);key=r['id']+'_'+str(bi);physical=(v,fr,tuple(np.round(b,4)),tuple(r['crop']));unique=physical if split=='train' else (physical,r['id']);rank=sorted([(iou(b,bbox(o)),o['id'],o) for o in vehicles],key=lambda a:(-a[0],a[1]))
  if unique in seen or not rank or rank[0][0]<.5 or (len(rank)>1 and rank[0][0]==rank[1][0]):
   excluded.append(dict(id=key,video=v,frame=fr,split=split,reason='duplicate' if unique in seen else 'no unique .5 official instance overlap',best_overlap=rank[0][0] if rank else 0));continue
  seen.add(unique);t=rle(rank[0][2]);truth=np.where(t,1,np.where(allvehicle,2,0)).astype('int8');truth[~valid]=-1;patch=truth[y:y2,x:x2];coverage=np.stack([sample(patch==cls) for cls in [0,1,2]],1);label=np.where(coverage.max(1)>=.9,coverage.argmax(1),-1).astype('int8')
  if (label==1).sum()<3 or (label==0).sum()<3:excluded.append(dict(id=key,video=v,frame=fr,split=split,reason='fewer than 3 pure target or background cells',target=int((label==1).sum())));continue
  xy=np.c_[x+(px+.5)*(x2-x)/391-.5,y+(py+.5)*(y2-y)/391-.5];geom=(xy-(b[:2]+b[2:])/2)/np.maximum(b[2:]-b[:2],1)
  sy,sx=np.indices(patch.shape);boxfill=(sx+x>=b[0])&(sx+x<=b[2])&(sy+y>=b[1])&(sy+y<=b[3]);proxy=np.where(boxfill,1,np.where(patch==2,2,0));proxy[patch<0]=-1;cover=np.stack([sample(proxy==cls) for cls in [0,1,2]],1);weak=np.where(cover.max(1)>=.9,cover.argmax(1),-1)
  # Missing labels for either supervision source get zero training weight, not synthetic background.
  if split=='train' and ((weak==1).sum()<3 or (weak==0).sum()<3):excluded.append(dict(id=key,video=v,frame=fr,split=split,reason='paired box proxy lacks three supervised cells'));continue
  for feat,values in [('Fly',fly),('Retina',ret),('RGB',rgb),('Geometry',np.zeros((721,0)))]:
   vals=np.c_[values,values[neighbors].mean(1)];padded=np.zeros((721,68),'float32');padded[:,:vals.shape[1]]=vals;arr[key+'__'+feat]=np.c_[padded,geom].astype('float32')
  arr[key+'__labels']=label;arr[key+'__proxy']=weak.astype('int8');arr[key+'__truth']=cv2.resize(patch.astype('float32'),(391,391),interpolation=cv2.INTER_NEAREST).astype('int8');arr[key+'__inside']=cv2.resize(boxfill.astype('uint8'),(391,391),interpolation=cv2.INTER_NEAREST).astype(bool)
  requests.append(dict(id=key,split=split,video=v,frame=fr,index=r['index'],name=name,raw_id=int(raw_id),physical=v+'_'+str(fr)+'_'+str(raw_id),eye=r['crop'],box=b.tolist(),official_id=rank[0][1],overlap=rank[0][0],cells={str(k):int((label==k).sum()) for k in [0,1,2]}))
with (OUT/'dataset.npz').open('xb') as file:np.savez_compressed(file,**arr)
save('dataset.json',dict(records=requests,excluded=excluded,sha256=sha(OUT/'dataset.npz'),n_train_videos=len({r['video'] for r in requests if r['split']=='train'}),n_val_videos=len({r['video'] for r in requests if r['split']=='val'})));print('SUPERVISED_DATA',[(s,len([r for r in requests if r['split']==s]),len({r['video'] for r in requests if r['split']==s})) for s in ['train','val']],flush=True)
