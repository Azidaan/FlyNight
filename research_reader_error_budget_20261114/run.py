from official import *
from scipy.spatial import cKDTree,ConvexHull
import torch
os.environ['CUDA_VISIBLE_DEVICES']='';torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
p=read(OUT/'protocol.json');src=ROOT/'research_supervised_contour_reader_20261114';z=np.load(src/'dataset.npz');db=read(src/'dataset.json');base=np.load(src/'predictions.npz');fit=[r for r in db['records'] if r['split']=='train'];val=[r for r in db['records'] if r['split']=='val'];videos={r['video'] for r in fit};pts=np.array(read(ROOT/'runs/z13_local_peak_20261021/geometry.json')['points'],int);yy,xx=np.indices((391,391));nearest=cKDTree(pts).query(np.c_[xx.ravel(),yy.ravel()])[1].reshape(391,391);sup=cv2.fillConvexPoly(np.zeros((391,391),'uint8'),pts[ConvexHull(pts).vertices],1).astype(bool);pred={};weights={};oracle=[];labels={};coverage=[];support=sup.ravel();cells=nearest.ravel()
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
def counts(t):
 return np.stack([np.bincount(cells[(t.ravel()==cl)&support],minlength=721) for cl in [0,1,2]],1)
for q in fit+val:
 key=q['id'];t=z[key+'__truth'];cnt=counts(t);tot=cnt.sum(1);soft=cnt/np.maximum(tot[:,None],1);visible=tot>0;labels[key]=(soft,visible);coverage.append(dict(key=key,split=q['split'],hard_valid=int((z[key+'__labels']>=0).sum()),with_any_label=int(visible.sum()),with_mixed_labels=int((np.sum(cnt>0,1)>1).sum()),all_counts=cnt.tolist()))
 if q['split']=='val':
  pos=cnt[:,1];neg=tot-pos;# Max Jaccard among complete cells; monotone sort by target/non-target mass.
  ratio=np.divide(pos,neg,out=np.where(pos>0,np.inf,0.).astype(float),where=neg>0);order=np.argsort(-ratio,kind='stable');cum_pos=np.cumsum(pos[order]);cum_neg=np.cumsum(neg[order]);obj=cum_pos/(pos.sum()+cum_neg);k=int(np.argmax(obj))+1;optimal=np.zeros(721,bool);optimal[order[:k]]=True;mm=optimal[nearest];P=(t==1)&sup;N=(t>=0)&(t!=1)&sup;perfect=float(obj[k-1]);assert abs(perfect-int((mm&P).sum())/int((mm&N).sum()+P.sum()))<1e-12
  # Independent fractional maximum certificate: each selected cell must satisfy positive/negative >= achieved optimum.
  assert np.all((pos-perfect*neg)[optimal]>=-1e-9) and np.all((pos-perfect*neg)[~optimal]<=1e-9)
  oracle.append(dict(key=key,video=q['video'],physical=q['physical'],max_iou=perfect,cell_count=k));pred['oracle__'+key]=optimal
for arm in ['Fly','Retina','Geometry','RGB']:
 xx0=[];xx1=[];yy1=[];ww=[];legacy=[]
 for q in fit:
  key=q['id'];x=z[key+'__'+arm];y=z[key+'__labels'];soft,valid=labels[key];xx0.append(x[y>=0]);xx1.append(x[valid]);yy1.append(soft[valid]);cls=soft[valid].sum(0);present=cls>0;nr=sum(r['video']==q['video'] for r in fit);factor=1/(len(videos)*nr*present.sum());w=soft[valid]*np.divide(1.,cls,out=np.zeros(3),where=present)*factor;ww.append(w)
 # Original standardization held fixed so replacing label support cannot alter representation scaling.
 mean=np.concatenate(xx0).mean(0);sd=np.concatenate(xx0).std(0);sd[sd<1e-6]=1;x=torch.tensor((np.concatenate(xx1)-mean)/sd);tw=torch.tensor(np.concatenate(ww),dtype=torch.float32);assert abs(tw.sum().item()-1)<1e-5
 for seed in [42,43,44]:
  torch.manual_seed(seed);net=torch.nn.Sequential(torch.nn.Linear(70,32),torch.nn.ReLU(),torch.nn.Linear(32,3));opt=torch.optim.Adam(net.parameters(),lr=.01)
  for _ in range(200):
   opt.zero_grad();loss=-(torch.log_softmax(net(x),1)*tw).sum();loss.backward();opt.step()
  name=f'{arm}__{seed}';weights[name+'__mean']=mean;weights[name+'__sd']=sd
  for k,u in net.state_dict().items():weights[name+'__'+k]=u.detach().numpy()
  with torch.no_grad():
   for q in val:pred[name+'__'+q['id']]=net(torch.tensor((z[q['id']+'__'+arm]-mean)/sd)).numpy()
 print('SOFT_CELL_TRAIN',arm,flush=True)
for n,a in [('predictions.npz',pred),('weights.npz',weights)]:
 with (OUT/n).open('xb') as f:np.savez_compressed(f,**a)
save('grid.json',dict(oracle=oracle,coverage=coverage,verification='Every queried cell Jaccard optimum certified, only for representational upper bound. Fit soft distributions see training images only.'))
