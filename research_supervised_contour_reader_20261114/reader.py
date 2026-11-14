from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,numpy as np,time
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);p=read(OUT/'protocol.json');data=read(OUT/'dataset.json');assert sha(OUT/'dataset.npz')==data['sha256'];assert data['n_train_videos']>=4 and data['n_val_videos']>=2,'NOT READY: preserve insufficient split';arr=np.load(OUT/'dataset.npz');rows=data['records'];fit=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val'];videos=set(r['video'] for r in fit);assert not videos&{r['video'] for r in val};pred={};models={};status=[];start=time.monotonic()
for mode in ['labels','proxy']:
 for arm in ['Fly','Retina','RGB','Geometry']:
  xx=[];yy=[];ww=[]
  for r in fit:
   key=r['id'];y=arr[key+'__'+mode];k=y>=0;x=arr[key+'__'+arm];xx.append(x[k]);yy.append(y[k]);classes,counts=np.unique(y[k],return_counts=True);b={cl:co for cl,co in zip(classes,counts)};nr=sum(a['video']==r['video'] for a in fit);ww.append([1/(len(videos)*nr*len(classes)*b[label]) for label in y[k]])
  x=np.concatenate(xx);yy=np.concatenate(yy);w=torch.tensor(np.concatenate(ww),dtype=torch.float32);assert abs(w.sum().item()-1)<1e-5;mu=x.mean(0);sd=x.std(0);sd[sd<1e-6]=1;x=torch.tensor((x-mu)/sd);y=torch.tensor(yy,dtype=torch.long)
  for seed in p['seeds']:
   torch.manual_seed(seed);net=torch.nn.Sequential(torch.nn.Linear(70,32),torch.nn.ReLU(),torch.nn.Linear(32,3));opt=torch.optim.Adam(net.parameters(),lr=.01)
   for _ in range(200):
    opt.zero_grad();v=net(x);loss=(torch.nn.functional.cross_entropy(v,y,reduction='none')*w).sum();loss.backward();opt.step()
   prefix=f'{mode}__{arm}__{seed}';models[prefix+'__mu']=mu;models[prefix+'__sd']=sd
   for n,val0 in net.state_dict().items():models[prefix+'__'+n]=val0.detach().numpy()
   with torch.no_grad():
    for q in val:pred[prefix+'__'+q['id']]=net(torch.tensor((arr[q['id']+'__'+arm]-mu)/sd)).numpy()
   status.append(dict(source=mode,arm=arm,seed=seed,last_train_loss=float(loss.detach())));print('TRAINED',prefix,flush=True)
for file,z in [('models.npz',models),('predictions.npz',pred)]:
 with (OUT/file).open('xb') as f:np.savez_compressed(f,**z)
save('execution_result.json',dict(passed=True,models=status,seconds=time.monotonic()-start,no_core_calls=True,query_labels_loaded_by_training=False,sha256=sha(OUT/'predictions.npz')))
