from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,time,torch
from data import load_record,measure,NEAREST,SUPPORT
torch.set_default_device('cpu');torch.set_num_threads(2)
if sys.argv[1]=='prepare':
    p=read(OUT/'protocol.json');a=read(OUT/'audit.json')
    save('capacity_protocol.json',dict(registered=now(),parent_sha256=sha(OUT/'protocol.json'),reason='Post-result diagnostic: correct-reference linear FlyVis+G misclassifies96of563clean background reference cells on2179. Two other query ROIs have zero labeled negatives, so their0FPR is undefined. No original predictions or criteria changed.',inputs={**p['inputs'],rel(OUT/'result.json'):sha(OUT/'result.json'),rel(OUT/'audit.json'):sha(OUT/'audit.json'),rel(Path(__file__)):sha(__file__)},methods=['Unregularized class-balanced least squares on exactly same standardized reference features and labels, SVD cutoff1e-12. Isolates removal of ridge regularization.','Nonlinear per-reference MLP d-64-3 ReLU, Adam0.003,500epochs,seeds42/43/44 averaged softmax, same reference-only normalization and class weights. Changes function class and training objective together; do not claim single-factor complexity causality.','Oracle best possible binary targetIoU for nearest-cell raster on known query pixels, using query truth ONLY for a marked diagnostic upper bound. Sort per-cell target/negative ratio; certify fractional optimum. Not a model or deployable mask.'],acceptance='No deployment gate or selection. Report every arm/case, support-cell and query errors, and oracle bound. All primary data/thresholds remain fixed. No added trainingframes, core inference or tracker.',deadline='2026-11-14T08:55:00+03:00'))
    journal('В reference_conditioned_masks_20261114 найдена подтверждённая ошибка оценки: ноль размеченных помех внутри двух рамок давал0ложных захватов; в audit.json и summary.json это помечено как неопределённое, прежнийresult.jsonсохранён. На2179линейныйFlyVis+Gошибается в96из563чистых фоновых ячеек уже прошлого обучающего кадра. До новых предсказаний зарегистрированы снятие регуляризации, более гибкий считыватель и идеальная верхняя граница самой решётки; capacity_protocol.json. Эти диагностики не заменяют исходный допуск.')
    raise SystemExit
p=read(OUT/'protocol.json');cp=read(OUT/'capacity_protocol.json')
for n,h in cp['inputs'].items():assert sha(ROOT/n)==h,n
records={r['id']:load_record(r) for r in p['records']};prior=np.load(OUT/'readers.npz');outputs={};weights={};fitting=[];oracles=[];start=time.perf_counter()
for case in [c for c in p['cases'] if c['primary']]:
    ref=records[case['reference']];query=records[case['query']];labels=ref['cell_labels'];valid=labels>=0;y=labels[valid];counts=np.bincount(y,minlength=3);assert counts.min()>=3
    for arm,raw in ref['features'].items():
        mu=prior[case['id']+'__'+arm+'__mu'];sd=prior[case['id']+'__'+arm+'__sd'];rx=(raw-mu)/sd;qx=(query['features'][arm]-mu)/sd;x=rx[valid];w=1/(3*counts[y]);xx=np.c_[x,np.ones(len(x))];coef=np.linalg.lstsq(xx*np.sqrt(w[:,None]),np.eye(3)[y]*np.sqrt(w[:,None]),rcond=1e-12)[0];prefix=case['id']+'__'+arm
        weights[prefix+'__unregularized']=coef
        for part,feat in [('reference',rx),('query',qx)]:outputs[prefix+'__unregularized__'+part]=np.c_[feat,np.ones(len(feat))]@coef
        tensor=torch.tensor(x,dtype=torch.float32,device='cpu');yt=torch.tensor(y,dtype=torch.long,device='cpu');wt=torch.tensor(w,dtype=torch.float32,device='cpu');rs=[];qs=[]
        for seed in [42,43,44]:
            torch.manual_seed(seed);model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,3)).to('cpu');opt=torch.optim.Adam(model.parameters(),lr=.003)
            for epoch in range(500):
                opt.zero_grad();loss=(torch.nn.functional.cross_entropy(model(tensor),yt,reduction='none')*wt).sum();loss.backward();opt.step()
            assert torch.isfinite(loss)
            fitting.append(dict(case=case['id'],arm=arm,seed=seed,loss=float(loss.detach())))
            for k,v in model.state_dict().items():weights[prefix+'__mlp__'+str(seed)+'__'+k]=v.detach().cpu().numpy()
            with torch.no_grad():
                rs.append(torch.softmax(model(torch.tensor(rx,dtype=torch.float32,device='cpu')),1).cpu().numpy());qs.append(torch.softmax(model(torch.tensor(qx,dtype=torch.float32,device='cpu')),1).cpu().numpy())
        outputs[prefix+'__mlp__reference']=np.mean(rs,0);outputs[prefix+'__mlp__query']=np.mean(qs,0);print(case['id'],arm,'capacity complete',flush=True)
with (OUT/'capacity_scores.npz').open('xb') as f:np.savez_compressed(f,**outputs)
with (OUT/'capacity_readers.npz').open('xb') as f:np.savez_compressed(f,**weights)
rows=[];support=[]
for key,score in outputs.items():
    cid,arm,method,part=key.split('__');case=next(c for c in p['cases'] if c['id']==cid);d=records[case[part]];metric,_=measure(score.argmax(1),d)
    if metric['inside_negative_pixels']==0:metric['inside_fpr']=None
    rows.append(dict(case=cid,arm=arm,method=method,part=part,**metric))
    if part=='reference':
        y=d['cell_labels'];known=y>=0;cm=np.bincount(y[known]*3+score.argmax(1)[known],minlength=9).reshape(3,3);support.append(dict(case=cid,arm=arm,method=method,confusion=cm.tolist()))
for case in [c for c in p['cases'] if c['primary']]:
    for part in ['reference','query']:
        d=records[case[part]];truth=d['truth'];known=SUPPORT&(truth>=0);ti=np.bincount(NEAREST[known&(truth==1)],minlength=721);fi=np.bincount(NEAREST[known&(truth!=1)],minlength=721);ratio=np.divide(ti,fi,out=np.full(721,np.inf),where=fi>0);ratio[(ti==0)&(fi==0)]=0;order=np.argsort(-ratio,kind='stable');tp=np.cumsum(ti[order]);fp=np.cumsum(fi[order]);ious=tp/(ti.sum()+fp);best=int(np.argmax(ious));j=float(ious[best]);residual=float(np.maximum(ti-j*fi,0).sum()-j*ti.sum());assert abs(residual)<1e-6
        oracles.append(dict(case=case['id'],part=part,max_known_target_iou=j,optimality_residual=residual,selected_cells=order[:best+1].tolist(),target_pixels_per_cell=ti.tolist(),negative_pixels_per_cell=fi.tolist(),not_model_result=True))
for n,h in cp['inputs'].items():assert sha(ROOT/n)==h,n
save('capacity_result.json',dict(records=rows,reference_confusions=support,oracle=oracles,fitting=fitting,source_unchanged=True,seconds=time.perf_counter()-start,scope='Post-result diagnostics; query masks only for evaluation/oracle. No promotion or change to original acceptance.'))
print('CAPACITY_FINISHED',len(rows),flush=True)
