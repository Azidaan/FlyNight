from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch,time
from data import load_record,measure,PTS,SUPPORT
torch.set_default_device('cpu');torch.set_num_threads(2)
if sys.argv[1]=='prepare':
    save('position_protocol.json',dict(registered=now(),reason='Frozen current-C34 replacement by reference-C34 gives much higher queryIoU. Test whether a rich fixed coordinate code or copying previous masks explains this without new neural evidence.',inputs={**read(OUT/'protocol.json')['inputs'],rel(OUT/'color_scores.npz'):sha(OUT/'color_scores.npz'),rel(OUT/'tensor_result.json'):sha(OUT/'tensor_result.json'),rel(Path(__file__)):sha(__file__)},controls=['FourierRGB76: replace68C34features by68deterministic position features: sin/cos on x/y at17geomspacefrequencies0.5..8cycles per391eye. Same dimensions/MLP64/Adam.003/500epochs/seeds42,43,44/reference-only fitting as prior76models. One fixed map, no frequency search.','Copy previous FlyRGB76 reference predicted cell labels to same query cell indices, no query observation.','Copy previous ZeroFlyRGB76 reference predicted cell labels to same query cell indices, no query observation.','Copy previous manual target raster to same normalized391eye, unknown treated asnot-target. Uses only past manual annotation, no currentmask selection.'],scope='Post-result explanatory controls only. References and queries mostly near in time; apparent accuracy may reflect little motion and partial cores. Fourier literature supports spatial representation, not object identity or our specific frequencies.',literature='https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html',deadline='2026-11-14T08:55:00+03:00'))
    journal('Зарегистрирован позиционный контроль reference_conditioned_masks_20261114: после высокого результата подстановки прошлыхC34проверяются независимый код координат той же размерности и перенос прежних масок без чтения текущего кадра. Источник: position_protocol.json; основание метода — Tancik2020FourierFeatures. Это проверка альтернативного объяснения, не признание распознавания личности.')
    raise SystemExit
p=read(OUT/'protocol.json');pp=read(OUT/'position_protocol.json');saved=np.load(OUT/'color_scores.npz');outputs={};weights={};rows=[];t0=time.perf_counter()
for n,h in pp['inputs'].items():assert sha(ROOT/n)==h,n
pos=PTS.astype(float)/391;freq=np.geomspace(.5,8,17);fourier=np.concatenate([fn(2*np.pi*pos[:,axis,None]*freq) for axis in [0,1] for fn in [np.sin,np.cos]],axis=1);assert fourier.shape==(721,68)
for case in [c for c in p['cases'] if c['primary']]:
    ref=load_record(next(r for r in p['records'] if r['id']==case['reference']));query=load_record(next(r for r in p['records'] if r['id']==case['query']));rf=np.c_[fourier,ref['features']['RGB+G']];qf=np.c_[fourier,query['features']['RGB+G']];mu=rf.mean(0);sd=rf.std(0);sd[sd<1e-8]=1;rx=(rf-mu)/sd;qx=(qf-mu)/sd;labels=ref['cell_labels'];v=labels>=0;y=labels[v];count=np.bincount(y,minlength=3);assert count.min()>=3;x=torch.tensor(rx[v],dtype=torch.float32,device='cpu');yt=torch.tensor(y,dtype=torch.long,device='cpu');wt=torch.tensor(1/(3*count[y]),dtype=torch.float32,device='cpu');rs=[];qs=[]
    for seed in [42,43,44]:
        torch.manual_seed(seed);model=torch.nn.Sequential(torch.nn.Linear(76,64),torch.nn.ReLU(),torch.nn.Linear(64,3)).to('cpu');opt=torch.optim.Adam(model.parameters(),lr=.003)
        for epoch in range(500):
            opt.zero_grad();loss=(torch.nn.functional.cross_entropy(model(x),yt,reduction='none')*wt).sum();loss.backward();opt.step()
        for k,val in model.state_dict().items():weights[case['id']+'__'+str(seed)+'__'+k]=val.detach().cpu().numpy()
        with torch.no_grad():rs.append(torch.softmax(model(torch.tensor(rx,dtype=torch.float32,device='cpu')),1).cpu().numpy());qs.append(torch.softmax(model(torch.tensor(qx,dtype=torch.float32,device='cpu')),1).cpu().numpy())
    weights[case['id']+'__mu']=mu;weights[case['id']+'__sd']=sd;outputs[case['id']+'__reference']=np.mean(rs,0);outputs[case['id']+'__query']=np.mean(qs,0)
    for name,lab in [('FourierRGB76',np.mean(qs,0).argmax(1)),('copy_FlyRGB_reference',saved[case['id']+'__FlyRGB76__reference'].argmax(1)),('copy_ZeroFlyRGB_reference',saved[case['id']+'__ZeroFlyRGB76__reference'].argmax(1))]:
        met,_=measure(lab,query)
        if not met['inside_negative_pixels']:met['inside_fpr']=None
        rows.append(dict(case=case['id'],variant=name,**met))
    pred=ref['truth']==1;truth=query['truth'];known=SUPPORT&(truth>=0);tp=int((known&pred&(truth==1)).sum());fp=int((known&pred&(truth!=1)).sum());fn=int((known&~pred&(truth==1)).sum());negative=known&query['inside']&(truth!=1);n=int(negative.sum());fi=int((pred&negative).sum());rows.append(dict(case=case['id'],variant='copy_manual_reference',tp=tp,fp=fp,fn=fn,iou=tp/max(1,tp+fp+fn),recall=tp/max(1,tp+fn),inside_negative_pixels=n,inside_false_positive=fi,inside_fpr=fi/n if n else None));print(case['id'],'position done',flush=True)
with (OUT/'position_scores.npz').open('xb') as f:np.savez_compressed(f,**outputs)
with (OUT/'position_readers.npz').open('xb') as f:np.savez_compressed(f,**weights)
for n,h in pp['inputs'].items():assert sha(ROOT/n)==h,n
save('position_result.json',dict(records=rows,source_unchanged=True,seconds=time.perf_counter()-t0,scope='Position-memory and no-current-observation controls, not deployment or unseen-video validation.'))
print(json.dumps(rows),flush=True)
