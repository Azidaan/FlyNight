from common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch,time
from data import load_record,measure
torch.set_default_device('cpu');torch.set_num_threads(2)
if sys.argv[1]=='prepare':
    p=read(OUT/'protocol.json')
    save('color_protocol.json',dict(registered=now(),parent_sha256=sha(OUT/'protocol.json'),reason='Post-result diagnostic: nonlinear RGB+G passes2179recall/insideFPcondition while nonlinear FlyVis+Gdoes not. Test removal of chromatic information and matched addition of C34; do not infer all failures originate in grayscale.',inputs={**p['inputs'],rel(OUT/'capacity_scores.npz'):sha(OUT/'capacity_scores.npz'),rel(OUT/'capacity_readers.npz'):sha(OUT/'capacity_readers.npz'),rel(OUT/'capacity_result.json'):sha(OUT/'capacity_result.json'),rel(OUT/'readers.npz'):sha(OUT/'readers.npz'),rel(Path(__file__)):sha(__file__)},method='Same per-reference MLP d-64-3,500epochs,Adam0.003,seeds42/43/44mean, class-balanced CE, reference-only mean/std. Reuse completed RGB8predictions rather than retraining.',arms=['RGB8cached','Gray8retrained: replace BGRself/local means by three copies of0.114B+0.587G+0.299R, sameG; no rounding','Gray8frozen: same desaturated vectors through originalRGBweights and originalRGBnormalization, separately marked distribution-shift control','ZeroFlyRGB76:68zero columns+RGB6+G2','FlyRGB76:C34self34+local34+RGB6+G2','ZeroFlyGray76:68zero columns+Gray6+G2','FlyGray76:C34self34+local34+Gray6+G2'],questions=['How do known-targetIoU, recall and within-ROI false positives change when color alone is removed with the same training policy?','At equal76inputdimensions and same initialization does C34add to RGB or grayscale? Report every case, including negative effects.'],limits='Exploratory after seeing primary failures. Cannot promote based on these familiar outcomes, cannot equate a useful RGB branch to improved FlyVis core. Undefined insideFPremainsNone when no annotated negative exists. No threshold tuning or core runs.',deadline='2026-11-14T08:55:00+03:00'))
    journal('Зарегистрирована проверка цвета внутри reference_conditioned_masks_20261114: одинаковые признаки и обучение при сохранении либо удалении цветовых различий; добавлениеC34кRGBсравнивается с68нулевыми каналами при одинаковой архитектуре76входов. Источник: color_protocol.json. Уже рассчитанныйRGB8не повторяется. Это последующая диагностика известных случаев, не новая независимая проверка переноса.')
    raise SystemExit
p=read(OUT/'protocol.json');cp=read(OUT/'color_protocol.json')
for n,h in cp['inputs'].items():assert sha(ROOT/n)==h,n
records={r['id']:load_record(r) for r in p['records']};cached=np.load(OUT/'capacity_scores.npz');previous=np.load(OUT/'capacity_readers.npz');normalization=np.load(OUT/'readers.npz');outputs={};weights={};losses=[];begin=time.perf_counter()
def variants(d):
    rgb=d['features']['RGB+G'];gray=rgb.copy();gray[:,:3]=(rgb[:,:3]@np.array([.114,.587,.299]))[:,None];gray[:,3:6]=(rgb[:,3:6]@np.array([.114,.587,.299]))[:,None];fly=d['features']['FlyVis'];zeros=np.zeros_like(fly)
    return dict(RGB8=rgb,Gray8=gray,ZeroFlyRGB76=np.c_[zeros,rgb],FlyRGB76=np.c_[fly,rgb],ZeroFlyGray76=np.c_[zeros,gray],FlyGray76=np.c_[fly,gray])
for case in [c for c in p['cases'] if c['primary']]:
    ref=records[case['reference']];query=records[case['query']];rf=variants(ref);qf=variants(query);yall=ref['cell_labels'];valid=yall>=0;y=yall[valid];counts=np.bincount(y,minlength=3);assert counts.min()>=3
    for part in ['reference','query']:outputs[case['id']+'__RGB8__'+part]=cached[case['id']+'__RGB+G__mlp__'+part]
    frozen={part:[] for part in ['reference','query']};mu=normalization[case['id']+'__RGB+G__mu'];sd=normalization[case['id']+'__RGB+G__sd']
    for seed in [42,43,44]:
        pref=case['id']+'__RGB+G__mlp__'+str(seed)+'__'
        for part,raw in [('reference',rf['Gray8']),('query',qf['Gray8'])]:
            x=((raw-mu)/sd).astype('float32');h=np.maximum(0,x@previous[pref+'0.weight'].T+previous[pref+'0.bias']);score=h@previous[pref+'2.weight'].T+previous[pref+'2.bias'];score-=score.max(1,keepdims=True);prob=np.exp(score);frozen[part].append(prob/prob.sum(1,keepdims=True))
    for part,values in frozen.items():outputs[case['id']+'__Gray8frozen__'+part]=np.mean(values,0)
    for arm in ['Gray8','ZeroFlyRGB76','FlyRGB76','ZeroFlyGray76','FlyGray76']:
        mu=rf[arm].mean(0);sd=rf[arm].std(0);sd[sd<1e-8]=1;rx=(rf[arm]-mu)/sd;qx=(qf[arm]-mu)/sd;x=torch.tensor(rx[valid],dtype=torch.float32,device='cpu');yt=torch.tensor(y,dtype=torch.long,device='cpu');wt=torch.tensor(1/(3*counts[y]),dtype=torch.float32,device='cpu');rs=[];qs=[]
        prefix=case['id']+'__'+arm;weights[prefix+'__mu']=mu;weights[prefix+'__sd']=sd
        for seed in [42,43,44]:
            torch.manual_seed(seed);model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,3)).to('cpu');opt=torch.optim.Adam(model.parameters(),lr=.003)
            for epoch in range(500):
                opt.zero_grad();loss=(torch.nn.functional.cross_entropy(model(x),yt,reduction='none')*wt).sum();loss.backward();opt.step()
            assert torch.isfinite(loss);losses.append(dict(case=case['id'],arm=arm,seed=seed,loss=float(loss.detach())))
            for k,v in model.state_dict().items():weights[prefix+'__'+str(seed)+'__'+k]=v.detach().cpu().numpy()
            with torch.no_grad():rs.append(torch.softmax(model(torch.tensor(rx,dtype=torch.float32,device='cpu')),1).cpu().numpy());qs.append(torch.softmax(model(torch.tensor(qx,dtype=torch.float32,device='cpu')),1).cpu().numpy())
        outputs[prefix+'__reference']=np.mean(rs,0);outputs[prefix+'__query']=np.mean(qs,0);print(case['id'],arm,'done',flush=True)
with (OUT/'color_scores.npz').open('xb') as f:np.savez_compressed(f,**outputs)
with (OUT/'color_readers.npz').open('xb') as f:np.savez_compressed(f,**weights)
rows=[];refconf=[]
for key,score in outputs.items():
    cid,arm,part=key.split('__');case=next(c for c in p['cases'] if c['id']==cid);d=records[case[part]];metric,_=measure(score.argmax(1),d)
    if metric['inside_negative_pixels']==0:metric['inside_fpr']=None
    rows.append(dict(case=cid,arm=arm,part=part,**metric))
    if part=='reference':
        y=d['cell_labels'];v=y>=0;cm=np.bincount(y[v]*3+score.argmax(1)[v],minlength=9).reshape(3,3);refconf.append(dict(case=cid,arm=arm,confusion=cm.tolist()))
for n,h in cp['inputs'].items():assert sha(ROOT/n)==h,n
save('color_result.json',dict(records=rows,reference_confusions=refconf,losses=losses,source_unchanged=True,seconds=time.perf_counter()-begin,scope='Same remembered cases, post-result mechanistic diagnostics. No promotion or arbitrary new-video claim.'))
print('COLOR_COMPLETE')
