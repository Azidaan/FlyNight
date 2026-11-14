from data import *
import time
from PIL import Image,ImageDraw,ImageFont
p=read(OUT/'protocol.json')
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
records={r['id']:load_record(r) for r in p['records']};outputs={};fitted={};readiness=[];start=time.perf_counter()
for case in p['cases']:
    ref=records[case['reference']];query=records[case['query']];yy=ref['cell_labels'];counts=np.bincount(yy[yy>=0],minlength=3);ready=counts.min()>=3;readiness.append(dict(case=case['id'],class_counts=counts.tolist(),ready=bool(ready),primary=case['primary']))
    if not ready:continue
    for arm,fullref in ref['features'].items():
        mu=fullref.mean(0);sd=fullref.std(0);sd[sd<1e-8]=1;allx=(fullref-mu)/sd;queryx=(query['features'][arm]-mu)/sd;valid=yy>=0;x=allx[valid];y=yy[valid]
        for control in ['correct','wrong_target','shuffled_labels']:
            target=y.copy()
            if control=='wrong_target':target=np.choose(target,[0,2,1])
            elif control=='shuffled_labels':target=np.random.default_rng(42).permutation(target)
            for reader in ['ridge','prototype']:
                prefix=case['id']+'__'+arm+'__'+control+'__'+reader
                if reader=='ridge':
                    xx=np.c_[x,np.ones(len(x))];weight=1/(3*np.bincount(target,minlength=3)[target]);reg=np.eye(xx.shape[1]);reg[-1,-1]=0;coef=np.linalg.solve(xx.T@(xx*weight[:,None])+reg,xx.T@(np.eye(3)[target]*weight[:,None]));q=np.c_[queryx,np.ones(len(queryx))]@coef;scores_ref=np.c_[allx,np.ones(len(allx))]@coef;fitted[prefix+'__coef']=coef
                else:
                    norm=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8);protos=np.stack([norm[target==k].mean(0) for k in range(3)]);protos/=np.maximum(np.linalg.norm(protos,axis=1,keepdims=True),1e-8);q=(queryx/np.maximum(np.linalg.norm(queryx,axis=1,keepdims=True),1e-8))@protos.T;scores_ref=(allx/np.maximum(np.linalg.norm(allx,axis=1,keepdims=True),1e-8))@protos.T;fitted[prefix+'__prototypes']=protos
                assert np.isfinite(q).all();outputs[prefix+'__query']=q;outputs[prefix+'__reference']=scores_ref
        fitted[case['id']+'__'+arm+'__mu']=mu;fitted[case['id']+'__'+arm+'__sd']=sd
with (OUT/'scores.npz').open('xb') as f:np.savez_compressed(f,**outputs)
with (OUT/'readers.npz').open('xb') as f:np.savez_compressed(f,**fitted)
# Frozen predictions above; query truth is only accessed in the evaluator below.
results=[];rasters={};font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
for case in p['cases']:
    if not next(r['ready'] for r in readiness if r['case']==case['id']):continue
    for key,scores in outputs.items():
        cid,arm,control,reader,part=key.split('__')
        if cid!=case['id']:continue
        d=records[case[part]];result,raster=measure(scores.argmax(1),d);results.append(dict(case=cid,arm=arm,control=control,reader=reader,part=part,**result));rasters[key]=raster
    qd=records[case['query']]
    for reader in ['ridge','prototype']:
        sheet=Image.new('RGB',(1564,451),'#20232a');draw=ImageDraw.Draw(sheet)
        for j,arm in enumerate(['Original','FlyVis+G','Retina+G','Geometry']):
            im=qd['image'].copy()
            if arm!='Original':
                labels=rasters[case['id']+'__'+arm+'__correct__'+reader+'__query']
                for cls,col in [(1,[0,220,0]),(2,[30,70,240])]:m=(labels==cls)&SUPPORT;im[m]=(im[m]*.45+np.array(col)*.55).astype(np.uint8)
            sheet.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),(j*391,36));draw.text((j*391+4,7),arm+' / '+reader,font=font,fill='white')
        draw.text((5,430),'Зелёное — предсказанная цель; оранжевое — сосед. Цель задана только прошлым размеченным кадром.',font=font,fill='white');sheet.save(OUT/(case['id']+'_'+reader+'_masks.png'))
with (OUT/'evaluation_arrays.npz').open('xb') as f:np.savez_compressed(f,**rasters,**{'truth__'+rid:d['truth'] for rid,d in records.items()},**{'inside__'+rid:d['inside'] for rid,d in records.items()},support=SUPPORT)
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
save('result.json',dict(records=results,readiness=readiness,seconds=time.perf_counter()-start,scores_sha256=sha(OUT/'scores.npz'),readers_sha256=sha(OUT/'readers.npz'),source_unchanged=True,query_labels_used_in_fit=False))
print('CONDITIONED_READOUT_FINISHED',readiness,len(results),flush=True)
