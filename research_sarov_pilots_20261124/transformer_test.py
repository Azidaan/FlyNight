from lab import OUT, ROOT, read, write, fingerprint, now
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import hashlib
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from geometry_data import TRAIN, VAL, FEATURES, prepare_scenes, split, sources, make_features, extrapolate, ious, assign

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Association(nn.Module):
    def __init__(self,attention):
        super().__init__()
        self.embed=nn.Linear(FEATURES,32)
        self.wait=nn.Parameter(torch.zeros(1,1,32))
        self.attention=attention
        if attention:
            self.block=nn.TransformerEncoderLayer(32,4,64,dropout=0.,activation="gelu",batch_first=True)
        else:
            self.block=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,32),nn.LayerNorm(32))
        self.readout=nn.Linear(32,1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self,x,padding=None):
        token=torch.cat([self.embed(x),self.wait.expand(x.shape[0],1,32)],dim=1)
        if self.attention:
            token=self.block(token,src_key_padding_mask=padding)
        else:
            token=self.block(token)
        values=self.readout(token).squeeze(-1)
        # Fixed reference rule is exactly zero learned residual. Last column means wait.
        logits=torch.cat((values[:,:-1]+8.*(x[:,:,4]-.5),values[:,-1:]),dim=1)
        return logits.masked_fill(padding,-1.e4) if padding is not None else logits


def training_rows(scenes):
    groups=defaultdict(list)
    for scene in scenes:
        if not len(scene["past"]) or not len(scene["boxes"]):
            continue
        x=make_features(scene["past"],scene["boxes"],scene["conf"])
        for f,y in zip(x,scene["y"]):
            groups[(len(f)+7)//8].append((f,int(y)))
    return groups


def fit(groups,attention):
    torch.manual_seed(42)
    rng=np.random.default_rng(42)
    model=Association(attention)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    losses=[]
    for ep in range(8):
        order=[]
        for group,data in sorted(groups.items()):
            idx=rng.permutation(len(data))
            order.extend((group,idx[k:k+64]) for k in range(0,len(idx),64))
        rng.shuffle(order)
        total=0.; n=0
        for group,idx in order:
            rows=[groups[group][i] for i in idx]
            size=max(len(f) for f,y in rows)
            batch=np.zeros((len(rows),size,FEATURES),np.float32)
            mask=np.ones((len(rows),size+1),bool)
            mask[:,-1]=False
            target=[]
            for i,(f,y) in enumerate(rows):
                batch[i,:len(f)]=f
                mask[i,:len(f)]=False
                target.append(y if y>=0 else size)
            optimizer.zero_grad(set_to_none=True)
            result=model(torch.from_numpy(batch),torch.from_numpy(mask))
            loss=nn.functional.cross_entropy(result,torch.tensor(target))
            loss.backward()
            optimizer.step()
            total+=float(loss.detach())*len(rows)
            n+=len(rows)
        losses.append(total/n)
        print("FIT",attention,ep+1,n,round(total/n,4),flush=True)
    return model.eval(),losses


def decide(scene,model=None):
    h,b=scene["past"],scene["boxes"]
    if len(h)==0 or len(b)==0:
        return np.full(len(h),-1,np.int64)
    if model is None:
        scores=8*(ious(extrapolate(h)[0],b)-.5)
    else:
        x=make_features(h,b,scene["conf"])
        values=model(torch.from_numpy(x)).detach().numpy()
        scores=values[:,:-1]-values[:,-1:]
    return assign(scores)


def stats(rows):
    y=np.array([r["truth"] for r in rows])
    hard=np.array([r["competition"] for r in rows],bool)&(y>=0)
    res={}
    for mode in ("geometry","local","attention"):
        a=np.array([r[mode] for r in rows])
        res[mode]={"total":len(y),"correct_all":int((a==y).sum()),"eligible_competing":int(hard.sum()),
                   "correct_competing":int(np.sum((a==y)&hard)),"missing":int((y<0).sum()),
                   "false_on_missing":int(((y<0)&(a>=0)).sum()),"wrong_owner":int(((a>=0)&(y>=0)&(a!=y)).sum()),
                   "wait_on_present":int(((a<0)&(y>=0)).sum()),"present":int((y>=0).sum())}
    return res


def timing(scenes,models):
    times=[]
    sample=scenes[::max(1,len(scenes)//320)][:360]
    keys=list(models)
    for s in sample[:20]:
        for m in models.values():
            decide(s,m)
    for rep in range(3):
        for i,s in enumerate(sample):
            shift=(rep+i)%len(keys)
            row=dict(video=s["video"],frame=s["frame"],rep=rep,owners=len(s["past"]),candidates=len(s["boxes"]))
            for key in keys[shift:]+keys[:shift]:
                t=time.perf_counter_ns()
                decide(s,models[key])
                row[key]=(time.perf_counter_ns()-t)/1e6
            times.append(row)
    dist={}
    arrays={k:np.array([r[k] for r in times]) for k in keys}
    arrays["attention_minus_geometry"]=arrays["attention"]-arrays["geometry"]
    for k,v in arrays.items():
        dist[k]={"mean":float(v.mean()),"p50":float(np.median(v)),"p95":float(np.quantile(v,.95)),
                 "p99":float(np.quantile(v,.99))}
    return times,dist


def main():
    tr,va=split()
    inputs=[OUT/"protocol.json",OUT/"lab.py",OUT/"geometry_data.py",Path(__file__),TRAIN/"selection.json",VAL/"meta.json",
            ROOT/"runs/runtime_benchmark/tracker_runtime.json",ROOT/"weights/baseline_seed42/best.pt"]
    for videos,training in ((tr,True),(va,False)):
        for v in videos:
            inputs.extend(sources(v,training))
    prov=fingerprint(inputs)
    prov.update(train=tr,test=va,torch=torch.__version__,seed=42,threads=torch.get_num_threads())
    write(OUT/"transformer_inputs_v2.json",prov)
    training=[s for v in tr for s in prepare_scenes(v,True)]
    groups=training_rows(training)
    models={"geometry":None}
    losses={}
    for mode in ("local","attention"):
        models[mode],losses[mode]=fit(groups,mode=="attention")
        torch.save(models[mode].state_dict(),OUT/(mode+".pt"))
    test=[s for v in va for s in prepare_scenes(v,False)]
    rows=[]
    with torch.inference_mode():
        for s in test:
            decisions={k:decide(s,m) for k,m in models.items()}
            hard=(ious(extrapolate(s["past"])[0],s["boxes"])>0).sum(1)>=2
            for i,y in enumerate(s["y"]):
                rows.append(dict(video=s["video"],frame=s["frame"],owner=i,truth=int(y),competition=bool(hard[i]),
                                 **{k:int(a[i]) for k,a in decisions.items()}))
        times,dist=timing(test,models)
    write(OUT/"transformer_rows.json",rows)
    write(OUT/"transformer_timings.json",times)
    per_video={v:stats([r for r in rows if r["video"]==v]) for v in va}
    rng=np.random.default_rng(42)
    scores=np.array([(s["attention"]["correct_competing"]-s["geometry"]["correct_competing"])/s["geometry"]["eligible_competing"]
                     for s in per_video.values() if s["geometry"]["eligible_competing"]])
    interval=np.quantile(rng.choice(scores,(10000,len(scores))).mean(1),[.025,.975]).tolist()
    assert fingerprint(inputs)["sha256"]==prov["sha256"]
    write(OUT/"transformer_result.json",dict(ended=now(),inputs_unchanged=True,train_videos=len(tr),test_videos=len(va),
          train_frames=len(training),test_frames=len(test),train_rows=sum(map(len,groups.values())),stats=stats(rows),
          per_video=per_video,paired_mean_video=float(scores.mean()),interval=interval,timings=dist,
          profile_frames=len(times)//3,weights={k:sum(p.numel() for p in m.parameters()) for k,m in models.items() if m},
          losses=losses,scope="Causal last boxes and YOLO candidates, reference-associated past histories, one-step assignment. CPU timing includes features and one-to-one assignment; excludes detector, FlyVis and full tracker."))
    print("MEASURED",len(rows),dist,flush=True)


if __name__=="__main__":
    main()
