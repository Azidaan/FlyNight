from lab import ROOT, OUT, GAPS, read, write, fingerprint, now
from pathlib import Path
import hashlib
import zipfile
import numpy as np
from transformer_test import Association, decide, stats, torch
from geometry_data import prepare_scenes, make_features, sources, VAL, assign, ious, extrapolate


def export_head(model):
    # Export the exact trained CPU head for a process which also hosts the GPU detector.
    shape=torch.zeros(2,6,14)
    with torch.inference_mode():
        traced=torch.jit.trace(model,shape,strict=True,check_trace=False)
        for m,n in ((1,1),(2,7),(13,19),(4,52)):
            x=torch.from_numpy(np.random.default_rng(m+n).normal(size=(m,n,14)).astype("float32"))
            torch.testing.assert_close(traced(x),model(x),rtol=1e-5,atol=1e-5)
        traced.save(str(OUT/"attention_runtime.pt"))
    return traced


def main():
    previous=read(OUT/"transformer_inputs_v2.json")["test"]
    with zipfile.ZipFile(ROOT/read(GAPS/"protocol.json")["labelzip"]) as z:
        mots={Path(n).stem for n in z.namelist() if "/rles/val/" in n and n.endswith(".json")}
    order=sorted((p.stem for p in (VAL/"val").glob("*.npz") if p.stem not in mots),key=lambda s:hashlib.sha256(s.encode()).hexdigest())
    assert order[:20]==previous
    extra=order[20:40]
    assert len(extra)==20 and not set(extra)&set(previous)
    inputs=[OUT/"confirmation_protocol.json",OUT/"confirmation.py",OUT/"geometry_data.py",OUT/"transformer_test.py",
            OUT/"attention.pt",OUT/"local.pt",OUT/"transformer_inputs_v2.json"]
    inputs.extend(p for v in extra for p in sources(v,False))
    stamp=fingerprint(inputs)
    write(OUT/"confirmation_inputs.json",dict(stamp,videos=extra))
    models={key:Association(key=="attention").eval() for key in ("attention","local")}
    for k,m in models.items():
        m.load_state_dict(torch.load(OUT/(k+".pt"),weights_only=True,map_location="cpu"))
    traced=export_head(models["attention"])
    rows=[]
    checked=0
    with torch.inference_mode():
        for v in extra:
            for s in prepare_scenes(v,False):
                h,b=s["past"],s["boxes"]
                geo=decide(s)
                local=decide(s,models["local"])
                att=decide(s,models["attention"])
                final=np.full(len(h),-1,np.int64)
                if len(h) and len(b):
                    features=torch.from_numpy(make_features(h,b,s["conf"]))
                    native=models["attention"](features)
                    deployed=traced(features)
                    torch.testing.assert_close(native,deployed,rtol=1e-5,atol=1e-5)
                    sc=(deployed[:,:-1]-deployed[:,-1:]).numpy()
                    assert np.array_equal(assign(sc),att)
                    active=np.flatnonzero(geo>=0)
                    final[active]=assign(sc[active])
                    checked+=1
                hard=(ious(extrapolate(h)[0],b)>0).sum(1)>=2
                for i,y in enumerate(s["y"]):
                    rows.append(dict(video=v,frame=s["frame"],owner=i,truth=int(y),competition=bool(hard[i]),
                                     geometry=int(geo[i]),local=int(local[i]),attention=int(att[i]),guard=int(final[i])))
    results=stats(rows)
    results["guard"]=stats([dict(r,attention=r["guard"]) for r in rows])["attention"]
    assert all(r["geometry"]>=0 or r["guard"]<0 for r in rows)
    assert stamp["sha256"]==fingerprint(inputs)["sha256"]
    write(OUT/"confirmation_rows.json",rows)
    write(OUT/"confirmation_result.json",dict(ended=now(),stats=results,verified_export_frames=checked,
            test=extra,all_inputs_unchanged=True))
    print(results,flush=True)


if __name__=="__main__":
    main()
