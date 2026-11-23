"""Warm paired partial pipeline: real JPEG, YOLO26s on GPU, CPU assignment."""
from lab import ROOT, OUT, read, write, sha, fingerprint, now
from geometry_data import prepare_scenes, make_features, extrapolate, ious, assign
from pathlib import Path
import time
import cv2
import numpy as np
import psutil
import torch
from tracking.gpu_lock import GpuLock
from ultralytics import YOLO


def dist(arr):
    a=np.array(arr)
    return dict(mean=float(a.mean()),p50=float(np.median(a)),p95=float(np.quantile(a,.95)),
                p99=float(np.quantile(a,.99)))


def main():
    assert (OUT/"confirmation_result.json").is_file() and (OUT/"gap_complete.json").is_file()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    vids=read(OUT/"transformer_inputs_v2.json")["test"]
    cases=[]
    for v in vids:
        scenes=prepare_scenes(v,False)
        gt=read(ROOT/"data/bdd100k/labels/box_track_20/val"/(v+".json"))
        for i in np.linspace(0,len(scenes)-1,4,dtype=int):
            s=scenes[i]
            image=ROOT/"data/bdd100k/images/track/val"/v/gt[s["frame"]]["name"]
            assert image.is_file()
            cases.append((s,image))
    weights=ROOT/"weights/baseline_seed42/best.pt"
    model_path=OUT/"attention_runtime.pt"
    inputs=[s for _,s in cases]+[weights,model_path,OUT/"confirmation_protocol.json",Path(__file__),OUT/"geometry_data.py"]
    prior=fingerprint(inputs)
    write(OUT/"live_inputs.json",prior)
    records=[]
    with GpuLock(r"Local\DissertationSingleGpu"),torch.inference_mode():
        assert torch.cuda.is_available()
        detector=YOLO(str(weights))
        proc=psutil.Process()
        rss0=proc.memory_info().rss
        net=torch.jit.load(str(model_path),map_location="cpu").eval()
        rss1=proc.memory_info().rss
        def run(s,image,mode):
            torch.cuda.synchronize()
            t0=time.perf_counter_ns()
            frame=cv2.imread(str(image))
            assert frame is not None
            rr=detector.predict(frame,imgsz=960,conf=.01,iou=.7,device=0,verbose=False,save=False)[0].boxes
            data=rr.data.detach().cpu().numpy()
            torch.cuda.synchronize()
            selected=(data[:,5]==2)&(data[:,4]>.1)
            b=np.ascontiguousarray(data[selected,:4])
            conf=np.ascontiguousarray(data[selected,4])
            t1=time.perf_counter_ns()
            h=s["past"]
            if not len(h) or not len(b):
                answer=np.full(len(h),-1)
            elif mode=="geometry":
                answer=assign(8*(ious(extrapolate(h)[0],b)-.5))
            else:
                y=net(torch.from_numpy(make_features(h,b,conf))).numpy()
                answer=assign(y[:,:-1]-y[:,-1:])
            t2=time.perf_counter_ns()
            return dict(total=(t2-t0)/1e6,association=(t2-t1)/1e6,
                        jpeg_detector_filter=(t1-t0)/1e6),b,conf,answer
        for i in range(8):
            s,f=cases[i]
            run(s,f,"geometry")
            run(s,f,"attention")
        for i,(s,f) in enumerate(cases):
            order=("geometry","attention") if i%2==0 else ("attention","geometry")
            results={m:run(s,f,m) for m in order}
            a,b=results["geometry"],results["attention"]
            np.testing.assert_array_equal(a[1],b[1]); np.testing.assert_array_equal(a[2],b[2])
            records.append(dict(video=s["video"],frame=s["frame"],owners=len(s["past"]),candidates=len(a[1]),
                                geometry=a[0],attention=b[0],first=order[0]))
    assert fingerprint(inputs)["sha256"]==prior["sha256"]
    summary={m:{k:dist([r[m][k] for r in records]) for k in ("total","association","jpeg_detector_filter")}
             for m in ("geometry","attention")}
    summary["paired_delta"]={k:dist([r["attention"][k]-r["geometry"][k] for r in records]) for k in ("total","association")}
    write(OUT/"live_timings.json",records)
    write(OUT/"live_result.json",dict(ended=now(),frames=len(cases),videos=len(vids),device=torch.cuda.get_device_name(0),
          weights_bytes=sum(p.numel()*p.element_size() for p in net.parameters()),rss_change_on_cpu_model_load=rss1-rss0,
          summary=summary,identical_live_detections=True,inputs_unchanged=True))
    print(summary,flush=True)


if __name__=="__main__":
    main()
