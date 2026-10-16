"""CPU analysis of complete predeclared comparisons, never HOTA."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/z13_candidates_20261016'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    from tracking.z13_checkpoint import atomic_json
    from scripts.z13_train import success
    assert not (OUT/'report.json').exists(), 'Do not overwrite results'
    p=read(OUT/'protocol.json'); arrays={}; stats={}
    expected=None
    for job in p['jobs']:
        a,m=job['arm'],job['mode']; folder=OUT/a/m
        done=read(folder/'completed.json')
        assert done['step']==p['steps'] and done['backbone_unchanged']
        assert sha(folder/'predictions.json')==done['prediction_sha256']
        rows=read(folder/'predictions.json')
        keys=[(r['video'],r['label'],r['episode'],r['search_position'],r['kind']) for r in rows]
        if expected is None: expected=keys
        else: assert expected==keys
        pos=[r for r in rows if r['kind']==1]; videos=sorted({r['video'] for r in pos})
        means=lambda field:np.array([np.mean([field(r) for r in pos if r['video']==v]) for v in videos])
        success_video=means(lambda r:float(success(r)))
        iou=means(lambda r:r['iou_new_size']); iou_delta=means(lambda r:r['iou_new_size']-r['iou_last_size'])
        scores=np.array([r['p_present'] for r in rows]); n=max(1,int(np.ceil(len(rows)*.01)))
        tau=float(np.sort(scores)[-n]); selected=[r for r in rows if r['p_present']>=tau]
        known=[r for r in selected if r['kind']>=0]
        bad= sum(not success(r) for r in known)/len(known) if known else None
        name=a+'_'+m
        stats[name]=dict(rows=len(rows),positive=len(pos),useful_fraction_pooled=float(np.mean([success(r) for r in pos])),
            useful_fraction_mean_video=float(success_video.mean()),mean_video_iou=float(iou.mean()),
            mean_video_size_iou_delta=float(iou_delta.mean()),top1_tau=tau,selected=len(selected),selected_known=len(known),
            selected_unknown=len(selected)-len(known),selected_known_bad_fraction=bad)
        arrays[name]=dict(success=success_video,iou=iou,size_delta=iou_delta,videos=videos)
    rng=np.random.default_rng(20261016); comparisons={}
    def interval(values):
        sample=rng.integers(len(values),size=(2000,len(values)))
        return dict(mean=float(values.mean()),ci95=np.quantile(values[sample].mean(1),[.025,.975]).tolist())
    pairs=[('D_foreground','D_base','success'),('D_foreground','B_foreground','success'),('D_foreground','C_foreground','success'),
           ('D_adaptive','D_fixed','success'),('D_adaptive','B_adaptive','success'),('D_adaptive','C_adaptive','success'),
           ('D_size','B_size','iou'),('D_size','C_size','iou')]
    for a,b,metric in pairs:
        assert arrays[a]['videos']==arrays[b]['videos']
        comparisons[a+'-'+b]=interval(arrays[a][metric]-arrays[b][metric])
    comparisons['D_size_delta']=interval(arrays['D_size']['size_delta'])
    def positive(key): return comparisons[key]['ci95'][0]>0
    def nonworse_bad(a,b):
        x,y=stats[a]['selected_known_bad_fraction'],stats[b]['selected_known_bad_fraction']
        return x is not None and y is not None and x<=y
    gates={
      'foreground':[positive('D_foreground-D_base'),positive('D_foreground-B_foreground'),nonworse_bad('D_foreground','D_base')],
      'size':[positive('D_size_delta'),positive('D_size-B_size'),read(OUT/'D/size/size_center_check.json')['passed']],
      'adaptive':[positive('D_adaptive-D_fixed'),positive('D_adaptive-B_adaptive'),nonworse_bad('D_adaptive','D_fixed')]}
    report=dict(scope=p['scope'],hota_measured=False,stats=stats,comparisons=comparisons,criteria=gates,
                passed={k:all(v) for k,v in gates.items()},limitations=p['limitations'],confidence=p['confidence'],
                protocol_sha256=sha(OUT/'protocol.json'))
    atomic_json(OUT/'report.json',report)
    print(json.dumps(report['passed']))

if __name__=='__main__': main()
