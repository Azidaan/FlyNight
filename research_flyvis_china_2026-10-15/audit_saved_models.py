"""CPU-only reading of existing models, losses and predictions; no fitting."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    import torch
    from scipy.spatial import cKDTree
    from scripts.z13_train import Episodes,success
    torch.set_num_threads(2)
    run=ROOT/'runs/z13_candidates_20261016'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    p=read(run/'protocol.json'); result={}
    for arm in 'BCD':
        for mode in ('base','foreground','size','fixed','adaptive'):
            path=run/arm/mode/'checkpoint.pt'
            c=torch.load(path,map_location='cpu',weights_only=False)
            history=c['history']
            result[arm+'_'+mode]=dict(step=c['step'],new_params={k:float(v) for k,v in c['readout'].items() if k in ('background_gain','time_bias','time_slope')},
                mean_loss_first50=float(np.mean([x['loss'] for x in history[:50]])),mean_loss_last50=float(np.mean([x['loss'] for x in history[-50:]])),
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    data=Episodes(p['evaluation'])
    c=torch.load(run/'D/size/checkpoint.pt',map_location='cpu',weights_only=False)
    points=c['readout']['xy'].numpy()
    spacing=float(np.median(cKDTree(points).query(points,k=2)[0][:,1]))
    pred=read(run/'D/base/predictions.json'); sizes=[]; errors=[]; useful=[]
    for r in pred:
        if r['kind']!=1: continue
        l=data.labels[r['label']]
        assert l[0]==r['episode'] and l[1]==r['search_position'] and l[2]==r['kind']
        width=l[5][2:]-l[5][:2]; crop=l[7][2:]-l[7][:2]
        sizes.append(2*width/crop/spacing); errors.append(r['err']/r['scale']); useful.append(success(r))
    sizes=np.asarray(sizes); errors=np.asarray(errors); useful=np.asarray(useful)
    bins=[]; short=sizes.min(1)
    for lo,hi in ((0,2),(2,4),(4,8),(8,float('inf'))):
        take=(short>lo)&(short<=hi)
        bins.append(dict(short_axis_steps_range=[lo,hi if np.isfinite(hi) else None],n=int(take.sum()),
            useful_fraction=float(useful[take].mean()) if take.any() else None,
            median_relative_error=float(np.median(errors[take])) if take.any() else None))
    out=dict(scope='Read-only CPU audit of existing train26 predictions and saved checkpoints. Descriptive association; not a causal resolution experiment or HOTA.',
        models=result,resolution=dict(n=len(sizes),quantiles=[.1,.5,.9],width_steps_quantiles=np.quantile(sizes[:,0],[.1,.5,.9]).tolist(),
        height_steps_quantiles=np.quantile(sizes[:,1],[.1,.5,.9]).tolist(),bins=bins),cuda_initialized=torch.cuda.is_initialized())
    assert not out['cuda_initialized']
    dest=Path(__file__).with_name('saved_model_audit_20261016.json')
    assert not dest.exists()
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(D_foreground=result['D_foreground'],D_adaptive=result['D_adaptive'],resolution=out['resolution'])))

if __name__=='__main__': main()
