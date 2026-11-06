"""Existing variants only: no build_variants or biological/connectome changes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,json,hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(2)
from tracking.flyvis_adapter import activate_flyvis
activate_flyvis(device='cpu')
from tracking.flyvis_connectome_swap import make_network,FREE
snapshot=OUT/'loading_before/flyvis_connectome_swap.py'
spec=importlib.util.spec_from_file_location('legacy_swap_snapshot',snapshot)
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
info=json.loads((ROOT/'data/flywire/flyvis_variants/variants_info.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={p:sha(ROOT/p) for p in info['files'].values()}
torch.manual_seed(14);stim=torch.rand(1,3,1,721,device='cpu')
rows=[]
for name,path in info['files'].items():
    old,old_ref,old_view=legacy.make_network(ROOT/path)
    new,new_ref,new_view=make_network(ROOT/path,device='cpu')
    weights=all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    keys={n:list(getattr(new,'node_params' if n.startswith('nodes') else 'edge_params')[n.split('_',1)[1]].keys)==list(getattr(new_ref,'node_params' if n.startswith('nodes') else 'edge_params')[n.split('_',1)[1]].keys) for n in FREE}
    with torch.inference_mode():
        a=old.simulate(stim,1/60,initial_state=None)
        b=new.simulate(stim,1/60,initial_state=None)
    rows.append(dict(name=name,path=path,all_weights_exact=weights,group_order_already_same=keys,
        activity_exact=torch.equal(a,b),max_abs_difference=float((a-b).abs().max())))
    del old,new,old_ref,new_ref,old_view,new_view,a,b
r=dict(variants=rows,source_variants_unchanged={p:sha(ROOT/p)==h for p,h in before.items()},
    source_variant_sha256=before,cuda_initialized=torch.cuda.is_initialized(),
    sha256={p:sha(ROOT/p) for p in ('tracking/flyvis_checkpoint.py','tracking/flyvis_connectome_swap.py','tracking/flyvis_adapter.py')})
r['passed']=all(x['all_weights_exact'] and x['activity_exact'] for x in rows) and all(r['source_variants_unchanged'].values()) and not r['cuda_initialized']
(OUT/'connectome_transfer_native_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(r,indent=2));assert r['passed']
