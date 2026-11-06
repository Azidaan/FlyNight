"""Read-only CPU audit; original code/checkpoints remain untouched."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import sys, json, hashlib, traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
torch.set_num_threads(2)
from tracking.flyvis_adapter import activate_flyvis
fv = activate_flyvis()
torch.set_default_device('cpu')
from tracking.flyvis_motion import FlyVisMotionEstimator
from tracking.flyvis_activity import OUTPUT_CELL_TYPES, type_column_indices
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
report = {'scope':'CPU engineering audit; no training, no HOTA, no repairs', 'checks':{}}
def check(name, fn):
    try: report['checks'][name] = fn()
    except Exception: report['checks'][name] = {'exception':traceback.format_exc()}
    (OUT/'core_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(name, json.dumps(report['checks'][name],ensure_ascii=False),flush=True)
model = FlyVisMotionEstimator(source_fps=30, flow_gain=1, fade_in_seconds=.1)
net = model.network
view = fv.NetworkView('flow/0000/000')
cp = Path(view.get_checkpoint('best'))
ck = torch.load(cp,map_location='cpu',weights_only=False)
report['checkpoint'] = {'path':str(cp),'sha256':sha(cp),'keys':list(ck)}
before = {k:v.detach().clone() for k,v in net.state_dict().items()}
def weights():
    result = {}
    for name,mod,stored in [('network',net,ck['network']),('decoder',model.decoder,ck['decoder']['flow'])]:
        state=mod.state_dict()
        result[name]={'same_keys':set(state)==set(stored),'exact':all(torch.equal(v,stored[k]) for k,v in state.items()),'finite':all(bool(torch.isfinite(v).all()) for v in state.values())}
    result['parameters']={k:{'shape':list(v.shape),'min':float(v.min()),'max':float(v.max()),'negative':int((v<0).sum())} for k,v in net.named_parameters()}
    return result
check('loaded_weights',weights)
def connections():
    c=net.connectome
    src=np.asarray(c.edges.source_index[:]); tar=np.asarray(c.edges.target_index[:])
    types=np.asarray(c.nodes.type[:]).astype(str); u=np.asarray(c.nodes.u[:]);v=np.asarray(c.nodes.v[:])
    ix=type_column_indices(types,OUTPUT_CELL_TYPES); r=np.asarray(net.stimulus.input_index)
    out={'nodes':net.n_nodes,'edges':net.n_edges,'source_indices_match':np.array_equal(src,net._source_indices.numpy()),'target_indices_match':np.array_equal(tar,net._target_indices.numpy()),'index_range_valid':bool(min(src.min(),tar.min())>=0 and max(src.max(),tar.max())<net.n_nodes),'input_shape':list(r.shape),'input_unique':len(np.unique(r))==r.size,'output_columns':{k:len(x) for k,x in ix.items()},'coordinates_match_receptors':all(np.array_equal(u[x],u[r[0]]) and np.array_equal(v[x],v[r[0]]) for x in ix.values())}
    p=net._param_api(); w=p.edges.weight
    out['effective_weights_finite']=bool(torch.isfinite(w).all())
    out['weight_product_exact']=bool(torch.equal(w,p.edges.sign*p.edges.syn_count*p.edges.syn_strength))
    # Independent numpy accumulation of the actual edge list, not target_sum.
    rng=np.random.default_rng(3); a=rng.normal(size=(1,net.n_nodes)).astype(np.float32)
    edge=w.detach().numpy()*np.maximum(a[:,src],0)
    expected=np.bincount(tar,weights=edge[0],minlength=net.n_nodes)
    actual=net.target_sum(torch.from_numpy(edge)).numpy()[0]
    out['target_sum_max_abs_error']=float(np.max(np.abs(expected-actual)))
    out['target_sum_close_1e_5']=bool(np.allclose(expected,actual,atol=1e-5,rtol=1e-5))
    return out
check('connections',connections)
def sequence():
    rng=np.random.default_rng(4); frames=[rng.integers(0,256,(96,128),dtype=np.uint8) for _ in range(3)]
    def run():
        model.reset();return [model.observe(x).vectors.copy() for x in frames]
    first=run();second=run()
    return {'reset_replay_exact':all(np.array_equal(a,b) for a,b in zip(first,second)),'finite':all(np.isfinite(a).all() for a in first),'weights_unchanged':all(torch.equal(v,before[k]) for k,v in net.state_dict().items())}
check('sequence_reset',sequence)
def eyetest():
    from tracking.foveated_eyes import EyeBank
    bank=EyeBank(model,fade_in_seconds=.1)
    gray=np.full((96,128),127,np.uint8)
    bank.birth(gray,{1:(50.,50.,20.,20.),2:(80.,50.,20.,20.)})
    result=bank.step(gray,{1:(50.,50.,20.,20.),2:(80.,50.,20.,20.)})
    return {'shapes':{k:list(v.shape) for k,v in result.items()},'same':np.array_equal(result[1],result[2])}
check('native_eye_bank',eyetest)
def snapshots():
    answer=[]
    for p in sorted((ROOT/'runs/b123_20261027/b1_training').glob('*/checkpoint.pt')):
        z=torch.load(p,map_location='cpu',weights_only=False);state=z.get('backbone',{})
        answer.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'arm':z.get('arm'),'finite':all(bool(torch.isfinite(v).all()) for v in state.values()),'core_keys_match':set(state)==set(before),'core_equal_stock':all(k in before and torch.equal(v,before[k]) for k,v in state.items()),'negative_strengths':int((state['edges_syn_strength']<0).sum()) if 'edges_syn_strength' in state else None})
    return answer
check('b1_checkpoints',snapshots)
report['cuda_initialized']=torch.cuda.is_initialized()
report['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'tracking/flyvis_motion.py',ROOT/'tracking/foveated_eyes.py',ROOT/'tracking/z13_readout.py',ROOT/'tracking/flyvis_adapter.py']}
(OUT/'core_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
