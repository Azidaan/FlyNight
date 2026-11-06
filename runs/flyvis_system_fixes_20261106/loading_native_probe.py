"""Native CPU regression: same stock weights and decoded outputs after strict load."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys, json, hashlib, importlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(2)
from tracking.flyvis_adapter import activate_flyvis
from tracking.flyvis_checkpoint import load_network_and_decoder
before_cuda=torch.cuda.is_initialized()
fv=activate_flyvis(device='cpu')
view=fv.NetworkView('flow/0000/000')
old_net=view.init_network();old_dec=view.init_decoder()['flow']
new_net,new_decs=load_network_and_decoder(view);new_dec=new_decs['flow']
weights_exact={name:all(torch.equal(v,second.state_dict()[k]) for k,v in first.state_dict().items())
    for name,first,second in [('network',old_net,new_net),('decoder',old_dec,new_dec)]}
torch.manual_seed(113)
stimulus=torch.rand(1,3,1,721,device='cpu')
with torch.inference_mode():
    old_net.eval().requires_grad_(False);new_net.eval().requires_grad_(False)
    old_dec.eval().requires_grad_(False);new_dec.eval().requires_grad_(False)
    old_activity=old_net.simulate(stimulus,1/60,initial_state=None)
    new_activity=new_net.simulate(stimulus,1/60,initial_state=None)
    old_output=old_dec(old_activity);new_output=new_dec(new_activity)
original_device=torch.get_default_device()
activate_flyvis()
report=dict(stock_checkpoint=str(view.get_checkpoint('best')),weights_exact=weights_exact,
    activity_exact=torch.equal(old_activity,new_activity),decoder_exact=torch.equal(old_output,new_output),
    max_abs_output_difference=float((old_output-new_output).abs().max()),
    cpu_devices=sorted({str(p.device) for m in (old_net,new_net,old_dec,new_dec) for p in m.parameters()}),
    device_aliases={m:str(importlib.import_module(m).device) for m in ('flyvis','flyvis.network.initialization','flyvis.task.decoder')},
    repeated_default_device_unchanged=torch.get_default_device()==original_device,
    cuda_initialized_before=before_cuda,cuda_initialized_after=torch.cuda.is_initialized(),
    sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('tracking/flyvis_adapter.py','tracking/flyvis_checkpoint.py')})
report['passed']=all(weights_exact.values()) and report['activity_exact'] and report['decoder_exact'] and report['cpu_devices']==['cpu'] and not report['cuda_initialized_after'] and report['repeated_default_device_unchanged']
out=Path(__file__).with_name('loading_native_report.json')
out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2));assert report['passed']
