from pathlib import Path
from copy import deepcopy
import importlib.util,hashlib,json,sys
ROOT=Path.cwd();OUT=ROOT/'runs/flyvis_repairs_20261106/loading_constraint_followup';sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2);torch.set_default_device('cpu')
from tracking.flyvis_adapter import activate_flyvis
from tracking.flyvis_checkpoint import validate_state,load_checkpoint_modules,load_network_and_decoder,read_checkpoint
from tracking.flyvis_connectome_swap import transfer_free_parameters
flyvis=activate_flyvis(device='cpu');assert not torch.cuda.is_initialized()
view=flyvis.NetworkView('flow/0000/000');net,dec=load_network_and_decoder(view);net.eval().requires_grad_(False)
paths=sorted((ROOT/'runs/flyvis/results/flow/0000').glob('*/best_chkpt'))
sourcepaths=['tracking/flyvis_checkpoint.py','tracking/flyvis_connectome_swap.py','tests/test_flyvis_checkpoint_strict.py','tests/test_flyvis_connectome_transfer.py']
def hashes(items):return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in items}
protected=[*paths,*[p.parent/'_meta.yaml' for p in paths],Path(view.get_checkpoint('best'))]
protected_before=hashes(protected);before=deepcopy(net.state_dict());ptrs={k:v.data_ptr() for k,v in net.state_dict().items()}
def verify_same():
 assert all(torch.equal(v,before[k]) and v.data_ptr()==ptrs[k] for k,v in net.state_dict().items())
 for route,registry in [('nodes',net.node_params),('edges',net.edge_params)]:
  for field,param in registry.items():assert param.parameter.raw_values is getattr(net,f'{route}_{field}')
rows=[]
for path in paths:
 payload=read_checkpoint(path);original=deepcopy(payload['network'])
 validate_state(net,payload['network'],'network');verify_same()
 assert all(torch.equal(v,original[k]) for k,v in payload['network'].items())
 rows.append(dict(path=path.relative_to(ROOT).as_posix(),validated=True,input_unchanged=True,live_storage_unchanged=True))
payload=read_checkpoint(view.get_checkpoint('best'))
stimulus=torch.full((1,3,net.n_nodes),.05,device='cpu')
with torch.no_grad():result=net(stimulus,dt=.02).cpu().numpy().copy()
old=np.load(OUT/'outputs.npz');assert np.array_equal(result,old['stock_frozen'])
faults=[]
for key,value in [('edges_syn_strength',-.1),('edges_syn_count',1000.)]:
 candidate=deepcopy(payload);candidate['network'][key][0]=value;original=deepcopy(candidate['network'])
 try:load_checkpoint_modules(net,dec,candidate)
 except ValueError as exc:faults.append(dict(key=key,rejected=True,error=str(exc)))
 else:raise AssertionError('invalid candidate accepted')
 verify_same();assert all(torch.equal(v,original[k]) for k,v in candidate['network'].items())
# Old/new transfer must start from exactly the same constructed destination;
# do not compare its fixed connectome counts against checkpoint raw counts.
spec=importlib.util.spec_from_file_location('old_connectome',OUT/'before/flyvis_connectome_swap.py');oldmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmodule)
destination=view.network_class(**view.dir.config.network.to_dict());destination.eval().requires_grad_(False)
original_destination=deepcopy(destination.state_dict())
oldmodule.transfer_free_parameters(destination,net)
old_state=deepcopy(destination.state_dict())
with torch.no_grad():old_transfer=destination(stimulus,dt=.02).cpu().numpy().copy()
destination.load_state_dict(original_destination);transfer_free_parameters(destination,net)
assert all(torch.equal(v,old_state[k]) for k,v in destination.state_dict().items())
with torch.no_grad():new_transfer=destination(stimulus,dt=.02).cpu().numpy().copy()
assert np.array_equal(old_transfer,new_transfer);verify_same()
# A direct invalid source must be refused without altering either instance.
with torch.no_grad():net.edges_syn_strength[0]=-.1
source_before=deepcopy(net.state_dict());dest_before=deepcopy(destination.state_dict())
try:transfer_free_parameters(destination,net)
except ValueError as exc:transfer_error=str(exc)
else:raise AssertionError('invalid transfer accepted')
assert all(torch.equal(v,source_before[k]) for k,v in net.state_dict().items())
assert all(torch.equal(v,dest_before[k]) for k,v in destination.state_dict().items())
assert protected_before==hashes(protected);assert not torch.cuda.is_initialized()
np.savez_compressed(OUT/'after_outputs.npz',stock=result,old_transfer=old_transfer,new_transfer=new_transfer)
report=dict(device='cpu',cuda_initialized=False,stock_count=len(rows),stock=rows,stock_activity_exact_before_after=True,old_new_transfer_state_exact=True,old_new_transfer_activity_exact=True,rejected=faults,transfer_rejection=transfer_error,source_destination_and_payload_unchanged=True,protected_files_unchanged=True,protected_sha256=protected_before,source_sha256=hashes([ROOT/p for p in sourcepaths]),probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitations='Three synthetic integration steps, no HOTA or training. Independent multiplication of individually finite semantic values can still overflow; no unproved scientific domain constraints added.')
(OUT/'after_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('stock','protected_sha256')},ensure_ascii=False,indent=2))
