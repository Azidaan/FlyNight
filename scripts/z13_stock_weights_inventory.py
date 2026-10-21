"""Supplementary checkpoint provenance audit; no model execution or GPU use."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import sha,write,now


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_default_device('cpu')
    stock=ROOT/'runs/flyvis/results/flow/0000/000/best_chkpt'
    archived=ROOT/'runs/z13_spatial_v2_20261013/C/full/model.pt'
    original=torch.load(stock,map_location='cpu',weights_only=False)
    saved=torch.load(archived,map_location='cpu',weights_only=False)
    a=original['network'];b=saved['backbone'];keys=['nodes_bias','nodes_time_const','edges_syn_strength']
    same={k:bool(torch.equal(a[k],b[k])) for k in keys}
    def info(path):
        st=path.stat()
        return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=st.st_size,modified_ns=st.st_mtime_ns)
    report=dict(recorded=now().isoformat(),current_stock=info(stock),archived_C=info(archived),free_parameter_groups_equal=same,
        free_values=sum(a[k].numel() for k in keys),all_common_tensor_states_equal=all(torch.equal(a[k],b[k]) for k in set(a)&set(b) if torch.is_tensor(a[k]) and torch.is_tensor(b[k])),
        limitation='Supplementary audit after launch, not falsely retroactively added to a prelaunch hash manifest. Frozen single-readout reproduction is checked separately on every evaluation row.')
    assert all(same.values());write(ROOT/'runs/z13_query_memory_20261021/stock_weights_inventory.json',report)
    print({k:report[k] for k in ('free_parameter_groups_equal','free_values','all_common_tensor_states_equal')})


if __name__=='__main__':main()
