"""Check every saved action against original frozen effect labels before recording."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_activity_amplitude_20261025 as base
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now


def main():
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    p=read(base.RUN/'protocol.json');total=0
    loader=base.context.original.namespace()['load_split']
    for representation in ('D','C','R1','geometry'):
        data=loader(torch,'D' if representation=='geometry' else representation,'train26')
        uid={int(u):i for i,u in enumerate(data['uid'])}
        assert len(uid)==8326
        for mode in ('norm','affine'):
            for seed in p['seeds']:
                rows=read(base.RUN/'training'/(representation+'_'+mode)/f'seed{seed}'/'rows.json')
                assert set(x['uid'] for x in rows)==set(uid)
                for row in rows:
                    i=uid[row['uid']];action=row['action'];known=row['kind']>=0
                    assert row['kind']==int(data['kind'][i])
                    assert action<0 or bool(data['mask'][i,action])
                    assert row['beneficial']==bool(action>=0 and known and data['good'][i,action])
                    assert row['harmful']==bool(action>=0 and known and data['harm'][i,action])
                    total+=1
    assert total==24*8326 and not torch.cuda.is_initialized()
    write(base.RUN/'action_label_audit.json',dict(at=now().isoformat(),model_rows_checked=total,original_unchanged_labels_verified=True,report_sha256=sha(base.RUN/'training/report.json')))
    base.finish()


if __name__=='__main__':main()
