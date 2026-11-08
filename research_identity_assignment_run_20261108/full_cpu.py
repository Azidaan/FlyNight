"""Sequential CPU child of the externally guarded full-pilot parent."""
import os,sys,argparse
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.full_retina import FULL,prepare_video


class Observer:
    def __init__(self,report):
        self.report=Path(report);assert read(self.report)['returncode'] is None
    def read(self):
        import psutil
        return dict(supervisor_report=str(self.report),limit_active=False,limit_bytes=None,
                    available_bytes=psutil.virtual_memory().available,policy='guarded_parent_process_tree')


def main():
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--parent-report',required=True);p.add_argument('--zero',action='store_true');a=p.parse_args()
    import torch
    torch.set_num_threads(2);obs=Observer(a.parent_report)
    from research_identity_assignment_run_20261108.full_run import rollout
    if a.zero:rollout(a.video,'B0',obs,zero=True)
    else:
        prepare_video(a.video,obs)
        for arm in ['G','G_R1']:rollout(a.video,arm,obs)
    assert not torch.cuda.is_initialized() and not any(n=='flyvis' or n.startswith('flyvis.') for n in sys.modules)


if __name__=='__main__':main()
