from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.full_retina import FULL


def main():
    p=read(FULL/'protocol.json');assert read(FULL/'preflight.json')['passed']
    for n,h in p['source_hashes'].items():assert sha(ROOT/n)==h
    names=list(p['source_hashes'])+['research_identity_assignment_run_20261108/full_technical.py',
        'research_identity_assignment_run_20261108/register_full_checks.py','research_identity_assignment_run_20261108/common.py',
        'tracking/spatial_identity_assignment.py','tracking/foveated_eyes.py','tracking/z13_region_assignment.py',
        'tools/flyvis-runtime/flyvis/datasets/rendering/eye.py','runs/identity_assignment_20261108/full/protocol.json',
        'runs/identity_assignment_20261108/scores_complete.json','runs/identity_assignment_20261108/features_complete.json']
    write(RUN/'full_technical_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources={n:sha(ROOT/n) for n in names}),new=True)


if __name__=='__main__':main()
