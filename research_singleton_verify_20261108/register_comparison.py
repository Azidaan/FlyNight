"""Freeze the new gate and tests, then attach the ready comparison stage."""
import sys,time,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_20261108.common import *


def main():
    import psutil
    receipt=read(RUN/'queue_started.json');process=psutil.Process(receipt['pid']);assert abs(process.create_time()-receipt['created'])<.01
    while not (RUN/'data_complete.json').exists():
        if not process.is_running():raise RuntimeError('The data queue ended before completion; preserve and recover the exact error.')
        if datetime.now(MSK)>=datetime(2026,11,8,13,55,tzinfo=MSK):return
        time.sleep(10)
    sources=read(RUN/'input_phase.json')['sources'].copy();sources.update(read(RUN/'feature_phase.json')['sources'])
    for name in ['tracking/own_template_verification.py','tracking/free_singleton_assignment.py','tracking/spatial_identity_assignment.py',
        'tests/test_own_template_verification.py','research_singleton_verify_20261108/score.py','research_singleton_verify_20261108/simulate.py',
        'research_singleton_verify_20261108/compare.py','research_singleton_verify_20261108/register_comparison.py',
        'research_singleton_free_20261108/simulate.py','research_identity_assignment_run_20261108/compare_recovery1.py',
        'research_identity_assignment_run_20261108/evaluator.py','research_identity_assignment_run_20261108/replay.py',
        'runs/singleton_verify_20261108/features_complete.json','runs/singleton_verify_20261108/data_complete.json',
        'runs/singleton_free_20261108/comparison_report.json','runs/singleton_free_20261108/independent_audit.json',
        'runs/singleton_free_20261108/technical.json','runs/spatial_identity_maps_20261107/image_normalization.json',
        'research_identity_assignment_protocol_20261108/queries.json','research_identity_assignment_protocol_20261108/evaluation_strata.json']:
        if name.endswith('.py'):ast.parse((ROOT/name).read_text(encoding='utf-8-sig'))
        sources[name]=sha(ROOT/name)
    write(RUN/'comparison_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,
        clarification='A finite absolute score passes inclusively at score>=threshold;off and undefined always abstain. Same rule for every matched representation and hash control.'),new=True)
    write(RUN/'comparison_registration.json',dict(passed=True,phase_sha256=sha(RUN/'comparison_phase.json')),new=True)


if __name__=='__main__':main()
