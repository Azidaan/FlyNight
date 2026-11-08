"""Sequential bounded transition from recovered features to scores and comparison."""
import sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *


def main():
    while not (RUN/'feature_queue_result.json').exists():
        if datetime.now(MSK)>=datetime(2026,11,9,7,35,tzinfo=MSK):raise TimeoutError('MSK deadline')
        time.sleep(2)
    prior=read(RUN/'feature_queue_result.json')
    if not prior['passed']:
        write(RUN/'comparison_queue_result.json',dict(passed=False,stage='features',prior=prior),new=True);return
    time.sleep(2)
    sources=['research_identity_assignment_run_20261108/common.py','tracking/spatial_identity_assignment.py',
        'research_identity_assignment_run_20261108/queue_compare.py','research_identity_assignment_protocol_20261108/protocol_v2.json',
        'runs/identity_assignment_20261108/export_complete.json','runs/identity_assignment_20261108/features_complete.json',
        'runs/spatial_identity_maps_20261107/image_normalization.json']
    write(RUN/'score_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources={n:sha(ROOT/n) for n in sources+[
        'research_identity_assignment_run_20261108/build_scores.py']}),new=True)
    for stage,script in [('score','build_scores.py'),('comparison','compare.py')]:
        if stage=='comparison':
            names=sources+['runs/identity_assignment_20261108/scores_complete.json',
                'research_identity_assignment_run_20261108/compare.py','research_identity_assignment_run_20261108/evaluator.py',
                'research_identity_assignment_run_20261108/replay.py','research_identity_assignment_run_20261108/simulate.py',
                'scripts/association_log.py','scripts/output_filter_oracle.py','tracking/xywh_bytetrack.py',
                '.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py',
                '.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py',
                'research_identity_assignment_protocol_20261108/queries.json',
                'research_identity_assignment_protocol_20261108/evaluation_strata.json']
            for video in read(RUN/'export_complete.json')['videos']:
                names.extend([f'data/bdd100k/labels/box_track_20/train/{video}.json',
                    f'runs/train_cache/{video}/alignment.json',
                    f'runs/z13_temporal_pair_readiness_20261025/supervision/{video}.json'])
            write(RUN/'comparison_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources={n:sha(ROOT/n) for n in names},
                tie_operational_definition='Fewer changed emitted track-to-raw-detection bindings across all selected fit20 frames; then smaller parameter.',
                recovery_policy='Preserve completed per-video scenario files. A technical recovery must version the failed source and phase; no scientific change.'),new=True)
        with (RUN/(stage+'_launcher.log')).open('x',encoding='utf-8') as out,(RUN/(stage+'_launcher.err')).open('x',encoding='utf-8') as err:
            result=subprocess.run([sys.executable,'-u','research_identity_assignment_run_20261108/'+script],cwd=ROOT,stdout=out,stderr=err)
        if result.returncode!=0:
            write(RUN/'comparison_queue_result.json',dict(passed=False,stage=stage,returncode=result.returncode),new=True);return
        time.sleep(2)
    write(RUN/'comparison_queue_result.json',dict(passed=True,comparison_sha256=sha(RUN/'comparison_report.json')),new=True)


if __name__=='__main__':main()
