"""Freeze executable sources, coverage definitions and endpoint rules before full outputs."""
from research_identity_assignment_run_20261108.common import *
from research_identity_assignment_run_20261108.full_retina import FULL


def main():
    p=read(FULL/'protocol.json');assert read(FULL/'technical_gate.json')['passed']
    for n,h in p['source_hashes'].items():assert sha(ROOT/n)==h,n
    assert p['selection_sha256']==sha(RUN/'selection.json')
    names=['research_identity_assignment_run_20261108/'+n for n in [
        'common.py','full_retina.py','full_planner.py','full_run.py','full_cpu.py','full_master.py',
        'full_evaluate.py','full_statistics.py','coverage_followup.py','register_full_execution.py','evaluator.py']]
    names+=['tracking/spatial_identity_assignment.py','tracking/xywh_bytetrack.py','tracking/offline_association.py',
        'tracking/z13_region_assignment.py','tracking/foveated_eyes.py','tracking/z13_readout.py','tracking/retina.py',
        'tracking/flyvis_adapter.py','tracking/flyvis_motion.py','tracking/flyvis_checkpoint.py','scripts/z13_target_model.py',
        '.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py','.venv/Lib/site-packages/ultralytics/trackers/basetrack.py',
        '.venv/Lib/site-packages/ultralytics/trackers/bot_sort.py','.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py',
        '.venv/Lib/site-packages/ultralytics/trackers/utils/stracks.py','.venv/Lib/site-packages/ultralytics/trackers/utils/kalman_filter.py',
        'runs/z13_local_peak_20261021/geometry.json','runs/z13_target_center_dev_20261018/C/normalization.json',
        'runs/z13_target_center_dev_20261018/R1/normalization.json','runs/flyvis/results/flow/0000/000/best_chkpt',
        'weights/baseline_seed42/best.pt','runs/identity_assignment_20261108/full/protocol.json',
        'runs/identity_assignment_20261108/full/technical_gate.json','evaluation/hota_eval.py',
        '.venv/Lib/site-packages/trackeval/metrics/hota.py','.venv/Lib/site-packages/trackeval/metrics/identity.py',
        'runs/flyvis_repairs_20261106/unification_20261107/system_memory_guard.py']
    write(RUN/'full_execution_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources={n:sha(ROOT/n) for n in names},
        endpoint=p['primary_hota'],inference=p['inference'],
        baseline_policy='One full zero-control parity against cached first val video; reuse all cached B0 trajectories and scalar metrics. Recover only missing official aggregation components, checking equality to cached scalars. Identical-output controls reuse metrics.',
        additional_metrics_definitions=dict(
            old_ID_reactivation='Previously emitted ID returns after >=1 missing output frame; full-video outputs, not merely a first-stage match.',
            prior_identity='Latest known Hungarian IoU>=.5 same-class GT identity at an annotated emitted observation before the gap; unknown remains unknown.',
            wrong_target_takeover='Reactivated ID has known prior identity and a different known identity on the exact annotated return frame. Unannotated or unmatched returns remain unknown.',
            sustained_ID_return='Correct known identity on exact return frame and next2 annotated observations where that target has a GT box. Missing future observations or unknown matches are censored; absent ID at an available target frame is failure.',
            duplicate_ID='At an annotated frame, multiple emitted IDs individually have best same-class GT IoU>=.5 to one target; count extra IDs. Unlike identity labeling, this check intentionally allows many-to-one.'),
        coverage_scope='Descriptive coverage audit only. No current tracker/crop/parameter change; no causal claim from area correlation.'),new=True)
    sources=['research_identity_assignment_run_20261108/coverage_followup.py','research_identity_assignment_run_20261108/evaluator.py',
        'research_identity_assignment_run_20261108/common.py','research_identity_assignment_run_20261108/full_cpu.py',
        'runs/identity_assignment_20261108/export_complete.json','runs/spatial_identity_maps_20261107/bottleneck_rows.json',
        'runs/spatial_identity_maps_20261107/bottleneck_report.json','runs/z13_local_peak_20261021/geometry.json',
        'research_identity_assignment_protocol_20261108/evaluation_strata.json']
    sources += [f'runs/identity_assignment_20261108/outcomes/evaluate26_{a}.json' for a in ['B0','G','G_FLYVIS']]
    write(RUN/'coverage_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources={n:sha(ROOT/n) for n in sources},
        question='Separate historical any-candidate clipping from target clipping, recoverable crop loss, hull-of-centers and direct receptor-filter support.',
        cohort='All1991 frozen queries, primary known and shadow separate. GT only for evaluator target boxes. Reuse all old clipping flags and saved outcomes.',
        numerical_full_fraction_tolerance=1e-8,geometry_audit='100fixed random rectangles; independent OpenCV polygon intersection and brute receptor-support integration.',
        cannot_claim='No larger view, altered core input, or recognition benefit is tested by this geometric audit.'),new=True)


if __name__=='__main__':main()
