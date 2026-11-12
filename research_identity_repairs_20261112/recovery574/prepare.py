import ast, json, hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

sources = ['appearance_recovery_policy.py', 'worker.py', 'audit.py', 'prepare.py']
for name in sources:
    path = OUT / name
    source = path.read_text(encoding='utf8').replace('\r\n', '\n').replace('\r', '\n')
    source = '\n'.join(line.rstrip() for line in source.splitlines()).rstrip() + '\n'
    ast.parse(source)
    path.write_bytes(source.encode('utf8'))
snapshot_path = 'research_identity_repairs_20261112/review/assignment_snapshots.json'
plan_path = 'research_identity_repairs_20261112/review/reference_extraction_plan.json'
lowdir = 'research_remaining_causes_20261112/independent_review/reid/'
inputs = [snapshot_path, plan_path] + [lowdir + name for name in ['low_descriptors.npz', 'low_protocol_v3.json', 'low_result.json', 'descriptors.npz', 'result.json']]
inputs += ['scripts/evaluate_botsort_video_rate.py', '.venv/Lib/site-packages/ultralytics/trackers/bot_sort.py', 'tracking/detection_cache.py', 'scripts/run_bdd_video_rate_experiment.py', 'runs/domain_matched_native_20261109/protocol.json']
snapshots = read(ROOT / snapshot_path)['snapshots']
for video in sorted({s['video'] for s in snapshots}):
    inputs += [f'data/bdd100k/labels/box_track_20/val/{video}.json', f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{video}.json']
protocol = dict(created=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(), experiment_id='identity_repair_excluded_low_recovery_20261112',
                question='Can an optional appearance-confirmation branch recover excluded low-confidence detections using a complete competitive row pool without changing native thresholds or assigned active identities?',
                problem='The true current continuation for574 exists among cached detections but both observations have confidence below native low threshold. Recovering a pair cannot be called a public-ID fix: the old falseproposal was already globallyblocked.',
                engineering_scope='Five previously examined cases and222 raw current-car detections; selected with known earlier outcomes. This is an engineering check, not heldout generalization.',
                inputs={name: sha(ROOT / name) for name in sorted(set(inputs))}, sources={name: sha(OUT / name) for name in sources},
                pending_descriptor_bank='research_identity_repairs_20261112/review/reference_bank.npz',
                bank_binding='Parent must register descriptor_binding.json with exact bank SHA256 and extraction execution/protocol hashes BEFORE GO and repair outcomes. Bank keys match extraction-plan video|frame; each array is boxes-by256. All5 original target references reused exactly; all107 row references available.',
                decision_inputs={'rows': 'Only tid,state,cls,predicted_xyxy from all107 saved first-stage rows.', 'candidates': 'Only raw_id,box,confidence,cls for all222 raw class2 detections; no GT labels, old_case or special IDs.', 'memory': 'MASA descriptors of actual prior observed reference boxes. Reference rawframe strictly before currentframe. All competing rows included.', 'ownership': 'Freeze ALL snapshot.assigned public (tid,raw_id), including later native stages/new tracks, not just first Hungarian pairs.'},
                policy={'candidate_aliases': 'Confidence-priority class-specific greedy NMS; suppress only IoU>0.7, representative highest confidence then raw_id tie ordering, preserve every alias. NMS is truth-free and not an identity grouping.',
                        'appearance': 'Borrow unchanged project BoT-SORT appearance_thresh=0.8 with exact source semantics cost=(1-cosine)/2 <=1-0.8, equivalent cosine about0.6. This is explicitly a borrowed hypothesis, not a calibrated MASA probability.',
                        'proximity': 'Borrow unchanged project BoT-SORT proximity_thresh=0.5: representative box IoU with actual saved predicted_xyxy >=0.5.',
                        'low_only': 'Only representative confidence<=native track_low_thresh0.1 may enter optional recovery. Native high0.25/low0.1 unchanged; no new tracks started.',
                        'competition': 'Require unique forward best appearance group among ALL raw groups and unique reverse best row among ALL complete reference rows, same class. No new margin threshold. Exact score tie=>defer.',
                        'continue': 'Native-unassigned tracked/lost row; best group excluded low, passes both absolute gates, strictly mutual and no alias has a frozen native owner. Emit separate proposal, never mutate tracker.',
                        'defer': 'Preserve native assignment, missing reference competition/descriptors, tied or nonmutual best, occupied alias or best group belongs to native candidate branch.',
                        'refuse': 'No raw candidate, or no excluded-low candidate passes fixed absolute appearance and proximity gates. Top1 by itself is never an absence detector.',
                        'insertion_cost': 'appearance_cost=(1-cosine)/2; parent adapter may solve unmatched rows vs newlow columns with0.2 sourcegate, freezing saved assignments. Proposal is not public output or lifecycle replay.'},
                fixed_gate_sources={'appearance_and_proximity_values': 'scripts/evaluate_botsort_video_rate.py botsort_args: appearance_thresh0.8/proximity_thresh0.5; with_reid=False there, so transfer to MASA is a tested hypothesis.',
                                    'appearance_semantics': '.venv/Lib/site-packages/ultralytics/trackers/bot_sort.py BOTSORT.get_dists divides embedding_distance by2 and gates at1-appearance_thresh.',
                                    'NMS': 'tracking/detection_cache.py CACHE_IOU=0.7 and scripts/run_bdd_video_rate_experiment.py --iou default0.7; repeated grouping preserves aliases and cannot redefine GT identity.',
                                    'native_thresholds': 'runs/domain_matched_native_20261109/protocol.json and frozen low_protocol_v3.json.'},
                controls={'disabled': 'No proposed changes; original native assignments preserved.', 'complete_recovery': 'Only registered rule above.', 'descriptor_permutation': 'Cyclic shift current descriptors by exactlyone slot within each complete frame, fixed before outcomes; boxes/confidence/rowbank unchanged.', 'missing_competitors': 'competition_complete=False mustdefer all rows.', 'no_descriptors': 'No vectors mustdefer all rows.', 'remove_all_target_own': 'Evaluation-only generator removes EVERY candidate whose existing annotation hits include target referenceGT identity; policy receives reduced unlabeled boxes/descriptors, never targetGT. Recompute NMS then require no false continuation for target.'},
                acceptance=['All107 reference rows and222 currentraw candidates retained before grouping; original5 target descriptor parity<1e-6 and all222 original cosine witnesses<1e-6. No altered threshold/fit or hidden reference selection.',
                            'Independent scalar cosine, IoU, greedy NMS and entire decision reconstruction agree within1e-9; baseline assignments exactly preserved and every proposed alias free.',
                            'Primary engineering gate:574 target getscontinue to a uniquely-own raw detection; no foreign or unknown new proposal anywhere among the5 complete saved pools. Report fail/defer honestly.',
                            'Original active assignments for832 and1102 preserved exactly; do not steal occupied raw/alias groups or displace baseline IDs. Their original C34 issues are separate.',
                            'On remove-all-own control, target mustrefuse/defer for all5cases; do not accept a wrong currentcandidate just because it ranksfirst. Missing/no-descriptor controls all defer.',
                            'Report descriptor-permutation results without suppressing accidental successes; no new tuning after outcomes. A failed rule remains failed.',
                            'No GT/old_id/known raw-id branching in puremodule; public-ID lifecycle and complete tracker metrics remain unmeasured.'],
                intended_candidate='appearance_recovery_policy.py is actual reusable pure candidate code, intended tracking/appearance_recovery_policy.py only after parent checks. It is not connected to full runtime.',
                limits=['Borrowed BoT-SORT gates have no existing MASA calibration guarantee.', 'Same5 selected scenes, no unseen evaluation.', 'Frozen native assignments may already include ID fragmentation; proposed pair recovery is not public ID correction.', 'NMS duplicate aliases use box overlap, never GT identity; near duplicate observations below0.7 remain competitors.', 'No new full tracker/TrackEval/core pass, no production activation before engineering gate.'],
                cpu=dict(CUDA_VISIBLE_DEVICES='', threads=2, policy_neural_calls=0, new_fits=0, core_calls=0, tracker_calls=0), safe_cutoff='2026-11-13T07:35:00+03:00', maximum_seconds=120)
with (OUT / 'proposal_protocol.json').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(json.dumps(protocol, ensure_ascii=False, indent=2).rstrip() + '\n')
print('RECOVERY_PROTOCOL_READY', sha(OUT / 'proposal_protocol.json'), len(protocol['inputs']))
