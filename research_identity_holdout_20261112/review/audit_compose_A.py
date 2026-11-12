"""Add execution-completeness and target-ID addenda without changing frozen audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import hashlib
from pathlib import Path
import argparse
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix-result', required=True)
    args, private = parser.parse_known_args()
    assert all(x.startswith('--audit-memory-worker=') for x in private)
    go = read(OUT/'matrix_GO_A.json')
    assert go['approved'] is True
    for name, digest in go['inputs'].items():
        assert sha(ROOT/name) == digest, name
    decisions = read(OUT/'decisions_A.json')['records']
    result = read(args.matrix_result)
    assert result['passed'] and len(result['records']) == len(decisions) == 216
    records = {(r['event_key'], r['variant']): r for r in result['records']}
    runtime = read(OUT/'A_runtime_manifest_v2.json')
    readiness = read(OUT/'anchored_readiness.json')
    eval_rows = {(r['event_key'], r['tid']): r for r in readiness['rows']}
    # Reuse only independent exact annotation routing and IoU convention,
    # not any policy or core decision. Labels never enter compose_event.
    from matrix_audit import annotation_objects, iou
    failure_checks, continuity = [], []
    for decision in decisions:
        event = next(e for e in runtime['events'] if e['event_key'] == decision['event_key'])
        measured = records[event['event_key'], decision['variant']]
        base = records[event['event_key'], 'baseline']
        if 'execution_failure' in decision:
            assert not decision['edge_updates'] and not decision['append_columns'] and not decision['defer_tids']
            assert measured['actual_pairs'] == base['actual_pairs'] and not measured['added'] and not measured['removed']
            failure_checks.append(dict(event_key=event['event_key'], variant=decision['variant'], failure=decision['execution_failure'], exact_baseline_preserved=True))
        original = {(event['rows'][i]['tid'], event['high_raw_ids'][j]) for i, j in event['actual_first']}
        assigned = {(int(t), int(r)) for t, r in event['assigned'].items()}
        before = assigned | original
        after = (assigned-original) | {(p['tid'], p['raw_id']) for p in measured['actual_pairs']}
        current = annotation_objects(event)
        raw = {r['raw_id']: r['box'] for r in event['raw_candidates']}
        def hits(rid):
            return [] if current is None or rid not in raw else [o['identity'] for o in current if iou(raw[rid], o['box']) >= .5]
        for proposal in event['eligible_proposal_keys']:
            tid = proposal['old_id']; er = eval_rows[event['event_key'], tid]
            for unit, expected in [('strict_last_reference', er['last_reference_unique_identity'] if er['strict_last_reference_evaluable'] else None), ('anchored_history', er['expected_anchored_persistent_identity'] if er['anchored_history_evaluable'] else None)]:
                def classify(ledger):
                    owned = [dict(raw_id=rid, hits=hits(rid)) for owner, rid in sorted(ledger) if owner == tid]
                    own = [x for x in owned if x['hits'] == [expected]] if expected is not None else []
                    others = [dict(tid=owner, raw_id=rid) for owner, rid in sorted(ledger) if owner != tid and expected is not None and hits(rid) == [expected]]
                    known = expected is not None and current is not None
                    return dict(target_tid_identity_continues=bool(len(owned) == 1 and own) if known else None,
                                target_assignments=owned, other_tid_expected_identity_claims=others,
                                existing_object_coverage_without_target_ID=bool(others and not own) if known else None,
                                duplicate_target_and_other_ID=bool(others and own) if known else None)
                present = bool(expected is not None and current is not None and any(o['identity'] == expected for o in current))
                continuity.append(dict(event_key=event['event_key'], video=event['video'], variant=decision['variant'], target_tid=tid, unit=unit, expected_identity=expected, expected_identity_present=present, baseline=classify(before), after=classify(after)))
    summaries = []
    for summary in result['summaries']:
        variant = summary['variant']; failures = sum(r['variant'] == variant for r in failure_checks)
        identity_counts = {}
        for unit in ['strict_last_reference', 'anchored_history']:
            rr = [r for r in continuity if r['variant'] == variant and r['unit'] == unit]
            known = [r for r in rr if r['expected_identity_present']]
            identity_counts[unit] = dict(all_targets=len(rr), expected_identity_present=len(known),
                                        baseline_target_ID_continues=sum(r['baseline']['target_tid_identity_continues'] is True for r in known),
                                        after_target_ID_continues=sum(r['after']['target_tid_identity_continues'] is True for r in known),
                                        after_object_covered_only_by_other_ID=sum(r['after']['existing_object_coverage_without_target_ID'] is True for r in known),
                                        after_duplicate_target_and_other_ID=sum(r['after']['duplicate_target_and_other_ID'] is True for r in known))
        summaries.append(dict(variant=variant, execution_failures=failures,
                              execution_complete=failures == 0,
                              target_identity_continuity=identity_counts,
                              frozen_object_coverage_gate=summary['full_deployment_gate'],
                              execution_qualified_deployment_gate=bool(summary['full_deployment_gate'] and failures == 0),
                              scientific_identity_benefit_claim=False))
    output = dict(passed=True, failure_checks=failure_checks, summaries=summaries,
                  target_identity_continuity_addendum=continuity,
                  matrix_result_sha256=sha(args.matrix_result),
                  registration='Target-ID continuity addendum requested after recovery A results existed; descriptive, not a replacement preregistered endpoint or selection rule.',
                  scope='Frozen primary benefit metric is physical-object coverage under any tid. It does not prove survival of target public ID. Report both separately. Historical states/other core votes fixed; no rollout/HOTA claim.')
    with (OUT/'compose_audit_A.json').open('x', encoding='utf8', newline='\n') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print('COMPOSE_AUDIT_A', len(failure_checks), len(continuity), flush=True)


if __name__ == '__main__':
    main()
