"""Read-only check of the 26.10 summary claims: loss weights, actual changes, AUROC pair coverage, observable past tag.

Reads saved JSON only; no torch, no models, no FlyVis, no training.
"""
from pathlib import Path
import collections, json, sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'verify_counts.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    protocol = read(ROOT / 'runs/z13_temporal_identity_20261025/training/protocol.json')
    videos = protocol['fit84'] + protocol['fit20']
    entries = {}
    for v in videos:
        for u, e in read(ROOT / f'runs/z13_box_supervision_followup_20261025/cache/videos/{v}.json')['entries'].items():
            entries[int(u)] = e

    # Weight of each target class in signed_action_loss: mean over actions inside a query, then over known queries.
    weight = collections.Counter(); known = 0; matched = 0
    for e in entries.values():
        matched += bool(e['context'][0][4])
        if e['kind'] < 0:
            continue
        known += 1; n = len(e['choices'])
        for c, g, h in zip(e['choices'], e['good'], e['harm']):
            weight[('null' if c < 0 else 'box') + ('_beneficial' if g else '_harmful' if h else '_neutral')] += 1 / n
    loss_weight = {k: v / known for k, v in sorted(weight.items())}

    # Selected actions split into no-op (equal to the original assignment state) and actual change.
    selected = {}
    for arm in ('D', 'D_blank', 'C', 'R1', 'geometry', 'D_no_identity'):
        c = collections.Counter()
        for path in sorted((ROOT / 'runs/z13_temporal_identity_20261025/training/internal' / arm).glob('seed*/fold*/rows.json')):
            for r in read(path):
                k = r['action']
                if k < 0:
                    continue
                e = entries[r['uid']]
                key = ('null' if e['choices'][k] < 0 else 'box') + ('_noop' if e['context'][k][5] else '_change')
                c[key] += 1
                c[key + '_beneficial'] += r['beneficial']; c[key + '_harmful'] += r['harmful']; c[key + '_unknown_query'] += r['kind'] < 0
        selected[arm] = dict(sorted(c.items()))

    # Known label pairs by query group, and whether an observable past assignment of the original tracker exists.
    pairs = collections.Counter(); past = collections.Counter()
    for v in videos:
        inputs = read(ROOT / f'runs/z13_temporal_pair_readiness_20261025/inputs/{v}.json')['rows']
        labels = read(ROOT / f'runs/z13_temporal_pair_readiness_20261025/supervision/{v}.json')['rows']
        index = {(i['episode_local'], i['current_array_index']): i['uid'] for i in inputs}
        for i, l in zip(inputs, labels):
            if not i['previous']:
                continue
            f = l['flags']
            group = 'unknown_kind' if not f['known'] else 'helpful' if f['oracle_restored'] else 'no_helpful'
            pairs[group] += len(l['correspondence']); pairs[group + '_positive'] += sum(x[2] for x in l['correspondence'])
            if group == 'unknown_kind':
                continue
            past[group] += 1
            past[group + '_gap_le12'] += i['current_frame'] - i['previous']['frame'] <= 12
            if i['previous']['episode_relative_index'] < 4:
                past[group + '_past_in_lead4_not_query'] += 1
                continue
            uid = index.get((i['episode_local'], i['previous']['array_index']))
            pe = entries.get(uid)
            if pe is None or pe['choices'][1:] != i['previous']['choices']:
                past[group + '_past_query_unbound'] += 1
                continue
            assigned = [j for j, row in enumerate(pe['context']) if row[5] and pe['choices'][j] >= 0]
            if not assigned:
                past[group + '_past_query_original_unmatched'] += 1
                continue
            identity = l['previous_identity'][assigned[0] - 1]
            past[group + ('_tag_unknown_identity' if identity is None else '_tag_is_target' if identity == l['target_identity'] else '_tag_other_object')] += 1

    result = dict(videos=len(videos), subset='fit84+fit20 (internal 104, reused in development)',
                  known_queries=known, rows=len(entries), original_matched_track_rows=matched,
                  loss_weight_share=loss_weight, selected_actions_3seeds=selected,
                  known_label_pairs_with_history=dict(pairs), past_observation=dict(sorted(past.items())),
                  definitions=dict(noop='selected action equals original assignment state: context[k][5] from assignment_state_context',
                                   loss_weight_share='signed_action_loss averages over actions inside a query, then over known queries; NULL included',
                                   pairs='per-query known correspondences over all queries with history; per seed equals model-evaluated pairs/3',
                                   lead4='first four episode positions precede queries; whether they are the last tracked frames is not verified here'),
                  sources=['runs/z13_box_supervision_followup_20261025/cache/videos/*.json',
                           'runs/z13_temporal_identity_20261025/training/internal/*/seed*/fold*/rows.json',
                           'runs/z13_temporal_pair_readiness_20261025/inputs/*.json',
                           'runs/z13_temporal_pair_readiness_20261025/supervision/*.json'],
                  caveat='Pooled descriptive sums over queries and three training seeds; not per-video intervals, not HOTA.')
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main())
