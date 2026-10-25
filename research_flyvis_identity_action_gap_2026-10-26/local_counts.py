"""Read-only counts from saved labels for the identity-to-action gap review; no torch, no FlyVis, no training."""
from pathlib import Path
import collections, json, sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'local_counts.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    protocol = read(ROOT / 'runs/z13_temporal_identity_20261025/training/protocol.json')
    videos = protocol['fit84'] + protocol['fit20']
    actions = collections.Counter()
    for v in videos:
        for e in read(ROOT / f'runs/z13_box_supervision_followup_20261025/cache/videos/{v}.json')['entries'].values():
            actions['rows'] += 1
            if e['kind'] < 0:
                actions['unknown_queries'] += 1
                continue
            actions['known_queries'] += 1
            actions[f"kind{e['kind']}"] += 1
            boxes = [(g, h) for j, g, h in zip(e['choices'], e['good'], e['harm']) if j >= 0]
            actions['box_actions'] += len(boxes)
            actions['box_beneficial_actions'] += sum(g for g, _ in boxes)
            actions['box_harmful_actions'] += sum(h for _, h in boxes)
            actions['null_beneficial'] += bool(e['good'][0])
            actions['null_harmful'] += bool(e['harm'][0])
            actions['queries_with_beneficial_box'] += any(g for g, _ in boxes)
            actions['queries_without_box'] += not boxes
    history = collections.Counter()
    for v in videos:
        inputs = read(ROOT / f'runs/z13_temporal_pair_readiness_20261025/inputs/{v}.json')['rows']
        labels = read(ROOT / f'runs/z13_temporal_pair_readiness_20261025/supervision/{v}.json')['rows']
        for i, l in zip(inputs, labels):
            f = l['flags']
            if not (f['known'] and f['restored_supported']):
                continue
            group = 'helpful' if f['oracle_restored'] else 'no_helpful_box'
            history[group] += 1
            history[group + '_two_or_more_current_regions'] += len(i['current_choices']) >= 2
            history[group + '_two_or_more_past_regions'] += len(i['previous']['choices']) >= 2
            history[group + '_target_identity_in_past_regions'] += bool(l['target_identity'] and l['target_identity'] in l['previous_identity'])
            history[group + '_target_linked'] += bool(f['oracle_target_linked'])
            history[group + '_known_pairs'] += bool(f['labeled_pairs'])
            history[group + '_positive_pairs'] += sum(x[2] for x in l['correspondence'])
            history[group + '_negative_pairs'] += sum(not x[2] for x in l['correspondence'])
    selected = {}
    for arm in ('D', 'D_blank', 'C', 'R1', 'geometry', 'D_no_identity'):
        c = collections.Counter()
        for path in sorted((ROOT / 'runs/z13_temporal_identity_20261025/training/internal' / arm).glob('seed*/fold*/rows.json')):
            for r in read(path):
                if r['action'] < 0:
                    continue
                k = 'null' if r['action'] == 0 else 'box'
                c[k] += 1
                c[k + '_beneficial'] += r['beneficial']
                c[k + '_harmful'] += r['harmful']
                c[k + '_unknown_query'] += r['kind'] < 0
        selected[arm] = dict(c)
    result = dict(videos=len(videos), subset='fit84+fit20 (internal 104, reused in development)',
                  selected_actions_3seeds=selected,
                  sources=['runs/z13_box_supervision_followup_20261025/cache/videos/*.json',
                           'runs/z13_temporal_pair_readiness_20261025/inputs/*.json',
                           'runs/z13_temporal_pair_readiness_20261025/supervision/*.json'],
                  action_labels=dict(actions), known_with_history=dict(history),
                  caveat='Pooled descriptive counts over queries, not per-video intervals. Pair counts here are label pairs per query, not the 182421 model-evaluated pairs of independent_audit.json (validation folds x 3 seeds).')
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main())
