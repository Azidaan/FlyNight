"""Audit saved spatial-readout results without rerunning any model."""
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now
from scripts.z13_hota_deadline_guard import WindowsProcesses
from scripts.z13_localization_labels_20261023 import verify_sources
RUN = ROOT / 'runs/z13_spatial_set_readout_20261025/recovery2'
OUT = RUN / 'training/result_audit'


def main():
    assert not (OUT / 'audit.json').exists()
    p = read(RUN / 'training/recovery4_gpu/protocol.json')
    verify_sources(p)
    end = read(RUN / 'training/recovery4_gpu/controller_result.json')
    assert end['exit_code'] == 1 and end['worker_ended'] and end['all_owned_ended'] and not end['remaining']
    ps = WindowsProcesses()
    assert all(ps.creation(int(pid)) != birth for pid, birth in end['owned'].items())
    from scripts.z13_spatial_set_recovery4_gpu_20261025 import source
    lines = source().splitlines()
    assert 'assert not torch.cuda.is_initialized()' in lines[337], lines[337]
    internal = read(RUN / 'training/internal_report.json')
    assert internal['complete'] and internal['models'] == 75
    assert not (RUN / 'training/final_report.json').exists()
    assert not list((RUN / 'training/final').glob('*/seed*/done.json'))
    follow = ROOT / 'runs/z13_box_supervision_followup_20261025/cache'
    manifest = read(follow / 'report.json')['manifest']
    videos = p['fit84'] + p['fit20']
    entries, video_of = {}, {}
    for v in videos:
        path = follow / 'videos' / (v + '.json')
        assert sha(path) == manifest[v]
        for u, row in read(path)['entries'].items():
            u = int(u)
            assert u not in entries
            entries[u], video_of[u] = row, v
    aggregate, hashes, counts = {}, {}, {}
    for arm in p['arms']:
        counts[arm] = dict(selected=0, beneficial=0, harmful=0, unknown_selected=0)
        for seed in p['seeds']:
            per, seen = {}, set()
            for f, split in enumerate(p['folds']):
                folder = RUN / 'training/internal' / arm / f'seed{seed}' / f'fold{f}'
                done, rec = read(folder / 'done.json'), read(folder / 'report.json')
                assert done['steps'] == p['steps'] and done['predictions_reproduced_exact']
                for filename, digest in [('model.pt', done['model_sha256']), ('report.json', done['report_sha256']), ('rows.json', rec['rows_sha256'])]:
                    assert sha(folder / filename) == digest
                    hashes[(folder / filename).relative_to(ROOT).as_posix()] = digest
                expected = {u for u, v in video_of.items() if v in split['validation']}
                rows = read(folder / 'rows.json')
                assert len(rows) == len(expected) and {x['uid'] for x in rows} == expected and not seen & expected
                seen |= expected
                numerator = {v: 0 for v in split['validation']}
                denominator = dict(numerator)
                cnt = dict(selected=0, beneficial=0, harmful=0, unknown_selected=0)
                for x in rows:
                    en, a = entries[x['uid']], x['action']
                    known = x['kind'] >= 0
                    assert x['video'] == video_of[x['uid']] and x['kind'] == en['kind'] and -1 <= a < len(en['choices'])
                    good = bool(known and a >= 0 and en['good'][a])
                    harm = bool(known and a >= 0 and en['harm'][a])
                    assert good == x['beneficial'] and harm == x['harmful']
                    cnt['selected'] += a >= 0
                    cnt['beneficial'] += good
                    cnt['harmful'] += harm
                    cnt['unknown_selected'] += a >= 0 and not known
                    numerator[x['video']] += int(good) - int(harm)
                    denominator[x['video']] += known
                actual = {v: numerator[v] / denominator[v] for v in numerator}
                assert actual == rec['per_video'] and cnt == rec['counts']
                per.update(actual)
                for k, value in cnt.items(): counts[arm][k] += value
                grid = rec['calibration']
                assert len(grid) == 11
                for g in grid:
                    assert set(g['per_video']) == set(split['calibration'])
                    assert g['mean'] == float(np.mean(list(g['per_video'].values())))
                best = max(range(len(grid)), key=lambda j: (grid[j]['mean'], -grid[j]['counts']['selected'], j))
                assert rec['threshold'] == grid[best]['threshold']
            assert seen == set(entries) and set(per) == set(videos)
            aggregate[arm, seed] = per
    means = {a: {v: float(np.mean([aggregate[a, s][v] for s in p['seeds']])) for v in videos} for a in p['arms']}
    assert means == internal['means']
    paired = {}
    for key, control in [('D-baseline', {v: 0. for v in videos})] + [('D-' + a, means[a]) for a in p['arms'][1:]]:
        d = np.array([means['D_spatial'][v] - control[v] for v in sorted(videos)])
        samples = d[np.random.default_rng(1).integers(0, len(d), (5000, len(d)))].mean(1)
        paired[key] = dict(mean_per_video=float(d.mean()), ci95=[float(np.quantile(samples, .025)), float(np.quantile(samples, .975))], better=int((d > 0).sum()), n=len(d))
    assert paired == internal['paired']
    passed = paired['D-baseline']['mean_per_video'] >= .02 and all(x['ci95'][0] > 0 for x in paired.values())
    assert passed == internal['admit_final'] == False
    write(OUT / 'audit.json', dict(at=now().isoformat(), complete=True, models=75, primary_rows_labels_counts_means_intervals_verified=True,
          thresholds_match_calibration=True, all_owned_ended=True, internal_report_sha256=sha(RUN / 'training/internal_report.json'),
          paired=paired, counts_summed_across_three_seeds=counts, source_hashes=hashes, internal_pass=False,
          final_evaluated=False, hota_measured=False, old_exit_code_preserved=1,
          service_failure='After all75 models and saved internal report, inherited CPU-only assertion rejects an initialized CUDA backend. Dynamic executed source line338 verified. No retraining or tolerance changes.'))
    print({k: {kk: vv for kk, vv in v.items()} for k, v in paired.items()}, flush=True)


if __name__ == '__main__': main()
