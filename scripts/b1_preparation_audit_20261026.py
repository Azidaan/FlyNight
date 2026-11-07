"""Независимое чтение артефактов подготовки Б1; модели не запускает."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/b1_real_inputs_20261026'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = read(RUN / 'protocol.json'); report = read(RUN / 'report.json')
    original_controller = read(RUN / 'controller_result.json')
    controller = read(RUN / 'recovery1_controller.json')
    assert original_controller['exit_code'] != 0
    assert controller['exit_code'] == 0 and controller['all_owned_ended'] and not controller['timed_out']
    assert datetime.fromisoformat(controller['updated']).timestamp() <= datetime.fromisoformat(p['deadline']).timestamp()
    for name, expected in p['sha256'].items():
        assert sha(ROOT / name) == expected, name
    service = read(RUN / 'service_recovery1.json')
    assert sha(RUN / 'protocol.json') == service['original_protocol_sha256']
    for name, expected in service['sha256'].items():
        assert sha(ROOT / name) == expected, name
    uids, regions, history, usable = set(), 0, 0, 0
    videos = p['fit84'] + p['fit20']
    for video in videos:
        rows = read(RUN / 'inputs' / (video + '.json'))
        assert rows['ground_truth_used_for_binding'] is False
        with np.load(ROOT / f'runs/z13_data/{video}.npz') as z:
            offsets, frames, reference_masks = z['ep_offset'], z['fr_frame'], z['ex_mask']
        for row in rows['rows']:
            assert row['uid'] not in uids; uids.add(row['uid'])
            episode = row['episode_local']; start, stop = map(int, offsets[episode:episode + 2])
            assert start == row['prefix_start'] and start + 3 == row['reference_array_index']
            assert start + 3 < row['current_array_index'] < stop
            assert row['choices'][0] == -1 and row['null_slot'] == 0
            assert np.array_equal(np.flatnonzero(reference_masks[episode]), row['reference_mask'])
            if row['has_history']:
                assert frames[start + 3] < frames[row['current_array_index']]
            regions += len(row['choices']) - 1; history += row['has_history']
            usable += row['has_history'] and len(row['choices']) > 1
    with np.load(RUN / 'base_inputs.npz') as z:
        assert set(z['uid'].tolist()) == uids
        assert z['F1'].shape == tuple(report['base_feature_shape'])
    with np.load(RUN / 'supervision.npz') as z:
        assert set(z['uid'].tolist()) == uids
    with np.load(RUN / 'base_predictions.npz') as z:
        assert set(z['uid'].tolist()) == uids
    assert (len(uids), regions, history, usable) == (report['queries'], report['real_regions'], report['history'], report['history_with_regions'])
    assert all(report[key] for key in ('zero_probability_exact', 'zero_gate_prediction_exact', 'zero_decisions_exact', 'model_reload_exact'))
    assert not report['hota_measured'] and report['train26_evaluations'] == report['core_training_steps'] == 0
    gpu = read(RUN / 'gpu_cost/report.json')
    gp = read(RUN / 'gpu_cost/protocol.json')
    gc = read(RUN / 'gpu_cost/profile_controller.json')
    for name, expected in gp['sha256'].items():
        assert sha(ROOT / name) == expected, name
    assert gc['exit_code'] == 0 and gc['all_owned_ended'] and not gc['timed_out']
    assert gpu['passed'] and gpu['core_parameters_unchanged'] and gpu['zero_start_exact']
    assert gpu['core_optimizer_steps'] == gpu['train26_evaluations'] == 0 and not gpu['hota_measured']
    assert [x['frames'] for x in gpu['samples']] == report['gpu_sample_lengths']
    assert all(all(np.isfinite(x['core_gradient_norms'])) and min(x['core_gradient_norms']) > 0 for x in gpu['samples'])
    result = dict(passed=True, videos=len(videos), queries=len(uids), real_regions=regions,
                  unchanged_bindings=True, labels_separate_from_inputs=True, same_episode_causal_reference=True,
                  zero_residual_regression=True, saved_model_reload=True, original_deadline_preserved=True,
                  historical_row_predictions_available=False, all_owned_workers_ended=True,
                  train26_evaluations=0, hota_measured=False, core_training_steps=0,
                  environment={name: importlib.metadata.version(name) for name in ('numpy', 'scikit-learn', 'torch', 'joblib')},
                  model_sha256={f.name: sha(f) for f in sorted(RUN.glob('*.joblib'))},
                  updated=datetime.now(timezone.utc).isoformat())
    (RUN / 'independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
