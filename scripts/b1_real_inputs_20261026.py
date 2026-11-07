"""Б1, подготовка: реальные входы и замороженная база, без нового теста train26."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '3'
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/b1_real_inputs_20261026'
MSK = timezone(timedelta(hours=3))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    path = RUN / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.next')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def status(stage, **values):
    write('status.json', dict(state='running', stage=stage, updated=datetime.now(MSK).isoformat(), **values))


def journal(text):
    path = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    raw = path.read_bytes()
    newline = '\r\n' if b'\r\n' in raw else '\n'
    marker = ('## Журнал изменений' + newline).encode()
    entry = f'\n**{datetime.now(MSK):%d.%m.%Y, %H:%M} MSK** — Б1: {text}\n\n'
    assert raw.count(marker) == 1
    path.write_bytes(raw.replace(marker, marker + entry.replace('\n', newline).encode(), 1))


def prepare():
    assert not RUN.exists(), 'Preserve any existing preparation'
    RUN.mkdir()
    p = read(ROOT / 'runs/z13_hybrid_fly_confirm_20261026/protocol.json')
    videos = p['fit84'] + p['fit20']
    files = ['scripts/b1_real_inputs_20261026.py', 'tracking/b1_real_inputs.py',
             'tracking/flyvis_temporal_residual.py', 'scripts/z13_identity_gate_readout_20261026.py',
             'scripts/z13_identity_gate_tagged_20261026.py', 'scripts/z13_identity_gate_relative_20261026.py',
             'scripts/z13_hybrid_identity_20261026.py', 'scripts/z13_hybrid_fly_confirm_20261026.py',
             'runs/z13_hybrid_fly_confirm_20261026/protocol.json',
             'runs/z13_hybrid_fly_confirm_20261026/report.json',
             'runs/z13_identity_gate_tagged_20261026/features_D.npz',
             'runs/z13_box_supervision_followup_20261025/temporal_readiness.json']
    for video in videos:
        files += [f'runs/z13_data/{video}.npz', f'runs/z13_temporal_pair_readiness_20261025/inputs/{video}.json',
                  f'runs/z13_box_supervision_followup_20261025/cache/videos/{video}.json',
                  f'runs/z13_masa_regions_20261026/recovery1/videos/{video}.npz',
                  f'runs/z13_masa_exemplar_snap_20261026/videos/{video}.npz']
    for split in ('fit84', 'fit20'):
        files += [f.relative_to(ROOT).as_posix() for f in sorted((ROOT / f'runs/z13_region_assignment_20261025/recovery2/features/D/{split}').glob('chunk_*.*'))]
    start = datetime.now(MSK)
    protocol = dict(id='flyvis_temporal_residual_association_v1', stage='second preparation step, real data and baseline',
                    created=start.isoformat(), timezone='Europe/Moscow', cpu_seconds=900, cpu_threads=3,
                    deadline=min(start + timedelta(seconds=900), datetime(2026, 10, 27, 7, 30, tzinfo=MSK)).isoformat(),
                    authorization='Автор: «Давай Б1 займёмся». Подготовка, воспроизведение базы и краткий замер; длительное обучение не запускается.',
                    fit84=p['fit84'], fit20=p['fit20'], evaluate26=[],
                    baseline='Exact old feature recipe and HistGradientBoosting settings; inner4 OOF fit84, full fit84 -> fit20; same gate and calibration. Newly reconstructed weights are saved.',
                    history='Original observed track mask ex_mask at episode slot3, same episode prefix ending at current; no identity/good/harm selects history. Missing history yields zero correction; retain all queries.',
                    actions='Unchanged implicit baseline action, explicit NULL at slot0, then real detection choices. No synthetic candidates.',
                    acceptance=['all104 input bindings preserved and all candidate masks supported',
                                'reference strictly earlier when history is enabled; no future prefix',
                                'zero identity residual exactly reproduces probabilities, gate outputs and calibrated decisions',
                                'saved model reload gives exact predictions; threshold reproduces old frozen base threshold'],
                    limitations='Reused development104. PCA deliberately reproduces old unlabeled fit104 recipe, not an independent validation. Original per-row scores/models were not saved: no claim of bitwise equality with unavailable historical outputs. No train26 evaluation or HOTA.',
                    sha256={name: sha(ROOT / name) for name in files})
    write('protocol.json', protocol)
    write('status.json', dict(state='ready', stage='real data binding and baseline reconstruction registered', updated=start.isoformat()))
    catalog = ROOT / 'experiments/project_history.json'
    original = catalog.read_bytes()
    content = json.loads(original.decode('utf-8'))
    card = next(c for c in content['experiments'] if c['id'] == protocol['id'])
    card['evidence'].append(dict(kind='file', ref='runs/b1_real_inputs_20261026/protocol.json',
                                 claim='Вторая ступень подготовки Б1 до расчёта: реальные входы и восстановление базы'))
    assert catalog.read_bytes() == original, 'Concurrent catalog edit; preserve it'
    text = json.dumps(content, ensure_ascii=False, indent=1) + '\n'
    catalog.write_bytes(text.replace('\n', '\r\n').encode() if b'\r\n' in original else text.encode())
    journal('зарегистрировано соединение реальных входов и восстановление базы. '
            '*Постановка.* `runs/b1_real_inputs_20261026/protocol.json`: все fit84+fit20, без новой оценки train26; '
            'история по наблюдаемой рамке, неизменные рамочные действия, прежний рецепт базы MASA. '
            'CPU до 900 с, внешний ограничитель, без обучения ядра. Сохраняются новые веса базы и точная проверка нулевой поправки. '
            'Прежние построчные прогнозы базы не сохранены, поэтому их побитное воспроизведение не заявляется. '
            'HOTA не измеряется, гипотеза остаётся непроверенной.')


def load_data(p):
    from scripts import z13_identity_gate_readout_20261026 as old
    parts, entries, bindings = [], {}, {}
    from tracking.b1_real_inputs import bind_observation
    counts = dict(queries=0, real_regions=0, history=0, history_with_regions=0)
    for split in ('fit84', 'fit20'):
        chunks = []
        for path in sorted((ROOT / f'runs/z13_region_assignment_20261025/recovery2/features/D/{split}').glob('chunk_*.npz')):
            assert sha(path) == read(path.with_suffix('.json'))['sha256']
            with np.load(path) as z:
                chunks.append({k: z[k] for k in z.files})
        part = old.merge(chunks)
        order = np.argsort(part['uid'])
        part = {k: v[order] for k, v in part.items()}
        part['name'] = np.array(p[split])[part['video']]
        extra = np.zeros((*part['mask'].shape, 6), np.float32)
        for video in p[split]:
            status('binding real observations', video=video, videos_done=len(bindings), videos_total=104)
            raw = read(ROOT / f'runs/z13_box_supervision_followup_20261025/cache/videos/{video}.json')['entries']
            pointers = read(ROOT / f'runs/z13_temporal_pair_readiness_20261025/inputs/{video}.json')['rows']
            with np.load(ROOT / f'runs/z13_data/{video}.npz') as z:
                archive = {k: z[k] for k in ('ep_offset', 'ex_mask', 'fr_frame', 'fr_crop')}
            rows = []
            for row in pointers:
                uid = int(row['uid']); entry = raw[str(uid)]
                bound = bind_observation(row, archive, entry['choices'])
                assert bound['reference_index'] < bound['sequence_frames'] - 1
                rows.append(bound); entries[uid] = entry
                counts['queries'] += 1; counts['real_regions'] += len(bound['choices']) - 1
                counts['history'] += bound['has_history']
                counts['history_with_regions'] += bound['has_history'] and len(bound['choices']) > 1
            bindings[video] = rows
            write(f'inputs/{video}.json', dict(rows=rows, ground_truth_used_for_binding=False))
        for i, uid in enumerate(part['uid']):
            e = entries[int(uid)]; n = int(part['mask'][i].sum())
            assert n == len(e['choices']) and int(part['kind'][i]) == e['kind']
            part['good'][i, :n] = e['good']; part['harm'][i, :n] = e['harm']
            extra[i, :n] = e['context']
        part['geo'] = np.concatenate([part['geo'], extra], -1)
        parts.append(part)
    d = old.merge(parts)
    assert set(map(int, d['uid'])) == {r['uid'] for rows in bindings.values() for r in rows}
    with np.load(ROOT / 'runs/z13_identity_gate_tagged_20261026/features_D.npz') as z:
        assert np.array_equal(z['uid'], d['uid'])
        d['Gtag'] = z['Gtag']; d['ident'] = z['ident']
    return d, bindings, counts


def run():
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from scripts import z13_identity_gate_readout_20261026 as old
    from scripts import z13_identity_gate_relative_20261026 as relative
    from scripts import z13_hybrid_identity_20261026 as hybrid
    from scripts.z13_hybrid_fly_confirm_20261026 import snapped_fit
    from scripts.z13_identity_gate_tagged_20261026 import policy_net
    start = time.monotonic(); p = read(RUN / 'protocol.json')
    assert not (RUN / 'report.json').exists()
    for name, expected in p['sha256'].items():
        assert sha(ROOT / name) == expected, name
    d, bindings, counts = load_data(p)
    previous = read(ROOT / 'runs/z13_box_supervision_followup_20261025/temporal_readiness.json')['previous_query']
    status('reconstructing identical appearance inputs')
    A = hybrid.appearance(d, previous)
    S = snapped_fit(d, np.ones(len(d['uid']), bool))
    G = np.concatenate([d['geo'][..., :8], d['Gtag'], A, S[..., :1]], -1)
    B = np.concatenate([G, relative.relative(G, d['mask'], list(range(G.shape[-1]))), S[..., 1:]], -1)
    np.savez_compressed(RUN / 'base_inputs.npz', uid=d['uid'], name=d['name'], F1=B, geo=d['geo'], mask=d['mask'])
    np.savez_compressed(RUN / 'supervision.npz', uid=d['uid'], ident=d['ident'], good=d['good'], harm=d['harm'], kind=d['kind'])
    real = d['mask'].copy(); real[:, 0] = False
    labeled = real & (d['ident'] >= 0)
    probability = np.full(d['mask'].shape, np.nan)
    inner = {v: i % 4 for i, v in enumerate(sorted(p['fit84']))}
    models = []
    for fold in range(5):
        train_v = [v for v in p['fit84'] if fold == 4 or inner[v] != fold]
        output_v = p['fit20'] if fold == 4 else [v for v in p['fit84'] if inner[v] == fold]
        tr = np.isin(d['name'], train_v)[:, None] & labeled
        te = np.isin(d['name'], output_v)[:, None] & real
        status('reconstructing frozen identity base', fold=fold, models_done=fold, models_total=6)
        path = RUN / f'identity_fold{fold}.joblib'
        if path.exists():
            model = joblib.load(path)
        else:
            model = HistGradientBoostingClassifier(**old.PROBE).fit(B[tr], d['ident'][tr])
            joblib.dump(model, path)
        probability[te] = model.predict_proba(B[te])[:, 1]
        models.append(model)
    features = old.gate_features(d, probability)
    known = d['kind'] >= 0
    alt = d['mask'] & (d['geo'][..., -1] < .5) & known[:, None]
    tr = np.isin(d['name'], p['fit84'])[:, None] & alt & (d['good'] | d['harm'])
    te = np.isin(d['name'], p['fit20'])[:, None] & alt
    status('reconstructing frozen action gate', models_done=5, models_total=6)
    gate_model = HistGradientBoostingClassifier(**old.PROBE).fit(features[tr], d['good'][tr].astype(int))
    joblib.dump(gate_model, RUN / 'gate.joblib')
    gate = np.full(probability.shape, np.nan); gate[te] = gate_model.predict_proba(features[te])[:, 1]
    grid = [(float(np.mean(list(policy_net(d, gate, p['fit20'], threshold)[0].values()))), threshold) for threshold in old.THRESHOLDS]
    threshold = max(grid)[1]
    historical_threshold = read(ROOT / 'runs/z13_hybrid_fly_confirm_20261026/report.json')['thresholds']['base']
    assert threshold == historical_threshold, (threshold, historical_threshold)
    status('zero residual and saved-weight regression')
    import torch
    torch.set_num_threads(3)
    from tracking.b1_real_inputs import corrected_identity_probability
    prob2 = corrected_identity_probability(torch.from_numpy(probability), torch.zeros_like(torch.from_numpy(probability)), torch.from_numpy(real)).numpy()
    assert np.array_equal(prob2, probability, equal_nan=True)
    features2 = old.gate_features(d, prob2)
    predicted2 = joblib.load(RUN / 'gate.joblib').predict_proba(features2[te])[:, 1]
    assert np.array_equal(predicted2, gate[te])
    # Reload the full-fit identity model on all calibration real regions.
    calibration_real = np.isin(d['name'], p['fit20'])[:, None] & real
    assert np.array_equal(joblib.load(RUN / 'identity_fold4.joblib').predict_proba(B[calibration_real])[:, 1], probability[calibration_real])
    scores2 = np.full(gate.shape, np.nan); scores2[te] = predicted2
    assert policy_net(d, scores2, p['fit20'], threshold) == policy_net(d, gate, p['fit20'], threshold)
    np.savez_compressed(RUN / 'base_predictions.npz', uid=d['uid'], identity_probability=probability, calibration_gate=gate)
    available = sorted((row for rows in bindings.values() for row in rows if row['has_history'] and len(row['choices']) > 1),
                       key=lambda row: (row['sequence_frames'], row['uid']))
    samples = [available[int(q * (len(available) - 1))] for q in (0., .5, .9)]
    write('gpu_samples.json', dict(selection='fixed length quantiles0/50/90; no label or prediction selection', rows=samples))
    report = dict(passed=True, stage='real data and baseline preparation', videos=104, **counts,
                  base_feature_shape=list(B.shape), threshold=threshold, historical_threshold_matches=True,
                  zero_probability_exact=True, zero_gate_prediction_exact=True, zero_decisions_exact=True,
                  model_reload_exact=True, train26_evaluations=0, hota_measured=False, core_training_steps=0,
                  seconds=time.monotonic() - start, cuda_initialized=torch.cuda.is_initialized(),
                  gpu_cost_measured=False, historical_row_predictions_available=False,
                  limitation=p['limitations'], gpu_sample_lengths=[r['sequence_frames'] for r in samples])
    write('report.json', report)
    write('status.json', dict(state='preparation_completed', stage='GPU cost not yet measured; scientific pilot not started', updated=datetime.now(MSK).isoformat()))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    try:
        prepare() if sys.argv[1] == '--prepare' else run()
    except Exception as error:
        if RUN.exists():
            write('failure.json', dict(error=repr(error), updated=datetime.now(MSK).isoformat()))
        raise
