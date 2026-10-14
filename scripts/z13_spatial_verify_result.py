"""CPU-only verification of final Z13 artifacts and the frozen threshold rule."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from scripts.z13_threshold_diagnostic import exact_curve
    from scripts.z13_train import success, PRECISION, MIN_INTERVENTIONS, SUCCESS_SCALE
    from tracking.z13_spatial import make_quality_head
    from tracking.z13_checkpoint import atomic_json
    torch.set_num_threads(2)
    run = ROOT / 'runs/z13_spatial_v2_20261013'
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    result = json.loads((run / 'training_result.json').read_text(encoding='utf-8'))
    assert PRECISION == protocol['threshold_precision'] == .9
    assert MIN_INTERVENTIONS == protocol['minimum_interventions'] == 50
    assert SUCCESS_SCALE == protocol['success_scale'] == .25
    assert set(protocol['fit']).isdisjoint(protocol['tau'])
    assert len(protocol['fit']) == 104 and len(protocol['tau']) == 26
    report = {'created': datetime.now().astimezone().isoformat(), 'cpu_only': True,
              'completed': result['completed'], 'threshold_rule': '>=50 interventions, >=90% successes',
              'hota_measured': False, 'heldout_opened': False, 'jobs': [], 'arms': {}}
    for job in protocol['jobs']:
        folder = run / job['arm'] / job['name']
        status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
        state = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
        ident = json.loads((folder / 'identity.json').read_text(encoding='utf-8'))
        assert status['stage'] == 'predictions_saved'
        assert state['step'] == len(state['history']) == protocol['steps'] == 4000
        assert state['identity'] == ident
        assert ident['config']['protocol'] == protocol and ident['config']['job'] == job
        assert set(job['fit']).isdisjoint(job['predict'])
        model = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
        assert model['identity'] == ident
        assert all(torch.equal(model['readout'][k], v) for k, v in state['readout'].items())
        assert all(torch.equal(model['backbone'][k], v) for k, v in state['backbone'].items())
        rows = json.loads((folder / 'predictions.json').read_text(encoding='utf-8'))
        assert status['rows'] == len(rows)
        report['jobs'].append({'arm': job['arm'], 'name': job['name'], 'steps': state['step'],
                               'prediction_rows': len(rows), 'checkpoint_and_export_match': True})
        del state, model
    reference_keys = None
    for arm in protocol['arms']:
        folder = run / arm
        raw = json.loads((folder / 'full/predictions.json').read_text(encoding='utf-8'))
        scored = json.loads((folder / 'threshold_predictions.json').read_text(encoding='utf-8'))
        assert len(raw) == len(scored)
        keys = [(r['episode'], r['search_position'], r['kind']) for r in raw]
        if reference_keys is None:
            reference_keys = keys
        assert keys == reference_keys
        model = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False)
        saved = torch.load(folder / 'quality_checkpoint.pt', map_location='cpu', weights_only=False)
        assert saved['step'] == protocol['quality_steps'] == 1500
        assert saved['protocol_hash'] == hashlib.sha256((run / 'protocol.json').read_bytes()).hexdigest()
        assert all(torch.equal(model['quality'][k], v) for k, v in saved['head'].items())
        head = make_quality_head(torch)()
        head.load_state_dict(model['quality'])
        with torch.no_grad():
            scores = head(torch.tensor([r['features'] for r in raw], dtype=torch.float32)).sigmoid().numpy()
        assert np.isfinite(scores).all()
        max_difference = float(np.max(np.abs(scores - np.asarray([r['p_present'] for r in scored]))))
        assert max_difference <= 1e-7
        for original, with_quality in zip(raw, scored):
            assert all(original[k] == with_quality[k] for k in original if k != 'p_present')
            assert with_quality['presence_original'] == original['p_present']
        exact = exact_curve(scored)
        presence = exact_curve(raw)
        assert exact == json.loads((folder / 'quality_curve.json').read_text(encoding='utf-8'))
        assert presence == json.loads((folder / 'presence_curve.json').read_text(encoding='utf-8'))
        summary = json.loads((folder / 'tau.json').read_text(encoding='utf-8'))
        assert summary == result['arms'][arm]
        assert all(summary[k] == v for k, v in exact.items() if k != 'curve')
        assert model['tau']['tau'] == summary['tau']
        best = max((r for r in exact['curve'] if r['interventions'] >= 50), key=lambda r: r['precision'])
        eligible_rows = [r for r in scored if r['kind'] >= 0]
        assert sum(success(r) for r in eligible_rows) == exact['successful_if_oracle_selected']
        report['arms'][arm] = {'best_eligible_group': best,
            'tau': exact['tau'], 'labelled': exact['labelled'], 'positive': exact['positive'],
            'negative': exact['negative'], 'unknown_excluded': len(scored) - exact['labelled'],
            'successful_available': exact['successful_if_oracle_selected'],
            'presence_only_best_precision': presence['max_precision_at_minimum_count'],
            'quality_score_replay_max_difference': max_difference,
            'saved_curves_reproduced_exactly': True}
    report['passed'] = True
    atomic_json(run / 'final_verification.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
