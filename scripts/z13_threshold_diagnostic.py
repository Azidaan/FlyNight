"""One bounded replay of fixed Z13 models; no training or dev evaluation."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_checkpoint import atomic_json


def exact_curve(rows):
    """All distinct score boundaries, retaining every tied score together."""
    import numpy as np
    from scripts.z13_train import success, MIN_INTERVENTIONS, PRECISION
    labelled = [r for r in rows if r['kind'] >= 0]
    ranked = sorted(labelled, key=lambda r: r['p_present'], reverse=True)
    if not ranked:
        return {'tau': None, 'curve': [], 'labelled': 0, 'successful_if_oracle_selected': 0}
    scores = np.array([r['p_present'] for r in ranked])
    successes = np.cumsum([success(r) for r in ranked])
    ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    curve = [{'tau': float(scores[i]), 'interventions': int(i + 1),
              'successful': int(successes[i]), 'precision': float(successes[i] / (i + 1))} for i in ends]
    eligible = [r for r in curve if r['interventions'] >= MIN_INTERVENTIONS]
    accepted = [r for r in eligible if r['precision'] >= PRECISION]
    best = min(accepted, key=lambda r: r['tau']) if accepted else None
    return {'tau': best['tau'] if best else None, 'accepted': best,
            'max_precision_at_minimum_count': max((r['precision'] for r in eligible), default=None),
            'successful_if_oracle_selected': int(successes[-1]), 'labelled': len(labelled),
            'positive': sum(r['kind'] == 1 for r in ranked), 'negative': sum(r['kind'] == 0 for r in ranked),
            'curve': curve}


def json_default(value):
    return value.item()


def worker(output, deadline):
    import numpy as np
    import torch
    from scripts.z13_train import Episodes, build, evaluate, choose_tau
    from tracking.z13_checkpoint import RunControl
    run = ROOT / 'runs/z13_codex_window_20261013'
    control = RunControl(output, output / 'stop.request', deadline)
    split = json.loads((run / 'D/split.json').read_text(encoding='utf-8'))
    assert len(split['tau']) == 26
    for a in ('B', 'C'):
        assert json.loads((run / a / 'split.json').read_text(encoding='utf-8')) == split
    data = Episodes(split['tau'])
    ids = [i for i, label in enumerate(data.labels) if not label[8]]
    report = {'scope': 'Fixed models; original 26 train threshold videos; no training, no dev, no heldout val', 'arms': {}}
    for arm in ('B', 'C', 'D'):
        control.check()
        torch.manual_seed(0)
        control.status(state='running', stage='prediction_replay', arm=arm)
        est, bank, geom, bb, ro, dev = build(arm, torch)
        model = torch.load(run / arm / 'model.pt', map_location='cpu', weights_only=False)
        ro.load_state_dict(model['readout'])
        ro.eval()
        if arm == 'B':
            bb.load_state_dict(model['backbone']); bb.eval()
        elif arm == 'D':
            bb.net.load_state_dict(model['network'])
        started = time.time()
        rows = evaluate(torch, data, bb, ro, dev, geom, bank.eye_size, ids, control=control)
        for r in rows:
            label = data.labels[r['label']]
            r['episode'] = int(label[0]); r['search_position'] = int(label[1])
        (output / f'{arm}_predictions.json').write_text(json.dumps(rows, default=json_default), encoding='utf-8')
        exact = exact_curve(rows)
        (output / f'{arm}_exact_curve.json').write_text(json.dumps(exact, indent=2), encoding='utf-8')
        old = choose_tau(rows)
        positives = [r for r in rows if r['kind'] == 1]
        detail = {k: v for k, v in exact.items() if k != 'curve'}
        detail.update(original_rounded_rule=old, replay_seconds=time.time() - started,
                      unknown=sum(r['kind'] == -1 for r in rows),
                      positive_within_quarter_scale=sum(r['err'] <= .25 * r['scale'] for r in positives),
                      positive_improves_prediction=sum(r['err'] < r['err_pred'] for r in positives),
                      median_positive_error_relative=float(np.median([r['err'] / r['scale'] for r in positives])) if positives else None,
                      median_baseline_error_relative=float(np.median([r['err_pred'] / r['scale'] for r in positives])) if positives else None)
        detail['rounding_changes_admission'] = old['tau'] is None and exact['tau'] is not None
        report['arms'][arm] = detail
        atomic_json(output / 'report.json', report)
        print(json.dumps({'arm': arm, **detail}, ensure_ascii=False), flush=True)
        del est, bank, bb, ro
        torch.cuda.empty_cache()
    control.status(state='completed', stage='diagnostic_complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--deadline')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/z13_threshold_diagnostic_20261013')
    args = parser.parse_args()
    output = args.output.resolve(); output.relative_to(ROOT); output.mkdir(exist_ok=True)
    if args.worker:
        worker(output, args.deadline)
        return
    from datetime import timedelta
    started = datetime.now().astimezone()
    deadline = (started + timedelta(seconds=570)).isoformat(timespec='seconds')
    flags = subprocess.CREATE_NO_WINDOW | subprocess.IDLE_PRIORITY_CLASS
    with (output / 'log.txt').open('a', encoding='utf-8') as log:
        proc = subprocess.Popen([str(ROOT / '.venv/Scripts/python.exe'), '-u', str(Path(__file__).resolve()),
                                 '--worker', '--output', str(output), '--deadline', deadline],
                                cwd=ROOT, creationflags=flags, stdout=log, stderr=subprocess.STDOUT)
        atomic_json(output / 'controller.json', {'pid': os.getpid(), 'child_pid': proc.pid,
                    'started': started.isoformat(), 'deadline': deadline, 'hard_limit_seconds': 600,
                    'permission': 'User authorized up to 10 minutes GPU for frozen prediction replay'})
        try:
            code = proc.wait(timeout=600)
            atomic_json(output / 'controller_result.json', {'exit_code': code, 'ended': datetime.now().astimezone().isoformat()})
        finally:
            if proc.poll() is None:
                subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], creationflags=subprocess.CREATE_NO_WINDOW, check=False)


if __name__ == '__main__':
    main()
