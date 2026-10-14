"""Evaluate the one fixed policy with the existing base+MASA evaluator."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'runs/z13_spatial_hota_v1_20261014'


def dirs(arm):
    return RUN / arm / 'signatures', RUN / arm / 'dino_compat', RUN / arm / 'masa'


def main():
    from scripts import z13_eval as evaluator
    from scripts import stitch_eval_v2 as stitch
    from scripts.fly_lifecycle_alive import sets
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    selected = sorted(v for v, *_ in sets(ROOT / 'runs/fly_signatures', ROOT / 'runs/fly_signatures_val')['dev'])
    assert selected == protocol['videos']
    assert 'dino_cos' not in stitch.HEADS['base_masa']
    assert all((dirs(arm)[1] / f'{vid}.npz').is_file()
               for arm in protocol['arms'] for vid in selected)
    # DINO is excluded from this fixed head. Shape-compatible zero features
    # satisfy the general feature reader, without computing an unused network.
    evaluator.dirs = dirs
    evaluator.OUT = RUN / 'eval'
    sys.argv = ['z13_eval.py', '--arms', *protocol['arms']]
    evaluator.main()
    path = RUN / 'eval/report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    report.update(protocol=str((RUN / 'protocol.json').relative_to(ROOT)),
                  diagnostic_policy=True, independent_confirmation=False,
                  old_90_percent_gate_unchanged=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
