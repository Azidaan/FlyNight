"""Read-only paired audit of saved localization rows; never fits a selector on train26."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/z13_readout_overlap_20261018'
OLD = ROOT / 'runs/z13_continue_20261017/baseline/rows.json'
NEW = ROOT / 'runs/z13_target_model_20261017/D/rows.json'


def audit(old, new):
    def index(rows):
        out = {r['label']: r for r in rows}
        if len(out) != len(rows):
            raise ValueError('Duplicate labels')
        return out
    a, b = index(old), index(new)
    if a.keys() != b.keys():
        raise ValueError('Different row sets')
    cells = {k: [] for k in ('both', 'old_only', 'new_only', 'neither')}
    videos = {}
    for key in sorted(a):
        x, y = a[key], b[key]
        if (x['video'], x['kind']) != (y['video'], y['kind']):
            raise ValueError('Different metadata')
        if x['kind'] != 1:
            continue
        if not np.isclose(x['scale'], y['scale']) or not np.isclose(x['err_pred'], y['err_pred']):
            raise ValueError('Different reference geometry')
        cell = 'both' if x['useful'] and y['useful'] else ('old_only' if x['useful'] else ('new_only' if y['useful'] else 'neither'))
        row = {'label': key, 'video': x['video'], 'pos': y['pos'],
               'old_error_rel': x['err'] / x['scale'], 'new_error_rel': y['err'] / y['scale'],
               'prediction_error_rel': x['err_pred'] / x['scale'],
               'center_disagreement_rel': float(np.linalg.norm(np.array(x['center'])-y['center'])/x['scale'])}
        cells[cell].append(row)
        v = videos.setdefault(x['video'], dict.fromkeys(('both', 'old_only', 'new_only', 'neither'), 0))
        v[cell] += 1
    summary = {}
    for k, rows in cells.items():
        summary[k] = {'n': len(rows), 'videos': len({r['video'] for r in rows})}
        for f in ('old_error_rel', 'new_error_rel', 'center_disagreement_rel'):
            summary[k]['median_' + f] = float(np.median([r[f] for r in rows])) if rows else None
        summary[k]['new_error_above_one_scale'] = sum(r['new_error_rel'] > 1 for r in rows)
    denom = {v: sum(c.values()) for v, c in videos.items()}
    # This oracle uses evaluation truth and is not an implementable model or HOTA ceiling.
    d = np.array([c['new_only']/denom[v] for v,c in sorted(videos.items())])
    boot = d[np.random.default_rng(1).integers(0, len(d), (5000,len(d)))].mean(1)
    result = {'rows': len(a), 'positives': sum(len(r) for r in cells.values()), 'cells': summary, 'per_video': videos,
              'oracle_gain_over_old_mean_video': float(d.mean()), 'oracle_ci95': np.quantile(boot,[.025,.975]).tolist(),
              'oracle_warning': 'Ground-truth selection between two saved centers; no deployable gain, no HOTA ceiling.',
              'decision': 'prepare_target_model_change' if len(cells['new_only']) else 'do_not_build_selector'}
    return result, cells


def main():
    if not (RUN/'protocol.json').exists() or (RUN/'report.json').exists():
        raise RuntimeError('Protocol required; completed audit must not be overwritten')
    def write(name, value):
        (RUN/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write('status.json', {'state':'running','stage':'paired saved rows','done':0,'total':1})
    proto=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'))
    for path, digest in proto['sources_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source changed: '+path)
    rep,cells=audit(json.loads(OLD.read_text()),json.loads(NEW.read_text()))
    write('report.json',rep); write('cells.json',cells)
    write('status.json',{'state':'completed','stage':'paired saved rows','done':1,'total':1})
    print(json.dumps(rep,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
