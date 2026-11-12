"""Read-only metadata inventory. No model imports, scores, fitting, or GPU use."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
protocol = json.loads((ROOT / 'research_stage_localization_20261112/protocol.json').read_text(encoding='utf-8'))
records = []
arms = ['xywh', 'tracktrack', 'bytetrack', 'botsort', 'centerC', 'centerD', 'centerR1', 'legacyB', 'legacyC', 'legacyD']
for case in protocol['cases']:
    c = case['case']
    if c['old_id'] not in [832, 574]:
        continue
    refs = {832: (534, 31244), 574: (931, 35025)}
    for side, frame, raw_id, box in [('reference', refs[c['old_id']][0], refs[c['old_id']][1], c['reference_box']), ('current', c['frame'], c['raw_id'], c['current_box'])]:
        for arm in arms:
            base = ROOT / 'runs/z13_val20_matched_20261024' / arm
            sig = base / 'signatures' / c['video'] / 'signatures.npz'
            emb = base / 'masa' / (c['video'] + '.npz')
            if not sig.exists():
                continue
            with np.load(sig) as z:
                endpoints = []
                for fkey, bkey, ekey in [('start', 'first_box', 'first'), ('end', 'last_box', 'last')]:
                    for i in np.flatnonzero(z[fkey] == frame):
                        endpoints.append(dict(row=int(i), endpoint=ekey, track_id=int(z['track_id'][i]), box=z[bkey][i].tolist(), exact_box_equal=bool(np.array_equal(z[bkey][i].astype(float), np.array(box))), max_box_absolute_difference=float(np.abs(z[bkey][i].astype(float)-np.array(box)).max())))
                item = dict(old_id=c['old_id'], side=side, frame=frame, requested_raw_id=raw_id, requested_box=box, arm=arm, signatures=str(sig.relative_to(ROOT)), masa_exists=emb.exists(), endpoint_rows=endpoints)
            if emb.exists():
                with np.load(emb) as e:
                    item['masa'] = str(emb.relative_to(ROOT))
                    item['masa_keys_shapes_dtypes'] = {k: dict(shape=list(e[k].shape), dtype=str(e[k].dtype)) for k in e.files}
            records.append(item)
with (OUT / 'cache_inventory.json').open('x', encoding='utf-8') as f:
    json.dump(dict(method='Metadata only; no appearance similarity or retrieval result computed. Signatures indexed by track life, not raw detection id; endpoint rows shown only for exactly matching frame.', records=records), f, ensure_ascii=False, indent=2)
print(json.dumps(dict(records=len(records), cache_arms=sorted({r['arm'] for r in records if r['masa_exists']}), exact_endpoint_boxes=[{k:r[k] for k in ['old_id','side','arm','frame','masa']} | {'rows':[x for x in r['endpoint_rows'] if x['exact_box_equal']]} for r in records if r['masa_exists'] and any(x['exact_box_equal'] for x in r['endpoint_rows'])]), ensure_ascii=False))
