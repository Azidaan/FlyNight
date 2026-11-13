"""CPU-only artifact audit, without consulting similarity outcomes."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
def read(path):
    return json.loads(path.read_text(encoding='utf8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

expected = 'e56f4392a3bb1a1f54e33dc31d8acb32f31c80d79a95697055c932f03e3c10cd'
meta = read(HERE / 'masks_reviewed.json')
receipt = read(HERE / 'mask_receipt.json')
queue = read(HERE.parent / 'pixel_queue_execution.json')
reviewed = np.load(HERE / 'masks_reviewed.npz', allow_pickle=False)
raw = np.load(HERE / 'masks.npz', allow_pickle=False)
frames = [1159, 1164, 1165]
groups = ['target', 'neighbor', 'occluder', 'background', 'uncertain']
renames = dict(target='red', neighbor='black', occluder='foreground', background='background', uncertain='uncertain')
assert set(reviewed.files) == {f'{frame}_{group}' for frame in frames for group in groups}
assert sha(HERE / 'masks_reviewed.npz') == expected == meta['sha256']
assert sha(HERE / 'masks.npz') == receipt['mask_sha256'] == meta['automatic_sha256']
sources = {HERE / 'extract_masks.py': receipt['source_sha256'], HERE / 'prompts.json': receipt['prompts_sha256'], ROOT / 'weights/sam2_official_20261027/sam2.1_hiera_tiny.pt': receipt['weights_sha256']}
for rec in receipt['records']:
    sources[HERE.parent / 'frames' / f"{rec['frame']}.png"] = rec['image_sha256']
assert all(sha(path) == checksum for path, checksum in sources.items())
records = []
for frame in frames:
    components = [reviewed[f'{frame}_{group}'] for group in groups]
    assert all(arr.shape == (720, 1280) and arr.dtype == np.bool_ for arr in components)
    totals = np.stack(components).sum(axis=0)
    assert np.all(totals == 1)
    counts = {group: int(reviewed[f'{frame}_{group}'].sum()) for group in groups}
    old = next(rec for rec in meta['records'] if rec['frame'] == frame)
    assert counts == {group: old['counts'][renames[group]] for group in groups}
    overlap = int((raw[f'{frame}_target_raw'] & raw[f'{frame}_neighbor_raw']).sum())
    assert overlap == old['sam_overlap_pixels']
    assert np.array_equal(reviewed[f'{frame}_occluder'], raw[f'{frame}_person_gray_raw'] | raw[f'{frame}_person_red_raw'] | raw[f'{frame}_cart_raw'])
    union = raw[f'{frame}_target_raw'] | raw[f'{frame}_neighbor_raw'] | reviewed[f'{frame}_occluder']
    assert np.array_equal(reviewed[f'{frame}_background'], ~union)
    assert not np.any(reviewed[f'{frame}_uncertain'] & reviewed[f'{frame}_background'])
    assert np.all(~reviewed[f'{frame}_target'] | raw[f'{frame}_target_raw'])
    assert np.all(~reviewed[f'{frame}_neighbor'] | raw[f'{frame}_neighbor_raw'])
    records.append(dict(frame=frame, counts=counts, sum_pixels=int(totals.sum()), disjoint_exhaustive=True, raw_vehicle_overlap_pixels=overlap))
frozen = datetime.fromisoformat(meta['created'])
started = datetime.fromisoformat(queue['jobs'][0]['started'])
assert frozen < started
out = dict(passed=True, audited_at=datetime.now(timezone(timedelta(hours=3))).isoformat(), mask_frozen_at=meta['created'], pixel_job_started_at=queue['jobs'][0]['started'], fixed_before_pixel_job_seconds=(started-frozen).total_seconds(), evidence_of_before_outcomes='Mask metadata creation time precedes guarded pixel worker start; this audit reads queue metadata but no feature outputs or measured similarity outcomes.', mask_sha256=expected, raw_failed_mask_sha256=meta['automatic_sha256'], raw_failed_masks_retained=True, records=records, current_provenance_verified={str(p.relative_to(ROOT)): h for p,h in sources.items()}, construction_sources_now={name: sha(HERE/name) for name in ['finalize_masks.py','export_reviewed.py','masks_reviewed.json','prompts.json']}, visual_review='All three core panels visually inspected in agent before similarity outcomes; parent independently inspected 1159 and 1165. Image-only manual cores do not constitute segmentation ground truth.', limitations=['Target and neighbor masks are conservative visible cores, not entire object silhouettes.', 'Raw SAM masks merge the cars on 1164 and 1165; uncertain pixels remain an explicit fifth group.', 'Background means scene outside raw masks and can contain other distant objects.', 'This audit verifies partition and provenance, not automatic segmentation accuracy or persistent identity.'])
with (HERE/'finalreview.json').open('x',encoding='utf8') as stream:
    json.dump(out,stream,ensure_ascii=False,indent=2)
    stream.write('\n')
print(json.dumps(dict(passed=True, sha256=expected, fixed_before_pixel_job_seconds=out['fixed_before_pixel_job_seconds']), ensure_ascii=False))
