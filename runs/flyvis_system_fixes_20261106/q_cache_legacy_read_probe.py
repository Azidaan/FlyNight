from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tracking.cache_readonly import require_cached_videos
from tracking.detection_cache import load_video_cache

folder = ROOT / 'runs/det_cache_q'
path = sorted((folder / 'val').glob('*.npz'))[0]
meta = folder / 'meta.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = {str(p.relative_to(ROOT)): sha(p) for p in (path, meta)}
require_cached_videos(folder, 'val', [path.stem], required=('q_mean', 'q_min', 'q_median'))
arrays = load_video_cache(folder, 'val', path.stem)
after = {str(p.relative_to(ROOT)): sha(p) for p in (path, meta)}
report = dict(passed=before == after, video=path.stem, frames=len(arrays['frame_names']),
              detections=len(arrays['conf']), unchanged_sha256=before,
              legacy_metadata_lacks_weight_hash='weights_sha256' not in json.loads(meta.read_text()),
              array_keys=sorted(arrays), source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
              (ROOT/'tracking/cache_readonly.py', ROOT/'tracking/cache_integrity.py', ROOT/'tracking/detection_cache.py')})
assert report['passed'] and report['legacy_metadata_lacks_weight_hash']
target = Path(__file__).with_name('q_cache_legacy_read_report.json')
target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
