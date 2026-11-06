"""Compare cached native CPU YOLO outputs before/after cache repairs."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from datasets.bdd100k_loader import DATA_ROOT
from tracking import detection_cache as repaired
from tracking.cache_integrity import weights_identity


def main(output_dir=None):
    out = (Path(output_dir) if output_dir is not None else
           ROOT/'runs/flyvis_repairs_20261106'/('cache_native_repeat_'+str(time.time_ns()))).resolve()
    out.relative_to(ROOT)
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    source_dir = DATA_ROOT/'images/track/train/0000f77c-6257be58'
    source = sorted(source_dir.glob('*.jpg'))[:3]
    assert len(source) == 3
    target = out/'input/images/track/train/parity'
    target.mkdir(parents=True)
    for path in source:
        shutil.copyfile(path, target/path.name)
    weights = ROOT/'weights/baseline_seed42/best.pt'
    identity = weights_identity(weights)
    original_file = ROOT/'runs/flyvis_repairs_20261106/cache_before/detection_cache.py'
    spec = importlib.util.spec_from_file_location('cache_original_20261106', original_file)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    times = {}
    for name, module in [('original', original), ('repaired', repaired)]:
        module.DATA_ROOT = out/'input'
        before = time.perf_counter()
        module.build_detection_cache(str(weights), ['parity'], 'train', out/name,
                                     device='cpu', imgsz=960)
        times[name] = time.perf_counter()-before
    with np.load(out/'original/train/parity.npz') as a, np.load(out/'repaired/train/parity.npz') as b:
        assert set(a.files) <= set(b.files)
        assert set(b.files)-set(a.files) == {'source_frame_sha256'}
        equality = {k: bool(np.array_equal(a[k], b[k])) for k in a.files}
        detections = len(a['conf'])
    assert all(equality.values())
    assert weights_identity(weights) == identity
    assert not torch.cuda.is_initialized()
    result = dict(passed=True, frames=3, detections=detections, arrays_exact=equality,
        weights=identity, weights_unchanged=True, cuda_initialized=torch.cuda.is_initialized(),
        seconds=times, source_frames=[str(p.relative_to(ROOT)) for p in source],
        original_source_sha256=hashlib.sha256(original_file.read_bytes()).hexdigest(),
        repaired_source_sha256=hashlib.sha256(Path(repaired.__file__).read_bytes()).hexdigest(),
        scope='Engineering equality on three existing train images, CPU imgsz960; no detector or tracking quality estimate')
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,help='New output directory inside the project; defaults to a unique name')
    main(parser.parse_args().output_dir)
