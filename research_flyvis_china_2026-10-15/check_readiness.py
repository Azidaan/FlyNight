"""Read-only CPU schema audit of the already authorized training split.

No torch, video decoding, model evaluation, dev or heldout access.
This is a data-readiness check, not an experiment on module effectiveness.
"""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'runs/z13_distribution_quality_20261015/protocol.json'


def array_shape(archive, name):
    with archive.open(name + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f'Unsupported NPY version: {version}')
    return list(shape), str(dtype)


def main():
    out = HERE / 'readiness.json'
    if out.exists():
        print(out.read_text(encoding='utf-8'))
        return
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    fit = protocol['fit']
    assert len(fit) == len(set(fit)) == 104
    assert not set(fit) & set(protocol['tau'])
    rows = []
    for video in fit:
        path = (ROOT / 'runs/z13_data' / (video + '.npz')).resolve()
        path.relative_to(ROOT.resolve())
        with zipfile.ZipFile(path) as archive:
            names = {n[:-4] for n in archive.namelist() if n.endswith('.npy')}
            shape, dtype = array_shape(archive, 'fr_hex')
        required = {'ep_offset', 'ep_identity', 'ep_id_dist', 'ex_mask', 'fr_hex',
                    'fr_frame', 'fr_crop', 'lab_ep', 'lab_pos', 'lab_kind', 'lab_gtbox'}
        assert required <= names, (video, required - names)
        assert len(shape) == 2 and shape[1] == 721 and dtype == 'float16'
        with np.load(path, allow_pickle=False) as z:
            offsets, frame = z['ep_offset'], z['fr_frame']
            crop, boxes, kind = z['fr_crop'], z['lab_gtbox'], z['lab_kind']
            ep, pos = z['lab_ep'], z['lab_pos']
            assert offsets[0] == 0 and offsets[-1] == shape[0] == len(frame)
            assert np.all(np.diff(offsets) >= 4)
            assert np.all((ep >= 0) & (ep < len(offsets) - 1))
            assert np.all((pos >= 0) & (pos + 4 < np.diff(offsets)[ep]))
            assert np.isin(kind, [-1, 0, 1]).all()
            assert np.isfinite(crop).all() and np.all(crop[:, 2:] > crop[:, :2])
            positive = boxes[kind == 1]
            assert np.isfinite(positive).all() and np.all(positive[:, 2:] > positive[:, :2])
            assert np.all(z['ep_identity'][ep[kind == 1]] != '')
            diffs = [np.diff(frame[a + 4:b]) for a, b in zip(offsets[:-1], offsets[1:])]
            gaps = sum(int(np.count_nonzero(d != 1)) for d in diffs)
            rows.append({'video': video, 'episodes': len(offsets)-1,
                         'positive_rows': int((kind == 1).sum()),
                         'unknown_rows': int((kind == -1).sum()),
                         'nonconsecutive_search_steps': gaps})
    report = {'passed': True, 'type': 'CPU data schema readiness only',
              'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'videos': len(rows), 'scope': 'fit104 only; no calibration/dev/heldout',
              'rows': rows,
              'ready': ['box targets on known positive annotated frames',
                        'eye input sequences and crop coordinates',
                        'known/unknown identity distinction'],
              'not_established': ['object segmentation masks', 'pure background labels',
                                  'feature availability outside original crops',
                                  'module quality, runtime or HOTA improvement'],
              'gpu_used': False}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
