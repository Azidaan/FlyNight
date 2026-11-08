"""Post-run provenance check; no tracker, images, neural models or metrics."""
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_singleton_working_domain_20261108.common import RUN, read, write, phase, sha


def main():
    sys.path.insert(0, str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
    from system_memory_guard import guard_entrypoint
    guard_entrypoint(timeout=180)
    import numpy as np
    p = phase()
    protocol = read(RUN/'origin_verification_protocol.json')
    assert sha(Path(__file__)) == protocol['source_sha256']
    checked_rows = checked_candidates = checked_confirmations = 0
    for video, entry in p['videos'].items():
        with np.load(ROOT/entry['raw'], allow_pickle=False) as z:
            frames, conf, classes, boxes = z['frame_idx'], z['conf'], z['cls'], z['xyxy']
        with np.load(RUN/'native'/(video+'.npz'), allow_pickle=False) as z:
            native = z['outputs']
        group = {}
        order = np.argsort(frames, kind='stable')
        frame_values = frames[order]
        for frame in np.unique(native[:, 0]).astype(int):
            lo, hi = np.searchsorted(frame_values, [frame, frame+1])
            all_ids = order[lo:hi]
            high = all_ids[conf[all_ids] >= .25]
            low = all_ids[(conf[all_ids] > .1) & (conf[all_ids] < .25)]
            group[frame] = high
            for row in native[native[:, 0] == frame]:
                local = int(row[8]); assert row[8] == local
                domain = high if row[6] >= .25 else low
                assert 0 <= local < len(domain), (video, frame, local)
                raw_id = domain[local]
                assert np.float32(conf[raw_id]) == row[6]
                assert int(classes[raw_id]) == int(row[7])
                checked_rows += 1
        data = read(RUN/'native'/(video+'.json'))
        for event in data['events']:
            f = event['frame']; high = group[f]
            public = native[native[:, 0] == f]
            for row in event['rows']:
                for candidate in row['candidates']:
                    raw_id = int(candidate['raw_id']); column = int(candidate['column'])
                    assert int(high[column]) == raw_id
                    assert np.array_equal(boxes[raw_id], np.asarray(candidate['box'], boxes.dtype))
                    assert float(conf[raw_id]) == candidate['confidence']
                    assert int(classes[raw_id]) == row['row_data']['cls']
                    checked_candidates += 1
                    if candidate['new_tid'] is not None:
                        current = public[public[:, 5] == candidate['new_tid']]
                        assert len(current) == 1 and int(current[0, 8]) == column
                        checked_confirmations += 1
    phase()
    write(RUN/'origin_verification.json', dict(passed=True, videos=20,
        native_index_rows=checked_rows, candidate_raw_id_checks=checked_candidates,
        confirmed_candidate_index_checks=checked_confirmations,
        source_sha256=sha(Path(__file__)), no_new_tracker_or_model_calls=True), new=True)


if __name__ == '__main__':
    main()
