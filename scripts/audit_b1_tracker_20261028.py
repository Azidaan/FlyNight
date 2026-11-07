"""Независимый CPU-повтор назначений, решений и HOTA полного Б1."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
from types import SimpleNamespace
import sys, time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, sha, matching, now


def run():
    import numpy as np, torch
    from tracking.b1_tracker_runtime import B1Tracker, FrozenBase, geometry_inputs
    from tracking.z13_readout import HexGeometry
    from ultralytics.trackers.utils import matching as assignment_matching
    from tracking.b1_learning import categorical_cost
    from tracking.b1_appearance import gate_features
    from tracking.flyvis_temporal_residual import TemporalResidualAssociation
    from tracking.offline_association import _DetSlice
    from scripts.run_bdd_video_rate_experiment import load_detection_cache, detections_for_frame, scalabel_frame
    from evaluation.hota_eval import evaluate_hota
    source = RUN / 'b1_tracker'; folder = RUN / 'b1_tracker_audit'
    p = read(source / 'protocol.json'); controller = read(source / 'controller_result.json')
    assert controller['exit_code'] == 0 and controller['all_owned_ended'] and not controller['timed_out']
    assert not any(matching(pid, birth) for pid, birth in controller['owned'].items())
    for path, h in p['sha256'].items():
        assert sha(ROOT / path) == h, path
    torch.set_num_threads(2); torch.set_default_device('cpu')
    complete = (source / 'report.json').exists()
    r = read(source / ('report.json' if complete else 'cost_stop.json'))
    videos = [s for s in p['sources'] if (source / 'completed' / (s['video'] + '.json')).exists()]
    assert len(videos) == (len(p['sources']) if complete else 1)
    base = FrozenBase()
    with np.load(source / 'geometry.npz') as z:
        geom = HexGeometry(z['points'], int(z['eye_size']))
    heads = {}
    for arm in ['C', 'D', 'R1', 'B']:
        head = TemporalResidualAssociation().cpu().eval()
        head.load_state_dict(torch.load(ROOT / f'runs/b123_20261027/b1_training/{arm}/checkpoint.pt', map_location='cpu', weights_only=False)['head'])
        heads[arm] = head
    checked = 0; all_metrics = {}; began = time.monotonic()
    class Replay(B1Tracker):
        def __init__(self, args, events, name):
            super().__init__(args, disabled=True)
            self.saved, self.cursor, self.name = events, 0, name
        def get_dists(self, tracks, detections):
            original = super().get_dists(tracks, detections)
            if not self._first:
                return original
            answer = original.copy()
            assignment = np.full(len(tracks), -1, int)
            if len(tracks) and len(detections):
                pairs, _, _ = assignment_matching.linear_assignment(original, thresh=self.args.match_thresh)
                for i, j in pairs:
                    assignment[i] = j
            while self.cursor < len(self.saved) and self.saved[self.cursor]['frame'] == self.frame_id - 1:
                e = self.saved[self.cursor]
                candidates = [i for i, t in enumerate(tracks) if int(t.track_id) == e['track']]
                assert len(candidates) == 1
                i = candidates[0]
                assert np.array_equal(original[i], np.asarray(e['original'], dtype=original.dtype))
                assert np.array_equal(tracks[i].mean[:2], e['prediction']) and int(tracks[i].cls) == e['track_class']
                choices, masks, rebuilt_geometry = geometry_inputs(geom, e['crop'], e['prediction'], detections,
                    e['track_class'], original, i, assignment)
                assert choices == e['choices'] and np.array_equal(rebuilt_geometry, np.array(e['geometry'], np.float32))
                assert e['reference_frame'] < e['frame'] and (e['previous_query_frame'] is None or e['previous_query_frame'] < e['frame'])
                geometry = np.asarray(e['geometry'], np.float32); count = len(geometry); width = max(18, count)
                padded = np.zeros((width, 14), np.float32); padded[:count] = geometry
                mask = np.arange(width) < count
                prob = np.full(width, np.nan); prob[:count] = e['identity']
                features = np.array(e['base_features'], np.float32)
                if count > 1:
                    predicted = base.identity.predict_proba(features[1:])[:, 1]
                    assert np.array_equal(predicted, np.array(e['identity_base'])[1:])
                if e['pooled'] is not None:
                    arm = 'D' if self.name == 'zero' else self.name
                    head = heads[arm]
                    with torch.no_grad():
                        q = head.project(torch.tensor(e['pooled']['reference'], dtype=torch.float32)[None, None])
                        k = head.project(torch.tensor(e['pooled']['current'], dtype=torch.float32)[None])
                        q = q.expand_as(k)
                        delta = torch.tanh(head.output(head.compare(torch.cat([q, k, q * k, (q-k).abs()], -1)))).squeeze().reshape(-1).numpy()
                    delta[0] = 0
                    if self.name == 'zero':
                        delta[:] = 0
                    assert np.allclose(delta, e['delta'], rtol=0, atol=2e-6), (self.name, e['frame'], 'CPU head')
                base_p = np.array(e['identity_base'], np.float64)
                change = np.expm1(np.array(e['delta'], np.float64))
                updated_p = base_p + base_p * (1-base_p) * change / (1 + base_p * change)
                assert np.allclose(updated_p, e['identity'], atol=1e-12, rtol=0, equal_nan=True)
                expected_action, gate = base.decision(prob, padded, mask)
                assert expected_action == e['action'] and np.array_equal(gate[:count], e['gate'], equal_nan=True)
                out = categorical_cost(original[i], e['choices'], expected_action)
                assert np.array_equal(out, np.asarray(e['updated'], dtype=out.dtype))
                answer[i] = out
                self.cursor += 1
            return answer
    variants = ['baseline', 'appearance', 'zero', 'C', 'D', 'R1', 'B']
    for source_row in videos:
        v = source_row['video']; expected = read(ROOT / source_row['baseline'])
        cache = load_detection_cache(ROOT / source_row['detections'])
        anchors = {int(a['video_frame_index']): i for i, a in enumerate(read(ROOT / source_row['alignment'])['matches'])}
        all_metrics[v] = {}
        for name in variants:
            events = read(source / 'events' / name / (v + '.json'))
            tracker = Replay(SimpleNamespace(**p['config']), events, name)
            rows, raw = [], []
            for frame in range(int(cache['frame_count'])):
                b, c, k = detections_for_frame(cache, frame)
                out = np.asarray(tracker.update(_DetSlice(b, c, k)), np.float32).reshape(-1, 8)
                if len(out):
                    raw.append(np.column_stack([np.full(len(out), frame), out[:, :7]]).astype(np.float32))
                if frame in anchors:
                    ref = anchors[frame]; template = {k: expected[ref][k] for k in ['name', 'frameIndex']}
                    rows.append(scalabel_frame(v, template, out))
            assert tracker.cursor == len(events) and rows == read(source / name / (v + '.json'))
            with np.load(source / 'raw' / name / (v + '.npz')) as z:
                assert np.array_equal(np.concatenate(raw) if raw else np.zeros((0, 8), np.float32), z['outputs'])
            metrics = evaluate_hota(source / name, [v], split='val')
            all_metrics[v][name] = metrics
            if complete:
                assert metrics == r['per_video'][name][v]
            checked += len(events)
            write(folder / 'status.json', dict(state='running', updated=now(), video=v, variant=name, events_checked=checked), required=False)
        assert read(source / 'appearance' / (v + '.json')) == read(source / 'zero' / (v + '.json'))
    contrasts = {}
    if complete:
        rng = np.random.default_rng(281028)
        for name in ['baseline', 'appearance', 'C', 'R1', 'B']:
            delta = np.array([all_metrics[s['video']]['D']['HOTA'] - all_metrics[s['video']][name]['HOTA'] for s in videos])
            boot = delta[rng.integers(len(delta), size=(5000, len(delta)))].mean(1)
            assert abs(delta.mean() - r['paired']['D-' + name]['mean_per_video']) < 1e-10
            contrasts[name] = dict(mean=float(delta.mean()), ci95=np.quantile(boot, [.025, .975]).tolist())
    result = dict(passed=True, full_series=complete, videos=len(videos), events_checked=checked,
        all_raw_and_json_replayed=True, all_heads_replayed_on_cpu=True, native_and_zero_exact=True,
        metrics_independently_repeated=True, independent_pairs=contrasts, cuda_initialized=torch.cuda.is_initialized(),
        source_hashes_verified=True, report_sha256=sha(source / ('report.json' if complete else 'cost_stop.json')),
        seconds=time.monotonic() - began, updated=now())
    assert not result['cuda_initialized']
    write(source / 'independent_audit.json', result); write(folder / 'report.json', result)


if __name__ == '__main__':
    run()
