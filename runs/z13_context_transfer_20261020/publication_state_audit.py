"""Check saved artifacts and process completion; do not rerun inference or statistics."""
import json
from pathlib import Path
import psutil

RUN = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    protocol = read(RUN / 'protocol.json')
    end = read(RUN / 'controller_result.json')
    assert end['worker_ended'] and end['exit_code'] == 1 and not end['hard_timeout']
    launch = read(RUN / 'controller.json')
    try:
        proc = psutil.Process(launch['pid'])
        assert abs(proc.create_time() - launch['create_time']) > .01 or not proc.is_running()
    except psutil.NoSuchProcess:
        pass
    result = dict(controller_result=end, own_worker_ended=True, inference_repeated=False,
                  statistical_analysis_run=False, efficiency_verdict=None, arms={})
    for arm in protocol['arms']:
        predictions = read(RUN / arm / 'predictions.json')
        done = read(RUN / arm / 'done.json')
        assert done['videos'] == len(protocol['videos']) == 35
        combined = []
        for video in protocol['videos']:
            saved = read(RUN / arm / 'videos' / video / 'directional_rows.json')
            reproduction = saved['reproduction']
            assert reproduction['max_center_difference'] < .5
            assert reproduction['max_peak_difference'] < .001
            assert reproduction['known_outcomes_match'] is True
            combined.extend(saved['rows'])
        assert combined == predictions['original']
        assert len(combined) == done['rows']
        assert all(len(rows) == done['rows'] for rows in predictions.values())
        result['arms'][arm] = dict(videos=done['videos'], rows=done['rows'],
                                  every_saved_reproduction_passed=True,
                                  aggregate_matches_saved_video_rows=True)
    assert not (RUN / 'report.json').exists()
    result['failure'] = 'KeyError useful in final selection aggregation; all predictions retained'
    (RUN / 'publication_state_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
