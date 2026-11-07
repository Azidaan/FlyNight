import ast
from scripts.run_b1_tracker_continuation1_20261028 import continuation_source
from scripts.audit_b1_tracker_continuation1_20261028 import audit_source


def test_completed_video_is_skipped_before_inference_and_cost_gate():
    source = continuation_source()
    ast.parse(source)
    skip = source.index("if v in seed['completed_videos']:")
    first_read = source.index("cache = load_detection_cache")
    assert skip < first_read
    assert source.index('continue', skip) < first_read
    assert "per = seed['per_video']" in source
    assert "assert sha(ROOT / path) == digest" in source
    assert source.count('if index == 0:') == 1
    assert "assert value == expected[ref]" in source
    assert "zero residual outputs" in source


def test_audit_keeps_full_raw_head_and_hota_checks():
    source = audit_source(); ast.parse(source)
    assert "source = RUN / 'b1_tracker/continuation1'" in source
    assert 'assert tracker.cursor == len(events)' in source
    assert 'metrics == r' in source
    assert 'len(p[\'sources\']) if complete else 1' in source
