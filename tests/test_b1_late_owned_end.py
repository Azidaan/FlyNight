import json
import pytest
from scripts import recovery_queue_late_exit_20261028 as recovery


def test_late_end_proof_preserves_failed_original_controller(tmp_path, monkeypatch):
    value = dict(owned={'123': 4.5}, exit_code=15, all_owned_ended=False, remaining=[123])
    source = tmp_path / 'controller_result.json'
    source.write_text(json.dumps(value), encoding='utf-8'); before = source.read_bytes()
    monkeypatch.setattr(recovery.service, 'matching', lambda *args: None)
    proof = recovery.verify_late_end(tmp_path, value)
    assert proof['actual_all_owned_ended'] and not proof['original_all_owned_ended']
    assert proof['original_exit_code'] == 15 and source.read_bytes() == before


def test_live_owned_process_cannot_be_adopted_as_finished(tmp_path, monkeypatch):
    value = dict(owned={'123': 4.5}, exit_code=15, all_owned_ended=False)
    monkeypatch.setattr(recovery.service, 'matching', lambda *args: object())
    with pytest.raises(AssertionError): recovery.verify_late_end(tmp_path, value)
    assert not (tmp_path / 'late_owned_end_verified.json').exists()
