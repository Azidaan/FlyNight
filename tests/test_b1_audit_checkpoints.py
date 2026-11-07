from pathlib import Path
import pytest
from tracking.b1_audit_checkpoints import validate_checkpoint,input_manifest

ARMS=['baseline','appearance','zero','C','D','R1','B']

def test_changed_input_rejects_previously_checked_video(tmp_path):
    source=tmp_path/'s'
    files=[source/'protocol.json',source/'geometry.npz',source/'completed/v.json']
    for arm in ARMS:files.extend([source/arm/'v.json',source/'events'/arm/'v.json',source/'raw'/arm/'v.npz'])
    for p in files:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixed')
    manifest=input_manifest(tmp_path,source,'v',ARMS)
    saved=dict(passed=True,input_manifest=manifest,metrics={a:{} for a in ARMS},events_checked=7)
    assert validate_checkpoint(saved,manifest)==saved
    (source/'events/D/v.json').write_bytes(b'changed')
    with pytest.raises(AssertionError):validate_checkpoint(saved,input_manifest(tmp_path,source,'v',ARMS))

def test_partial_checkpoint_is_not_a_completed_audit():
    with pytest.raises(AssertionError):validate_checkpoint(dict(passed=False,input_manifest={},metrics={a:{} for a in ARMS},events_checked=100),{})

def test_checkpointed_audit_preserves_scientific_assertions():
    import ast
    from scripts.audit_b1_checkpointed_20261028 import audit_source
    code=audit_source();ast.parse(code)
    for check in ['assert tracker.cursor == len(events)','metrics == r','CPU head','assert np.array_equal(original[i]','assert np.array_equal(np.concatenate(raw)','assert not result[\'cuda_initialized\']']:
        assert check in code
    assert "if checkpoint.exists():" in code and "write(checkpoint,dict(passed=True" in code
