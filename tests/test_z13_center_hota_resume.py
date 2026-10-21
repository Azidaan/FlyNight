import pytest
from pathlib import Path
from scripts.z13_center_hota_resume_20261022 import require_unchanged_markers,controller_source,verify_stitched_frames


def test_resume_rejects_wrong_identity_or_changed_job_instead_of_repeating_it():
    jobs=[dict(id='A',stage='signatures'),dict(id='B',stage='stitch')];identity={'hash':'frozen'}
    marker=dict(job=jobs[0],identity=identity,exit_code=0)
    assert require_unchanged_markers(jobs,[marker],identity)==jobs[1:]
    with pytest.raises(AssertionError):require_unchanged_markers(jobs,[dict(marker,identity={'hash':'other'})],identity)
    with pytest.raises(AssertionError):require_unchanged_markers(jobs,[dict(marker,job=dict(id='A',stage='eval'))],identity)
    with pytest.raises(AssertionError):require_unchanged_markers(jobs,[marker,marker],identity)


def test_resume_retains_completion_identity_and_changes_only_owned_execution_paths():
    source=(Path(__file__).resolve().parents[1]/'scripts/z13_center_hota_controller_20261021.py').read_text(encoding='utf-8');out=controller_source(source)
    assert "RUN/'completed_jobs'" in out and "RUN/'execution_identity.json'" in out
    assert "RUN/'resume1_controller_result.json'" in out and "RUN/'full_controller_result.json'" not in out
    assert "read(RUN/'resume1_authorization.json')['hard_stop']" in out and "'--attempt','resume1'" in out
    assert "assert read(RUN/'jobs.json')==jobs" in out
    compile(out,'verified_resumed_controller','exec')


def test_stitch_verification_uses_reference_frames_not_raw_video_frame_count():
    frames=[dict(name='a.jpg',frameIndex=0),dict(name='b.jpg',frameIndex=1)]
    matches=[dict(reference_name='a.jpg',reference_index=0,video_frame_index=0),dict(reference_name='b.jpg',reference_index=1,video_frame_index=6)]
    verify_stitched_frames(frames,matches)
    with pytest.raises(AssertionError):verify_stitched_frames(frames[:1],matches)
    with pytest.raises(AssertionError):verify_stitched_frames(frames[::-1],matches)
