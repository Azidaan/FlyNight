import json
import pytest
from scripts import history_stage_budget_policy_20261029 as m


def setup_policy(tmp_path, monkeypatch):
    run=tmp_path/'run';run.mkdir()
    auth=run/'stage_budget_policy_20261029'/'authorization.json'
    auth.parent.mkdir()
    stop='2026-10-29T07:40:00+03:00'
    m.service.write(run/'authorization.json',dict(hard_stop=stop))
    m.service.write(auth,dict(human_message='Только бюджеты этапов; общую остановку в 07:40 оставить',
        stage_time_limits_cancelled=True,hard_stop=stop,save_at='2026-10-29T07:30:00+03:00'))
    monkeypatch.setattr(m,'RUN',run);monkeypatch.setattr(m,'POLICY',auth)
    return run,auth


def test_cancellation_does_not_allow_global_extension(tmp_path,monkeypatch):
    _,auth=setup_policy(tmp_path,monkeypatch)
    value=m.service.read(auth);value['hard_stop']='2026-10-30T07:40:00+03:00'
    m.service.write(auth,value)
    with pytest.raises(AssertionError):m.policy()


def test_completed_gpu_never_reentered(tmp_path,monkeypatch):
    run,auth=setup_policy(tmp_path,monkeypatch)
    source=run/'history_tracker_author_continuation1';source.mkdir()
    m.service.write(source/'controller_result.json',dict(all_owned_ended=True,owned={},exit_code=0))
    m.service.write(source/'report.json',dict(complete=True))
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'transformed',lambda *a:pytest.fail('GPU inference must not be entered'))
    m.execute('gpu')
    proof=m.service.read(auth.parent/'selected_gpu_result.json')
    assert proof['complete'] and proof['gpu_repeated'] is False


def test_non_time_failure_cannot_be_restarted_as_time_extension(tmp_path,monkeypatch):
    run,_=setup_policy(tmp_path,monkeypatch)
    source=run/'history_tracker_author_continuation1';source.mkdir()
    m.service.write(source/'controller_result.json',dict(all_owned_ended=True,owned={},exit_code=1,timed_out=False))
    (source/'stderr.log').write_text('AssertionError: raw output differs',encoding='utf-8')
    with pytest.raises(AssertionError,match='raw output differs'):m.execute('gpu')
