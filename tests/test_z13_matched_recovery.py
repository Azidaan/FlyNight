from scripts import z13_matched_pipeline_recovery_20261024 as R


def test_recovery_prices_only_unfinished_scope_and_keeps_budget():
    from datetime import datetime
    old=R.read(R.OLD/'protocol.json');controller=R.read(R.OLD/'controller.json')
    p=R.read(R.REC/'protocol.json');s=p['service_recovery'];price=R.read(R.OLD/'price.json')
    assert s['corrected_remaining_videos']==6
    assert s['conservative_remaining_seconds']==2*price['first_complete_video_seconds']*6+600
    assert s['conservative_remaining_seconds']<s['remaining_original_seconds']
    assert p['budget_seconds']==s['remaining_original_seconds']<old['budget_seconds']
    assert p['save_at']==controller['save_at'] and p['hard_stop']==controller['hard_stop']
    assert (datetime.fromisoformat(p['save_at'])-datetime.fromisoformat(s['original_started'])).total_seconds()<=old['budget_seconds']
    assert all(p['sha256'][n]==h for n,h in old['sha256'].items())


def test_completed_first_video_receipts_retained_without_rerun():
    completed=list((R.OLD/'jobs').glob('*.json'));assert len(completed)==7
    for old in completed:
        assert R.read(R.REC/'jobs'/old.name)==R.read(old)
        for name,h in R.read(old)['outputs'].items():assert R.sha(R.ROOT/name)==h
