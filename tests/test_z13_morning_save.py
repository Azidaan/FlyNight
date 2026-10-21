import pytest
from scripts.z13_center_hota_morning_save_20261021 import ended


class Processes:
    def __init__(self,live):self.live=live
    def creation(self,pid):return self.live.get(pid)


def result():
    return dict(all_own_workers_ended=True,remaining_owned_processes={},known_owned_processes={'42':100})


def test_saved_empty_remainder_cannot_hide_a_live_original_worker():
    with pytest.raises(AssertionError):ended(result(),Processes({42:100}),dict(pid=7,birth=70))
    with pytest.raises(AssertionError):ended(result(),Processes({7:70}),dict(pid=7,birth=70))


def test_reused_pids_are_not_mistaken_for_original_owned_processes():
    ended(result(),Processes({42:101,7:71}),dict(pid=7,birth=70))
