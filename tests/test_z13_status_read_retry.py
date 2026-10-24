"""A transient UI/status lock must not change scientific data or freshness."""
from unittest.mock import patch
import pytest
from scripts import z13_val20_native_masa_recovery2_20261024 as R


def test_status_retry_retains_original_timestamp():
    old={'updated':'2026-10-24T09:51:59+03:00','done':1200}
    with patch.object(R,'read',side_effect=[PermissionError(),FileNotFoundError(),old]) as reader,patch.object(R.time,'sleep') as sleep:
        assert R.status_read('status.json') is old
        assert reader.call_count==3
        assert [c.args for c in sleep.call_args_list]==[(.05,),(.05,)]


def test_status_retry_is_bounded_and_does_not_hide_other_errors():
    with patch.object(R,'read',side_effect=PermissionError()) as reader,patch.object(R.time,'sleep') as sleep:
        with pytest.raises(PermissionError):R.status_read('status.json')
        assert reader.call_count==40 and sleep.call_count==39
    with patch.object(R,'read',side_effect=ValueError('invalid JSON')) as reader,patch.object(R.time,'sleep') as sleep:
        with pytest.raises(ValueError):R.status_read('status.json')
        assert reader.call_count==1 and sleep.call_count==0
