import json
from pathlib import Path
import numpy as np
import pytest
from tracking.history_cost_replay import replay_cost


def test_saved_source_costs_are_exact_without_tolerance():
    root=Path(__file__).resolve().parents[1]
    folder=root/'runs/continuation_20261028/history_tracker_pilot/videos/b1d10d08-743fd86c/events'
    assert len(list(folder.glob('*.json')))==32
    count=0
    for file in folder.glob('*.json'):
        alpha=0. if file.stem in ['native','zero'] else .2
        for event in json.loads(file.read_text(encoding='utf-8')):
            assert float(replay_cost(event['original'],event['score'],alpha,np.float32))==event['corrected']
            count+=1
    assert count==20891


def test_invalid_record_is_not_silently_repaired():
    with pytest.raises(ValueError):replay_cost(.2,float('nan'),.2,np.float32)
