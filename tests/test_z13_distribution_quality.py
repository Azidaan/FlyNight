import copy
import json
from pathlib import Path
import pytest
from scripts.z13_distribution_quality import validate_jobs


def protocol():
    root = Path(__file__).resolve().parents[1]
    p = json.loads((root / 'runs/z13_spatial_v2_20261013/protocol.json').read_text(encoding='utf-8'))
    p['jobs'] = [j for j in p['jobs'] if j['name'] != 'full']
    return p


def test_original_video_disjoint_folds():
    validate_jobs(protocol())


def test_rejects_prediction_video_in_fit():
    p = copy.deepcopy(protocol())
    p['jobs'][0]['fit'][0] = p['jobs'][0]['predict'][0]
    with pytest.raises(AssertionError):
        validate_jobs(p)


def test_rejects_calibration_video_in_localization_fit():
    p = copy.deepcopy(protocol())
    p['fit'][0] = p['tau'][0]
    with pytest.raises(AssertionError):
        validate_jobs(p)


def test_rejects_duplicate_fold_predictions():
    p = copy.deepcopy(protocol())
    p['jobs'][1]['fit'] = p['jobs'][0]['fit'][:]
    p['jobs'][1]['predict'] = p['jobs'][0]['predict'][:]
    with pytest.raises(AssertionError):
        validate_jobs(p)
