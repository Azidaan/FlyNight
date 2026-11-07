import json
from pathlib import Path
import pytest
from tracking.flyvis_state_subgroups import GROUPS, validate_partition

def test_partition_matches_connectome_without_readout_cells():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root/"tools/flyvis-runtime/flyvis/connectome/fib25-fib19_v2.2.json").read_text())
    parent = [x["name"] for x in data["nodes"] if x["name"] not in data["input_units"]+data["output_units"]]
    result = validate_partition(parent)
    assert len(GROUPS) == 8
    assert sorted(sum((result[k] for k in GROUPS), [])) == sorted(result["parent"])

def test_wrong_parent_is_rejected():
    with pytest.raises(ValueError):
        validate_partition(sum(GROUPS.values(), [])[:-1]+["T4a"])
