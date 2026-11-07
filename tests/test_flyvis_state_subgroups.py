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


def test_completion_hashes_survive_the_chain_finishing(tmp_path):
    from scripts.finalize_flyvis_state_subgroups_20261107 import closed_artifact_hashes

    for name in ("record_tests.log", "independent_audit.json", "finalize.log",
                 "chain.stdout.log", "chain.stderr.log", "postprocess.stage.json"):
        (tmp_path / name).write_text("before", encoding="utf-8")
    saved = closed_artifact_hashes(tmp_path)
    assert set(saved) == {"record_tests.log", "independent_audit.json"}
    for name in ("finalize.log", "chain.stdout.log", "chain.stderr.log", "postprocess.stage.json",
                 "completion_verified.json", "finalize.ended.json", "chain_result.json", "chain_failure.json"):
        (tmp_path / name).write_text("after", encoding="utf-8")
    assert closed_artifact_hashes(tmp_path) == saved
    (tmp_path / "independent_audit.json").write_text("changed", encoding="utf-8")
    assert closed_artifact_hashes(tmp_path) != saved
