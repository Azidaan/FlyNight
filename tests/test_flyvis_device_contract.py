"""Устройство и корни уже импортированного FlyVis меняются только явно."""
import os
import sys
from types import ModuleType

import pytest
import torch
from tracking import flyvis_adapter as adapter


def installed_fake(tmp_path, monkeypatch):
    runtime, data = tmp_path / "runtime", tmp_path / "data"
    (runtime / "flyvis").mkdir(parents=True)
    (data / "results" / "flow").mkdir(parents=True)
    modules = []
    for name in ("flyvis", "flyvis.network.initialization", "flyvis.task.decoder"):
        module = ModuleType(name)
        module.device = torch.device("cuda")
        monkeypatch.setitem(sys.modules, name, module)
        modules.append(module)
    modules[0].__file__ = str(runtime / "flyvis" / "__init__.py")
    modules[0].root_dir = data
    monkeypatch.setattr(adapter, "_patch_datamate_hdf5_writer", lambda: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("FLYVIS_ROOT_DIR", "unchanged-before-call")
    return runtime, data, modules


def test_explicit_cpu_updates_rng_decoder_and_default(tmp_path, monkeypatch):
    runtime, data, modules = installed_fake(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(torch, "set_default_device", calls.append)
    assert adapter.activate_flyvis(runtime, data, device="cpu") is modules[0]
    assert all(m.device == torch.device("cpu") for m in modules)
    assert calls == [torch.device("cpu")]


def test_none_preserves_existing_device_and_default(tmp_path, monkeypatch):
    runtime, data, modules = installed_fake(tmp_path, monkeypatch)
    monkeypatch.setattr(torch, "set_default_device", lambda device: pytest.fail("Unexpected device reset"))
    adapter.activate_flyvis(runtime, data)
    assert all(m.device == torch.device("cuda") for m in modules)


@pytest.mark.parametrize("changed", ["runtime", "data"])
def test_different_root_rejected_before_environment_mutation(tmp_path, monkeypatch, changed):
    runtime, data, _ = installed_fake(tmp_path, monkeypatch)
    if changed == "runtime":
        runtime = tmp_path / "other_runtime"
        (runtime / "flyvis").mkdir(parents=True)
    else:
        data = tmp_path / "other_data"
        (data / "results" / "flow").mkdir(parents=True)
    before_path = list(sys.path)
    with pytest.raises(RuntimeError, match="другого runtime/data_dir"):
        adapter.activate_flyvis(runtime, data)
    assert os.environ["FLYVIS_ROOT_DIR"] == "unchanged-before-call"
    assert sys.path == before_path


@pytest.mark.parametrize("device", ["meta", "mps"])
def test_unsupported_device_rejected(tmp_path, monkeypatch, device):
    runtime, data, modules = installed_fake(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="cpu или cuda"):
        adapter.activate_flyvis(runtime, data, device=device)
    assert all(m.device == torch.device("cuda") for m in modules)


def test_unavailable_cuda_rejected(tmp_path, monkeypatch):
    runtime, data, _ = installed_fake(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="недоступна"):
        adapter.activate_flyvis(runtime, data, device="cuda")


def test_first_import_prioritizes_requested_runtime_even_if_already_on_path(tmp_path, monkeypatch):
    runtime, data, _ = installed_fake(tmp_path, monkeypatch)
    monkeypatch.delitem(sys.modules, "flyvis")
    wrong = tmp_path / "other"
    (wrong / "flyvis").mkdir(parents=True)
    (wrong / "flyvis" / "__init__.py").write_text("raise AssertionError('Wrong FlyVis runtime')", encoding="utf-8")
    (runtime / "flyvis" / "__init__.py").write_text("from pathlib import Path\nimport os\nroot_dir=Path(os.environ['FLYVIS_ROOT_DIR'])\n", encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(wrong), *sys.path, str(runtime)])
    module = adapter.activate_flyvis(runtime, data)
    assert str(runtime / "flyvis") in module.__file__
    assert sys.path[0] == str(runtime)
