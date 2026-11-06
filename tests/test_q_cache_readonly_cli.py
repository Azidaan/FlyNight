"""Old cache reads are explicit; analysis cannot silently rebuild or relabel them."""
import importlib
import json
import sys

import numpy as np
import pytest

from tracking.cache_readonly import require_cached_videos


MODULES = ('run_tracking_three_modes', 'sweep_lqa_k',
           'q_aggregation_ablation', 'correlation_experiment')


def cache_arrays():
    return dict(xyxy=np.array([[1, 2, 8, 9], [3, 4, 9, 12]], np.float32),
                conf=np.array([.8, .9], np.float32), cls=np.zeros(2, np.int16),
                frame_idx=np.array([0, 1], np.int32),
                frame_names=np.array(['000.jpg', '001.jpg']),
                q_mean=np.array([.2, .7], np.float32),
                q_min=np.array([.1, .4], np.float32),
                q_median=np.array([.2, .8], np.float32))


def save_cache(root, arrays=None):
    folder = root / 'val'
    folder.mkdir(parents=True)
    np.savez_compressed(folder / 'video.npz', **(cache_arrays() if arrays is None else arrays))
    (root / 'meta.json').write_text(json.dumps({'model': 'historic.pt', 'imgsz': 640}))
    return {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}


def test_explicit_legacy_read_preserves_arrays_and_metadata(tmp_path):
    before = save_cache(tmp_path)
    assert require_cached_videos(tmp_path, 'val', ['video'], required=('q_mean',)) == tmp_path / 'val'
    assert {p: p.read_bytes() for p in before} == before


def test_readonly_requires_complete_selection_and_needed_quality(tmp_path):
    arrays = cache_arrays()
    del arrays['q_median']
    before = save_cache(tmp_path, arrays)
    with pytest.raises(FileNotFoundError, match='missing'):
        require_cached_videos(tmp_path, 'val', ['video', 'missing'])
    with pytest.raises(ValueError, match='q_median'):
        require_cached_videos(tmp_path, 'val', ['video'], required=('q_median',))
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize('bad', ['misaligned', 'nonfinite'])
def test_bad_quality_arrays_fail_before_analysis(tmp_path, bad):
    arrays = cache_arrays()
    arrays['q_mean'] = np.array([.1]) if bad == 'misaligned' else np.array([.1, np.nan])
    save_cache(tmp_path, arrays)
    with pytest.raises(ValueError, match='q_mean'):
        require_cached_videos(tmp_path, 'val', ['video'], required=('q_mean',))


@pytest.mark.parametrize('name', MODULES)
def test_cached_only_cli_needs_no_model_never_calls_builder_and_checks_inputs(tmp_path, monkeypatch, name):
    module = importlib.import_module('scripts.' + name)
    before = save_cache(tmp_path)
    monkeypatch.setattr(sys, 'argv', [name, '--cached-only', '--cache-dir', str(tmp_path)])
    monkeypatch.setattr(module, 'subset_videos', lambda *a: {'night': ['video']})
    def forbidden_builder(*a, **kw):
        raise AssertionError('Read-only analysis called the detector builder')
    monkeypatch.setattr(module, 'build_detection_cache_with_q', forbidden_builder)
    class CheckedInput(Exception):
        pass
    def stop_after_real_check(folder, split, videos, *, required):
        assert folder == tmp_path and split == 'val' and videos == ['video']
        assert set(required) == ({'q_mean', 'q_min', 'q_median'} if name == 'q_aggregation_ablation' else {'q_mean'})
        require_cached_videos(folder, split, videos, required=required)
        raise CheckedInput
    monkeypatch.setattr(module, 'require_cached_videos', stop_after_real_check)
    with pytest.raises(CheckedInput):
        module.main()
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize('name', MODULES)
def test_building_cli_requires_model_before_dataset_access(monkeypatch, name):
    module = importlib.import_module('scripts.' + name)
    monkeypatch.setattr(sys, 'argv', [name])
    def forbidden_data(*a):
        raise AssertionError('Dataset accessed before argument validation')
    monkeypatch.setattr(module, 'subset_videos', forbidden_data)
    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code == 2


def test_correlation_reads_selected_cache_directory(tmp_path):
    from scripts.correlation_experiment import collect_pairs
    save_cache(tmp_path)
    conf, quality, frames = collect_pairs('val', ['video'], 'q_mean', .85, cache_dir=tmp_path)
    np.testing.assert_array_equal(conf, np.array([.9], np.float32))
    np.testing.assert_array_equal(quality, np.array([.7], np.float32))
    assert frames == [('video', 1)]
