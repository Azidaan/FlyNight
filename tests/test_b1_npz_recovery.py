import numpy as np
from scripts.b1_npz_recovery_20261026 import CachedArchive


def test_lazy_archive_preserves_arrays_and_reads_each_key_once(tmp_path):
    path = tmp_path / 'sample.npz'
    source = np.arange(12).reshape(3, 4)
    np.savez_compressed(path, value=source, unused=np.zeros(10))
    with CachedArchive(np.load(path)) as cached:
        assert cached.files == ['value', 'unused']
        assert np.array_equal(cached['value'], source)
        assert cached['value'] is cached['value']
        assert set(cached.values) == {'value'}
