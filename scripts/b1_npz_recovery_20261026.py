"""Служебное продолжение Б1: кэш ключей NPZ, исходный научный код неизменен."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '3'
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import b1_real_inputs_20261026 as worker


class CachedArchive:
    def __init__(self, archive):
        self.archive = archive
        self.files = archive.files
        self.values = {}

    def __getitem__(self, key):
        if key not in self.values:
            self.values[key] = self.archive[key]
        return self.values[key]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.archive.close()
        self.values.clear()


def install():
    original = np.load
    def load(*args, **kwargs):
        result = original(*args, **kwargs)
        return CachedArchive(result) if isinstance(result, np.lib.npyio.NpzFile) else result
    np.load = load
    return original


def main():
    protocol = worker.read(worker.RUN / 'protocol.json')
    assert datetime.now(worker.MSK) < datetime.fromisoformat(protocol['deadline'])
    assert not (worker.RUN / 'report.json').exists()
    assert (worker.RUN / 'service_recovery1.json').exists()
    import tracking.b1_real_inputs as inputs
    prepared = {}
    for path in (worker.RUN / 'inputs').glob('*.json'):
        prepared.update({row['uid']: row for row in worker.read(path)['rows']})
    bind_original = inputs.bind_observation
    def bind(row, archive, choices):
        cached = prepared.get(int(row['uid']))
        if cached is None:
            return bind_original(row, archive, choices)
        assert cached['choices'] == list(choices) and cached['current_array_index'] == row['current_array_index']
        return cached
    inputs.bind_observation = bind
    original_write = worker.write
    def write(name, value):
        path = worker.RUN / name
        if name.startswith('inputs/') and path.exists():
            assert worker.read(path) == value
            return
        original_write(name, value)
    worker.write = write
    install()
    try:
        worker.run()
    except Exception as error:
        worker.write('recovery1_failure.json', dict(error=repr(error), updated=datetime.now(worker.MSK).isoformat()))
        raise


if __name__ == '__main__':
    main()
