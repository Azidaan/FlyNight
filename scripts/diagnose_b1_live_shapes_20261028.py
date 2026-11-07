"""Сохранить формы реального отказа головы, не подменяя отказ успехом."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, now


def main():
    from scripts import run_b1_tracker_20261028 as original
    import tracking.b1_tracker_runtime as runtime
    from tracking.b1_deterministic_runtime import DeterministicResidualEngine
    original.FOLDER = RUN / 'b1_live_shape_diagnostic'
    runtime.ResidualEngine = DeterministicResidualEngine
    # Успешная законченная проверка той же реализации не повторяется.
    assert read(RUN / 'b1_tracker/recovery1/incremental_regression.json')['passed']
    original.regression = lambda *args: None
    try:
        original.run()
    except Exception as error:
        trace = error.__traceback__; frames = []
        while trace is not None:
            f = trace.tb_frame
            if 'b1_' in f.f_code.co_filename or 'temporal_residual' in f.f_code.co_filename:
                shapes = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in f.f_locals.items() if hasattr(v, 'shape')}
                frames.append(dict(file=f.f_code.co_filename, function=f.f_code.co_name, line=trace.tb_lineno, shapes=shapes,
                    arm=getattr(f.f_locals.get('self'), 'arm', None)))
            trace = trace.tb_next
        write(original.FOLDER / 'failure_shapes.json', dict(error=repr(error), frames=frames, updated=now()))
        raise


if __name__ == '__main__': main()
