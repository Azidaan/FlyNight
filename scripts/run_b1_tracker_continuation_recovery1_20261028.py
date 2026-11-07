"""Убрать зависающий перенос индексов; прежняя серия и абсолютный срок."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import sys, hashlib, faulthandler
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, write, sha, matching, now


def renderer_parity(bank, folder):
    import numpy as np, torch
    sample = torch.arange(2 * bank.eye_size * bank.eye_size, device='cuda', dtype=torch.float32).reshape(2, 1, bank.eye_size, bank.eye_size)
    points = bank.cached_sampler.centers + np.array([bank.eye_size//2, bank.eye_size//2])
    expected = sample.cpu()[:, :, points[:, 0], points[:, 1]].reshape(2, 1, 1, -1)
    actual = bank.cached_sampler(sample).cpu()
    assert torch.equal(expected, actual)
    write(folder / 'renderer_parity.json', dict(passed=True, exact=True, sampled_values_unchanged=True,
        columns=len(points), shapes=[list(expected.shape), list(actual.shape)], updated=now()))


def main():
    from scripts import run_b1_tracker_20261028 as original
    from scripts.run_b1_tracker_continuation1_20261028 import continuation_source
    import tracking.b1_tracker_runtime as runtime
    import tracking.foveated_eyes as eyes
    from tracking.b1_deterministic_runtime import DeterministicResidualEngine
    from tracking.b1_cached_eye import CachedCanonicalEyeBank
    folder = RUN / 'b1_tracker/continuation_recovery1'; p = read(folder / 'protocol.json')
    for sub in ['b1_tracker/continuation1', 'b1_tracker_audit/continuation1']:
        c = read(RUN / sub / 'controller_result.json')
        if not c['all_owned_ended']:
            late = read(RUN / sub / 'late_owned_end_verified.json')
            assert late['passed'] and late['actual_all_owned_ended'] and late['original_controller_sha256'] == sha(RUN / sub / 'controller_result.json')
        assert not any(matching(k, v) for k, v in c['owned'].items())
    runtime.ResidualEngine = DeterministicResidualEngine
    class CheckedBank(CachedCanonicalEyeBank):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            renderer_parity(self, folder)
    eyes.EyeBank = CheckedBank
    def reuse_regression(engines, target):
        path = RUN / 'b1_tracker/recovery1/incremental_regression.json'
        proof = read(path); assert proof['passed']
        write(target / 'incremental_regression.json', dict(proof, reused_completed_proof=True,
            source=path.relative_to(ROOT).as_posix(), source_sha256=sha(path)))
    code = continuation_source().replace('if frame % 25 == 0:', 'if frame % 5 == 0:')
    assert hashlib.sha256(code.encode()).hexdigest() == p['executed_worker_sha256']
    ns = {**vars(original), '__name__': 'b1_continuation_recovery_worker'}
    exec(compile(code, str(ROOT / 'scripts/run_b1_tracker_20261028.py'), 'exec'), ns)
    def watched_write(path, data, required=True):
        if Path(path).name == 'status.json':
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(600, repeat=False, file=sys.stderr, exit=True)
        return write(path, data, required=required)
    ns.update(FOLDER=folder, regression=reuse_regression, write=watched_write)
    faulthandler.enable(file=sys.stderr)
    faulthandler.dump_traceback_later(600, repeat=False, file=sys.stderr, exit=True)
    try: ns['run']()
    finally: faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__': main()
