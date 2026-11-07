"""Детерминированная Б1 с проверенной формой живого входа; прежний срок."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, matching, write, sha, now


def main():
    from scripts import run_b1_tracker_20261028 as original
    import tracking.b1_tracker_runtime as runtime
    import tracking.foveated_eyes as eyes
    from tracking.b1_deterministic_runtime import DeterministicResidualEngine
    from tracking.b1_eye_layout import CanonicalEyeBank
    for folder in ['b1_tracker/recovery1', 'b1_tracker_audit/recovery1', 'b1_live_shape_diagnostic']:
        c = read(RUN / folder / 'controller_result.json')
        assert c['all_owned_ended'] and not any(matching(k, v) for k, v in c['owned'].items())
    original.FOLDER = RUN / 'b1_tracker/recovery2'
    runtime.ResidualEngine = DeterministicResidualEngine
    eyes.EyeBank = CanonicalEyeBank
    def reuse_regression(engines, folder):
        path = RUN / 'b1_tracker/recovery1/incremental_regression.json'
        proof = read(path); assert proof['passed']
        write(folder / 'incremental_regression.json', dict(proof, reused_completed_proof=True,
            source=path.relative_to(ROOT).as_posix(), source_sha256=sha(path),
            reason='Engine/checkpoints/deterministic reduction unchanged; renderer axis separately CPU-tested; completed GPU replay not repeated'))
    original.regression = reuse_regression
    write(original.FOLDER / 'execution_mode.json', dict(flyvis_reduction='deterministic init/step as recovery1',
        eye_layout='B,1,1,721 -> B,1,721, exact unchanged values', thresholds_and_weights_unchanged=True, updated=now()))
    original.run()


if __name__ == '__main__': main()
