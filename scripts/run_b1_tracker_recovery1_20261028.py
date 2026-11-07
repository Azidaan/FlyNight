"""Отдельное исполнение той же Б1 с детерминированной суммой, старый срок."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read, matching, now, write


def main():
    from scripts import run_b1_tracker_20261028 as original
    import tracking.b1_tracker_runtime as runtime
    from tracking.b1_deterministic_runtime import DeterministicResidualEngine
    for folder in ['b1_tracker', 'b1_tracker_audit', 'b1_incremental_diagnostic']:
        c = read(RUN / folder / 'controller_result.json')
        assert c['all_owned_ended'] and not any(matching(k, v) for k, v in c['owned'].items())
    original.FOLDER = RUN / 'b1_tracker/recovery1'
    runtime.ResidualEngine = DeterministicResidualEngine
    write(original.FOLDER / 'execution_mode.json', dict(flyvis_reduction='torch deterministic algorithms scoped to init/step',
        cublas_workspace=':4096:8', masa_fp32_unchanged=True, thresholds_and_weights_unchanged=True,
        original_exact_assertions_unchanged=True, updated=now()))
    original.run()


if __name__ == '__main__':
    main()
