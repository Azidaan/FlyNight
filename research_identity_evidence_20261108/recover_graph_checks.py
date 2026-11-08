"""Finish interrupted record checks only; never restart scientific stages."""
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_identity_assignment_run_20261108.common import read, write, sha, datetime, MSK

RUN = ROOT / 'runs/identity_evidence_20261108/graph_update/recovery1'


def main():
    sys.path.insert(0, str(ROOT / 'runs/flyvis_repairs_20261106/unification_20261107'))
    from system_memory_guard import guard_entrypoint
    observation = guard_entrypoint(timeout=900)
    protocol = read(RUN / 'recovery_protocol.json')
    for name, digest in protocol['sources'].items():
        assert sha(ROOT / name) == digest, name
    original = read(ROOT / 'runs/identity_evidence_20261108/window_close.json')
    assert original['all_owned_ended'] and not original['graph_applied']
    assert not (RUN.parent / 'result.json').exists()
    import research_identity_evidence_20261108.rebuild_graph as implementation
    implementation.RUN = RUN
    implementation.main(observation)
    assert read(RUN / 'result.json')['passed']
    for name, digest in protocol['sources'].items():
        assert sha(ROOT / name) == digest, name
    write(RUN / 'queue_result.json', dict(
        passed=True, completed=datetime.now(MSK).isoformat(),
        original_interruption_preserved=True, no_scientific_stages=True,
        result_sha256=sha(RUN / 'result.json'),
        recovery_protocol_sha256=sha(RUN / 'recovery_protocol.json')),
        new=True)


if __name__ == '__main__':
    main()
