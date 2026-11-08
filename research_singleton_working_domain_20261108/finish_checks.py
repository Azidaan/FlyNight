"""Bookkeeping checks after the bounded scientific stage has ended."""
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_singleton_working_domain_20261108.common import RUN, phase, read, write, sha, datetime, MSK


def main():
    sys.path.insert(0, str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
    from system_memory_guard import guard_entrypoint
    observation = guard_entrypoint(timeout=900)
    phase()
    assert read(RUN/'queue_result.json')['passed']
    assert read(RUN/'recovery2/deadline_result.json')['all_owned_ended']
    import research_identity_evidence_20261108.rebuild_graph as implementation
    implementation.RUN = RUN/'graph_update'
    implementation.main(observation)
    phase()
    write(RUN/'finish_checks.json', dict(passed=True, scientific_calculations_repeated=False,
        graph_result_sha256=sha(RUN/'graph_update/result.json'),
        finished=datetime.now(MSK).isoformat()), new=True)


if __name__ == '__main__':
    main()
