"""Finish interrupted bookkeeping only; original scientific worker and deadline retained."""
import inspect
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import z13_temporal_pair_readiness_20261025 as S
from scripts.z13_build_vault_preserving_edits import snapshot_local_edits
from scripts.z13_hota_deadline_guard import WindowsProcesses


def main():
    interruption = S.read(S.RUN/'prepare_interruption.json')
    assert WindowsProcesses().creation(interruption['pid']) != interruption['birth']
    assert not (S.RUN/'status.json').exists()
    p = S.read(S.RUN/'protocol.json')
    assert S.now() < S.datetime.fromisoformat(p['save_at'])
    protected = {path.relative_to(ROOT).as_posix(): digest for path, digest in snapshot_local_edits(ROOT).items()}
    protected['.gitignore'] = S.sha(ROOT/'.gitignore')
    S.write(S.RUN/'foreign_snapshot.json', protected)
    # Execute only the unexecuted registration tail of the original frozen prepare().
    source = inspect.getsource(S.prepare)
    tail = source[source.index('    catalog = read('):]
    namespace = {**vars(S), 'p': p, 'start': S.datetime.fromisoformat(p['created'])}
    exec(compile('def finish_prepare():\n'+tail, __file__, 'exec'), namespace)
    namespace['finish_prepare']()
    S.write(S.RUN/'prepare_recovery1.json', dict(at=S.now().isoformat(), original_protocol_sha256=S.sha(S.RUN/'protocol.json'),
        original_script_sha256=S.sha(Path(S.__file__)), recovery_script_sha256=S.sha(Path(__file__)),
        original_deadline_retained=p['hard_stop'], scientific_worker_unchanged=True,
        protected_paths=len(protected), interrupted_helper_ended=True))


if __name__ == '__main__':
    main()
