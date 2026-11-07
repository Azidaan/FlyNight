"""Исходные проверки CPU, адреса второй версии восстановления."""
from pathlib import Path
import sys, hashlib
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read


if __name__ == '__main__':
    import scripts.audit_b1_tracker_20261028 as original
    source = Path(original.__file__).read_text(encoding='utf-8')
    old = "source = RUN / 'b1_tracker'; folder = RUN / 'b1_tracker_audit'"
    assert source.count(old) == 1
    source = source.replace(old, "source = RUN / 'b1_tracker/recovery2'; folder = RUN / 'b1_tracker_audit/recovery2'")
    assert hashlib.sha256(source.encode()).hexdigest() == read(RUN / 'b1_tracker/recovery2/protocol.json')['executed_audit_sha256']
    namespace = {**vars(original), '__name__': 'b1_recovery2_auditor'}
    exec(compile(source, str(Path(original.__file__)), 'exec'), namespace)
    namespace['run']()
