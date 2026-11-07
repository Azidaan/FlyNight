"""Те же CPU-проверки20видео, адрес восстановленного исполнения."""
from pathlib import Path
import sys, hashlib
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read


if __name__ == '__main__':
    import scripts.audit_b1_tracker_20261028 as original
    code = (ROOT / 'scripts/audit_b1_tracker_20261028.py').read_text(encoding='utf-8')
    old = "source = RUN / 'b1_tracker'; folder = RUN / 'b1_tracker_audit'"
    assert code.count(old) == 1
    code = code.replace(old, "source = RUN / 'b1_tracker/continuation_recovery1'; folder = RUN / 'b1_tracker_audit/continuation_recovery1'")
    assert hashlib.sha256(code.encode()).hexdigest() == read(RUN / 'b1_tracker/continuation_recovery1/protocol.json')['executed_audit_sha256']
    ns = {**vars(original), '__name__': 'b1_continuation_recovery_auditor'}
    exec(compile(code, str(Path(original.__file__)), 'exec'), ns)
    ns['run']()
