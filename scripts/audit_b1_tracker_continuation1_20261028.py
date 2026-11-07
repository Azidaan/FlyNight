"""Неизменные проверки всех20видео, включая объединение сохранённой части."""
from pathlib import Path
import sys, hashlib
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.recovery_research_service_20261027 import RUN, read


def audit_source():
    source = (ROOT / 'scripts/audit_b1_tracker_20261028.py').read_text(encoding='utf-8')
    old = "source = RUN / 'b1_tracker'; folder = RUN / 'b1_tracker_audit'"
    assert source.count(old) == 1
    return source.replace(old, "source = RUN / 'b1_tracker/continuation1'; folder = RUN / 'b1_tracker_audit/continuation1'")


if __name__ == '__main__':
    import scripts.audit_b1_tracker_20261028 as original
    code = audit_source()
    assert hashlib.sha256(code.encode()).hexdigest() == read(RUN / 'b1_tracker/continuation1/protocol.json')['executed_audit_sha256']
    ns = {**vars(original), '__name__': 'b1_continuation_auditor'}
    exec(compile(code, str(Path(original.__file__)), 'exec'), ns)
    ns['run']()
