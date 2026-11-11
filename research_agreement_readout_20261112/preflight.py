"""Sequential catalog checks under the established system-memory supervisor."""
from common import *
import contextlib

if __name__ == '__main__':
    guard()
    mode = sys.argv[1] if len(sys.argv) > 1 else 'before'
    p = verify()
    from scripts.validate_project_history import main as validate
    sys.argv = ['scripts/validate_project_history.py']
    with (OUT / (mode + '_catalog_validation.log')).open('x', encoding='utf8') as f, contextlib.redirect_stdout(f):
        assert validate() == 0
    import pytest
    class Counts:
        passed = 0
        total = 0
        def pytest_runtest_logreport(self, report):
            if report.when == 'call' and report.passed:
                self.passed += 1
        def pytest_sessionfinish(self, session, exitstatus):
            self.total = session.testscollected
    counts = Counts()
    with (OUT / (mode + '_catalog_tests.log')).open('x', encoding='utf8') as f, contextlib.redirect_stdout(f):
        code = pytest.main(['tests/test_project_history.py', '-q'], plugins=[counts])
    assert code == 0 and counts.passed == counts.total
    verify()
    save(mode + '_preflight.json', dict(passed=True, catalog_tests=counts.passed, reused_math_proof='research_agreement_readout_20261109/math_checks.json',
                                      current_catalog_sha256=sha(ROOT/'experiments/project_history.json'), protocol_sha256=sha(OUT/'protocol.json')))
    print('CATALOG_CHECKS_PASSED', mode, counts.passed)
