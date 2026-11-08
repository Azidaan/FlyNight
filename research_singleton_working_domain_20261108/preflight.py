from research_singleton_working_domain_20261108.common import *


def main():
    guard(600)
    phase()
    import runpy
    previous = sys.argv
    try:
        sys.argv = ['scripts/validate_project_history.py']
        runpy.run_path(str(ROOT / sys.argv[0]), run_name='__main__')
    except SystemExit as e:
        assert e.code in [None, 0]
    finally:
        sys.argv = previous
    import pytest
    assert pytest.main(['-q', 'tests/test_project_history.py', '--basetemp', str(RUN/'preflight_temp')]) == 0
    phase()
    write(RUN / 'preflight.json', dict(passed=True, catalogue_tests=58), new=True)


if __name__ == '__main__':
    main()
