"""Correct only argparse isolation for a previously unstarted validation stage."""
import study as s
from study import *
assert not (OUT/'final_checks.json').exists()
old=(OUT/'checks.log').read_text(encoding='utf8');assert '"returncode": 2' in old
failed=OUT/'catalog_validation.log';assert failed.exists() and failed.stat().st_size==0
failed.rename(OUT/'catalog_validation_failed_arguments.log')
write('checks_recovery_protocol.json',dict(registered=stamp(),reason='runpy inherited the action argument checks; validator argparse rejected it before catalogue validation or pytest began. Set sys.argv to the validator program only. No catalogue, scientific setting, test or result changed. Previous failure preserved.',sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),OUT/'finish_study.py',OUT/'checks.log',OUT/'catalog_validation_failed_arguments.log']}))
import finish_study
sys.argv=['scripts/validate_project_history.py']
finish_study.checks()
