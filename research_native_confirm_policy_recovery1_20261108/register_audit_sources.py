import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_native_confirm_policy_20261108.common import *


def main():
    assert not (RUN/'comparison_registration.json').exists()
    names=[p.relative_to(ROOT).as_posix() for folder in ['research_native_confirm_policy_20261108','research_native_confirm_policy_recovery1_20261108'] for p in (ROOT/folder).glob('*.py')]
    write(RUN/'independent_audit_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources={n:sha(ROOT/n) for n in names},reason='Bind all original and recovery sources, including the independent auditor added after the input phase. No sources modified and no comparison result yet.'),new=True)


if __name__=='__main__':main()
