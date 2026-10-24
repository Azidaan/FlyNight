"""Correct dependency binding for the resumed inputs, without worker changes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_val20_inputs_recovery1_20261024 as X
from scripts import z13_val20_queue_20261024 as Q
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now


def bind():
    X.I.STAGE=X.REC;Q.I.STAGE=X.REC;Q.R.I.STAGE=X.REC


if __name__=='__main__':
    bind()
    if '--prepare-raw' in sys.argv:Q.R.prepare()
    else:
        original_execute=Q.execute;original_update=Q.update
        def execute(name,args,allowed_failure=False):
            if args==['scripts/z13_val20_raw_20261024.py','--prepare']:args=[str(Path(__file__)),'--prepare-raw']
            return original_execute(name,args,allowed_failure)
        def update(stage,ident,ok,obtained,interpretation,decision):return original_update(stage,ident,ok,obtained.replace('z13_val20_matched_20261024/inputs/','z13_val20_matched_20261024/inputs_recovery1/'),interpretation,decision)
        Q.execute=execute;Q.update=update
        write(X.REC/'queue_execution_v2.json',dict(at=now().isoformat(),sha256=sha(Path(__file__)),reason='Raw preparation is a fresh process: explicitly bind resumed input directory there too. Scientific workers/protocols/data unchanged.'))
        try:Q.main()
        except Exception as e:write(Q.W/'val20_chain_recovery2_failure.json',dict(at=now().isoformat(),error=repr(e)));raise
