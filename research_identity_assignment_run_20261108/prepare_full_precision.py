"""Register a precision-only technical recovery, preserving failed and provisional artifacts."""
import sys,shutil,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *
FULL=RUN/'full_precision'


def main():
    old=RUN/'full';p=read(old/'protocol.json');assert read(old/'zero_parity.json')['passed']
    FULL.mkdir(exist_ok=True)
    p['technical_recovery']=dict(parent_protocol_sha256=sha(old/'protocol.json'),
        reason='Retina outputs native float64 after frame0. Prior full pilot cast before resize, unlike frozen EyeBank input path.',
        fix='Lossless float64 streaming storage with original per-frame dtype restored before resizing. Bounded frame LRU; immutable checkpoints.',
        scientific_changes=False,disposable_scratch='Only full decoded frames after all used retinal sequences and final outputs are sealed; source movies retained.')
    write(FULL/'protocol.json',p,new=True)
    shutil.copytree(old/'runtime_inputs',FULL/'runtime_inputs')
    write(FULL/'zero_parity.json',dict(read(old/'zero_parity.json'),reused_from=str((old/'zero_parity.json').relative_to(ROOT)),
        rationale='Zero corrections do not access representations or decoded input. Tracker and conversion source hashes unchanged.'),new=True)
    video=next(iter(p['videos']))
    for arm in ['B0','G']:
        for kind,suffixes in [('outputs',['.json']),('metrics',['.json','.details.json.gz'])]:
            for suffix in suffixes:
                source=old/kind/arm/(video+suffix);dest=FULL/kind/arm/source.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        receipt=read(old/'rollouts'/arm/(video+'.receipt.json'));assert receipt['passed'] and not receipt['feature_files']
        artifacts={}
        for name,h in receipt['artifacts'].items():
            source=ROOT/name;assert sha(source)==h;dest=FULL/source.relative_to(old);dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copyfile(source,dest)
            assert sha(dest)==h;artifacts[dest.relative_to(ROOT).as_posix()]=h
        receipt.update(artifacts=artifacts,reused_from=str((old/'rollouts'/arm/(video+'.receipt.json')).relative_to(ROOT)),
            reason='beta=0: no image or FlyVis input accessed; exact unchanged geometric tracker.')
        write(FULL/'rollouts'/arm/(video+'.receipt.json'),receipt,new=True)
    sources=dict(read(RUN/'full_execution_recovery2_phase.json')['sources'])
    for path in (ROOT/'research_identity_assignment_run_20261108').glob('*.py'):
        if path.name.startswith('full_precision') or path.name=='prepare_full_precision.py':
            ast.parse(path.read_text(encoding='utf-8-sig'));sources[path.relative_to(ROOT).as_posix()]=sha(path)
    sources[(FULL/'protocol.json').relative_to(ROOT).as_posix()]=sha(FULL/'protocol.json')
    write(RUN/'full_precision_technical_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,
        purpose='Actual native float64 and float32 input exactness gate before full corrected execution.'),new=True)
    write(FULL/'recovery_registration.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,
        prior_memory_stop=str((old/'isolated_children/cpu_b23b2649-8349d2a1.ended.json').relative_to(ROOT)),
        retained_prior_artifacts=True,scientific_settings_unchanged=True,old_signal_results_provisional_until_exact_input_validation=True),new=True)


if __name__=='__main__':main()
