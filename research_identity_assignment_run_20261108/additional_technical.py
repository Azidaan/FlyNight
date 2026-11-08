"""Tiny independent event fixtures: gaps, sparse unknowns, persistence and duplicates."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.full_additional import return_events
from research_identity_assignment_run_20261108.common import *


def row(tid):return [0,0,10,10,tid,.9,2]


def main():
    truth={f:[('a','car',None)] for f in [0,2,4,6]}
    outputs={f:[row(1)] for f in [0,2,3,4,5,6]}
    matches={f:{1:'a'} for f in truth}
    a=return_events(outputs,truth,matches,{2:1})
    assert a['totals']==dict(reactivations=1,correct_returns=1,sustained_success=1,duplicate_extra_IDs=1,annotated_frames_with_duplicates=1)
    b=return_events({f:[row(1)] for f in [0,2,3,5,6]},truth,matches,{})
    assert b['events'][0]['sustained']=='failed'
    c=return_events({f:[row(1)] for f in [0,3,4,5,6]},truth,matches,{})
    assert c['events'][0]['kind']=='unknown'
    d=return_events(outputs,truth,{0:{1:'a'},2:{1:'b'},4:{1:'a'},6:{1:'a'}},{})
    assert d['events'][0]['kind']=='wrong'
    e=return_events(outputs,truth,{0:{1:'a'},2:{1:'a'},6:{1:'a'}},{})
    assert e['events'][0]['sustained']=='censored_unknown'
    f=return_events({0:[row(1)],6:[row(1)]},truth,matches,{})
    assert f['events'][0]['sustained']=='censored_insufficient_future'
    write(RUN/'full_precision/additional_technical.json',dict(passed=True,fixtures=6,definitions_source_sha256=sha(RUN/'full_execution_phase.json'),
        calculator_sha256=sha(ROOT/'research_identity_assignment_run_20261108/full_additional.py')),new=True)


if __name__=='__main__':main()
