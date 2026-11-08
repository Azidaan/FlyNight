"""Run-local paths; scientific helpers from the preserved parent execution."""
from research_identity_assignment_run_20261108.common import *
BASE=RUN
RUN=ROOT/'runs/singleton_free_20261108'
ARMS=['B0','U_THRESHOLD','U_GEO','U_SHIFT','U_IMAGE','U_R1','U_DIS','U_FLYVIS','U_PERM']


def phase(name='phase.json'):
    value=read(RUN/name)
    for p,h in value['sources'].items():assert sha(ROOT/p)==h,p
    return value


def status(stage,**values):write(RUN/'status.json',dict(stage=stage,updated=datetime.now(MSK).isoformat(),**values))


def guard():
    import sys
    sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
    from system_memory_guard import guard_entrypoint
    deadline=datetime(2026,11,8,14,0,tzinfo=MSK)
    return guard_entrypoint(timeout=max(1,int((deadline-datetime.now(MSK)).total_seconds())))
