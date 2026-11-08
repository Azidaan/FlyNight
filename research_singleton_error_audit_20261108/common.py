from research_singleton_age_20261108.common import *
AGE=RUN
RUN=ROOT/'runs/singleton_error_audit_20261108'


def status(stage,**values):write(RUN/'status.json',dict(stage=stage,updated=datetime.now(MSK).isoformat(),**values))
def phase():
    p=read(RUN/'phase.json')
    for name,h in p['sources'].items():assert sha(ROOT/name)==h,name
    return p
