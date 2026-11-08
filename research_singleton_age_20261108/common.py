from research_singleton_verify_20261108.common import *
OWN=RUN
RUN=ROOT/'runs/singleton_age_20261108'
FULL=RUN/'full'
OWN_FULL=OWN/'full'


def status(stage,**values):write(RUN/'status.json',dict(stage=stage,updated=datetime.now(MSK).isoformat(),**values))
def phase(name='phase.json'):
    p=read(RUN/name)
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p
