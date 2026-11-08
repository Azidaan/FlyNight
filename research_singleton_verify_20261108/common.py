from research_identity_assignment_run_20261108.common import *
BASE=RUN
PREVIOUS=ROOT/'runs/singleton_free_20261108'
RUN=ROOT/'runs/singleton_verify_20261108'


def sequence_key(video,sequence):return hashlib.sha256(json.dumps([video,sequence],separators=(',',':')).encode()).hexdigest()[:32]
def status(stage,**values):write(RUN/'status.json',dict(stage=stage,updated=datetime.now(MSK).isoformat(),**values))
def phase(name='phase.json'):
    p=read(RUN/name)
    for n,h in p['sources'].items():assert sha(ROOT/n)==h,n
    return p
def guard():
    sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
    from system_memory_guard import guard_entrypoint
    return guard_entrypoint(timeout=max(1,int((datetime(2026,11,8,14,0,tzinfo=MSK)-datetime.now(MSK)).total_seconds())))
