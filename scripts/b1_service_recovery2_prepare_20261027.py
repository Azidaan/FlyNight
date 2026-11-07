"""Остановить только свои зависшие процессы и зафиксировать служебное продолжение."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
import psutil

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/b123_20261027';TRAIN=RUN/'b1_training'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def stop(identity,expected):
    p=psutil.Process(identity['pid'])
    assert abs(p.create_time()-identity['created'])<.01
    assert any(expected in token for token in p.cmdline()),p.pid
    tree=p.children(recursive=True)+[p]
    identities=[dict(pid=q.pid,created=q.create_time()) for q in tree]
    for q in reversed(tree):
        try:q.kill()
        except psutil.NoSuchProcess:pass
    psutil.wait_procs(tree,timeout=10)
    for identity in identities:
        if psutil.pid_exists(identity['pid']):
            assert abs(psutil.Process(identity['pid']).create_time()-identity['created'])>=.01
    return identities


def main():
    assert read(RUN/'queue_status.json')['state']=='waiting'
    assert read(TRAIN/'status.json')['done']==6039
    registry=read(RUN/'process_registry.json')['processes']
    queue=next(q for q in registry if q['pid']==2668)
    queues=stop(queue,'b123_ready_queue_recovery1_20261027.py')
    worker=read(TRAIN/'recovery1_launch.json')
    workers=stop(worker,'b1_training_recovery1_20261027.py')
    proof=dict(at=datetime.now(timezone.utc).isoformat(),
        reason='D progress stopped at6039; repeated own stack samples in FlyBackbone.step advanced indexing assignment. Exact cause unresolved; use distinct-index index_copy after CPU value and gradient equivalence test.',
        stopped_queue=queues,stopped_worker=workers,all_owned_workers_ended=True,
        C_checkpoint_sha256=sha(TRAIN/'C/checkpoint.pt'),D_checkpoint_sha256=sha(TRAIN/'D/checkpoint.pt'),
        source_status=read(TRAIN/'status.json'),scientific_settings_changed=False,
        deadline_preserved=read(TRAIN/'protocol.json')['deadline'])
    save(TRAIN/'service_interruption2.json',proof)
    files=['scripts/b1_training_recovery2_20261027.py','tracking/b1_index_copy_stimulus.py',
           'tests/test_b1_index_copy_stimulus.py','runs/b123_20261027/index_copy_tests.log',
           'runs/b123_20261027/b1_training/service_interruption2.json']
    save(TRAIN/'service_recovery2.json',dict(at=datetime.now(timezone.utc).isoformat(),
        deadline=read(TRAIN/'protocol.json')['deadline'],scientific_settings_changed=False,
        resume='Retain completed C report/probabilities and D checkpoint cursor6039; no completed training repeated.',
        sha256={name:sha(ROOT/name) for name in files}))
    old=ROOT/'scripts/b123_ready_queue_recovery1_20261027.py'
    new=ROOT/'scripts/b123_ready_queue_recovery2_20261027.py'
    assert not new.exists()
    text=old.read_text(encoding='utf-8')
    assert 'b1_training/recovery1_controller.json' in text
    new.write_text(text.replace('b1_training/recovery1_controller.json','b1_training/recovery2_controller.json'),encoding='utf-8')
    save(RUN/'queue_recovery2_manifest.json',dict(dependency='b1_training/recovery2_controller.json',
         original='scripts/b123_ready_queue_recovery1_20261027.py',source=new.relative_to(ROOT).as_posix(),sha256=sha(new),
         previous_queue_ended=True,scientific_settings_changed=False))
    print(json.dumps(proof))


if __name__=='__main__':main()
