"""Локальные готовые переходы Б1–Б3; новые научные условия очередь не придумывает."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,subprocess,sys
import psutil

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/b123_20261027'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.next')
    temp.write_bytes((json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode());temp.replace(p)


def main():
    stop=datetime.fromisoformat(read(RUN/'authorization.json')['save_at']).timestamp()
    b1=RUN/'b1_training/recovery2_controller.json'
    while time.time()<stop:
        if not b1.exists():
            write(RUN/'queue_status.json',dict(state='waiting',stage='B1 training controller',updated=datetime.now(timezone.utc).isoformat()))
            time.sleep(5);continue
        prior=read(b1)
        if not prior.get('all_owned_ended'):raise RuntimeError('B1 process tree is not confirmed ended')
        if prior['exit_code']!=0:write(RUN/'b1_attention.json',dict(controller=prior,requires_agent_review=True))
        jobs=[]
        for path in sorted((RUN/'ready_jobs').glob('*.json')):
            if not (RUN/'jobs_done'/path.name).exists():jobs.append((path,read(path)))
        if not jobs:
            write(RUN/'queue_status.json',dict(state='waiting_for_ready_stage',stage='No prepared job; implementation or review needed',updated=datetime.now(timezone.utc).isoformat()))
            time.sleep(5);continue
        path,job=jobs[0]
        folder=(ROOT/job['folder']).resolve();folder.relative_to(RUN.resolve())
        script=(ROOT/job['script']).resolve();script.relative_to(ROOT.resolve())
        assert sha(script)==job['script_sha256'] and sha(folder/'protocol.json')==job['protocol_sha256']
        write(RUN/'queue_status.json',dict(state='running',stage=job['title'],job=path.name,updated=datetime.now(timezone.utc).isoformat()))
        with (folder/'parent.stdout.log').open('w',encoding='utf-8') as out,(folder/'parent.stderr.log').open('w',encoding='utf-8') as err:
            process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/b123_stage_controller_20261027.py'),
                '--folder',job['folder'],'--script',job['script']],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            owner=psutil.Process(process.pid);owner.nice(psutil.IDLE_PRIORITY_CLASS)
            registered=subprocess.run([sys.executable,str(ROOT/'scripts/b123_register_process_20261027.py'),
                '--pid',str(owner.pid),'--folder',job['folder']],cwd=ROOT,capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
            assert registered.returncode==0,registered.stderr
            code=process.wait()
        result_path=folder/'controller_result.json'
        result=read(result_path) if result_path.exists() else dict(error='controller ended without result')
        if not result.get('all_owned_ended'):raise RuntimeError('Stage ownership/end is not confirmed')
        write(RUN/'jobs_done'/path.name,dict(exit_code=code,result=result,job=job,finished=datetime.now(timezone.utc).isoformat()))
        if code!=0:write(folder/'agent_attention.json',dict(controller=result,requires_agent_review=True))
    write(RUN/'queue_status.json',dict(state='stopped_at_save_time',updated=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    try:main()
    except Exception as e:
        write(RUN/'queue_failure.json',dict(error=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
