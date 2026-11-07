"""Read-only process audit and final record; never starts research work."""
from pathlib import Path
from datetime import datetime, timezone
import json,hashlib,psutil
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/continuation_20261028'

def main():
    result=json.loads((RUN/'deadline_result.json').read_text(encoding='utf-8'))
    identities={(int(k),float(v)) for k,v in result['owned'].items()}
    registry=json.loads((RUN/'registry.json').read_text(encoding='utf-8'))
    identities.update((int(x['pid']),float(x['created'])) for x in registry['processes'])
    controllers=[]
    for path in RUN.rglob('controller_result.json'):
        c=json.loads(path.read_text(encoding='utf-8'))
        identities.update((int(k),float(v)) for k,v in c.get('owned',{}).items())
        controllers.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),exit_code=c.get('exit_code'),all_owned_ended=c.get('all_owned_ended')))
    for name in ['queue.launch.json','guard.launch.json']:
        item=json.loads((RUN/name).read_text(encoding='utf-8'))
        identities.add((int(item['pid']),float(item['created'])))
    live=[];reused=[]
    for pid,birth in sorted(identities):
        try:
            p=psutil.Process(pid);actual=p.create_time()
            if abs(actual-birth)<.01:live.append(dict(pid=pid,birth=birth))
            else:reused.append(dict(pid=pid,old_birth=birth,current_birth=actual))
        except psutil.NoSuchProcess:pass
    final=dict(checked_at=datetime.now(timezone.utc).isoformat(),guard_reported_all_owned_ended=result['all_owned_ended'],
        guard_result_updated=result['updated'],guard_remaining=result['remaining'],checked_pid_birth_count=len(identities),
        all_recorded_owned_actually_ended=not live,live_owned=live,reused_pid_identities=reused,
        old_controller_results_preserved=controllers,research_started=False,
        deadline_result_sha256=hashlib.sha256((RUN/'deadline_result.json').read_bytes()).hexdigest())
    target=RUN/'final_stop_verified_20261029.json'
    assert not target.exists(),'Keep the previous final audit'
    target.write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    assert result['all_owned_ended'] and not result['remaining'] and not live,live
    text='\n\n29.10.2026, 07:44 MSK — Финальная проверка после доставки завершающего вызова. Независимый guard записал deadline_result.json/all_owned_ended=true в07:39:30МСК, очередь окончилась приsave07:30. Проверены все PID+birth из guard/registry/всех controller_result/queue и guardlaunch: живых собственных деревьев нет. Первичный акт final_stop_verified_20261029.json сохраняет число проверенных идентичностей, хеш deadline_result и исходные коды контроллеров. Новых расчётов не запускалось. Автоматизация1 удаляется после этой фактической проверки; возобновление только по новому разрешению автора.\n'
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:f.write(text)
    print(json.dumps({k:v for k,v in final.items() if k not in ['old_controller_results_preserved','reused_pid_identities']},ensure_ascii=False))

if __name__=='__main__':main()
