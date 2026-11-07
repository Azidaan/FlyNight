"""Оформить проверенные19 без научного вывода20; обязательные проверки до сохранения."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime
import sys,time,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching,now
from scripts.b123_safe_journal_20261027 import journal

def main():
    folder=RUN/'b1_partial_finalization';end=datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp()
    p=read(folder/'protocol.json')
    for path,digest in p['sha256'].items():assert sha(ROOT/path)==digest,path
    previous=RUN/'b1_partial_audit';c=read(previous/'controller_result.json');r=read(previous/'report.json')
    assert c['exit_code']==0 and c['all_owned_ended'] and not any(matching(k,v) for k,v in c['owned'].items())
    assert r['passed'] and r['videos']==19 and not r['full_series'] and r['scientific_verdict'] is None
    if end-time.time()<240:
        write(folder/'report.json',dict(complete=False,reason='Недостаточно времени для обязательных проверок; каталог не менялся',audit_passed=True,updated=now()));return
    history=read(ROOT/'experiments/project_history.json');card=next(c for c in history['experiments'] if c['id']=='b1_full_tracker_pilot_v1')
    new=dict(obtained='runs/recovery_research_20261027/b1_partial_audit/partial_source.json:19полных видео из20, последнее не завершено к прежнему срокуGPU. runs/recovery_research_20261027/b1_partial_audit/report.json: passed=true/full_series=false/videos=19, events_checked='+str(r['events_checked'])+', строгий CPU-повтор всех готовых назначений, голов, raw/JSON и метрик. scientific_verdict=null. Старые контроллеры350/360сотказами сохранены; новых GPU-расчётов не было.',
        interpretation='Готовые19технически проверены, но полная заранее заданная серия20не завершена. По этому подмножеству научный вывод о пользе/вредеБ1не делается.',
        decision='Сохранить проверенные19и исходники поSHA. Для полногоБ1 в новом разрешённом окне остаётся последнее видео, сведение и полный аудит20; готовые19не повторять. Подготовка1/3/4не является научным результатомБ1. Прежний бюджет не продлён.')
    for field,value in new.items():
        card.setdefault('revisions',[]).append(dict(date='2026-10-28',commit='',field=field,old=card[field],reason='Завершился строгий независимый аудит19готовых видео; полного20нет'))
        card[field]=value
    card['acceptance'][0]['met']=False
    card['acceptance'][0]['evidence']='runs/recovery_research_20261027/b1_partial_audit/report.json: строгие проверки19прошли, но требуемые20не завершены'
    card['evidence'].append(dict(kind='file',ref='runs/recovery_research_20261027/b1_partial_audit/report.json',claim='Независимый CPU-повтор готовых19, не полный научный итог20'))
    write(ROOT/'experiments/project_history.json',history)
    outcomes=[]
    commands=[['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q'],
        ['scripts/restore_obsidian_sections_20261027.py'],['-m','pytest','tests/test_obsidian_fundamental_overview.py','-q']]
    for index,args in enumerate(commands):
        write(folder/'status.json',dict(state='running',step=index,command=args,updated=now()),required=False)
        with (folder/f'check_{index}.log').open('w',encoding='utf-8') as out:
            result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=max(1,end-time.time()-10))
        outcomes.append(dict(command=args,exit_code=result.returncode));assert result.returncode==0,args
    write(folder/'report.json',dict(complete=True,partial_audit_passed=True,scientific_verdict=None,checks=outcomes,updated=now()))
    journal('b1_partial_audit_completed','Готовые19видеоБ1строго проверены; полная серия20не завершена',
        '*Получено.* `runs/recovery_research_20261027/b1_partial_audit/report.json`: passed=true, videos=19, full_series=false, events_checked='+str(r['events_checked'])+'. '
        '*Вывод.* Решения/головы/raw/JSON/метрики готовых видео независимо повторены; научный вывод20отсутствует. '
        '*Решение.* Не повторять19; в новом окне остаётся последнее видео/сведение/аудит20. `b1_partial_finalization/report.json`: обязательные каталог/pytest/сохраняющий граф/тесты прошли. Старые отказы и научные ограничения сохранены.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n23в. Аудит390готовых19и оформление420завершены: b1_partial_audit/report.json passed/full_series=false, б1_partial_finalization/report.json checks all0. Полного20/scientific_verdictпо-прежнему нет. Готовые19неповторять,старые350/360отказынеизменны. Проверить фактическиеcontroller/PIDend и результаты400/411,сохранить точный остаток1/3/4.\n')

if __name__=='__main__':main()
