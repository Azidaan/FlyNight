"""Сохранить истечение срока аудита без подмены его успехом."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime
import sys,time,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching,now
from scripts.b123_safe_journal_20261027 import journal

def main():
    folder=RUN/'b1_final_timeout_save';audit=RUN/'b1_partial_audit';c=read(audit/'controller_result.json')
    actual=[k for k,v in c['owned'].items() if matching(k,v)]
    assert not actual,'Сначала проверить фактический конец позднего дочернего процесса'
    write(audit/'late_owned_end_verified.json',dict(passed=True,actual_all_owned_ended=True,original_controller_sha256=sha(audit/'controller_result.json'),original_exit_code=c['exit_code'],updated=now()))
    status=read(audit/'status.json')
    write(folder/'outcome.json',dict(b1_complete=False,saved_videos=19,requested_videos=20,audit_complete=False,
        audit_controller_sha256=sha(audit/'controller_result.json'),audit_last_status=status,scientific_verdict=None,
        remaining=['Последнее видео Б1','Сведение20 и независимый полный аудит','Реальный экспорт новых входов1/3/4, совмещение камеры, обучаемый интерфейс, цена/обучение/HOTA'],updated=now()))
    history=read(ROOT/'experiments/project_history.json');card=next(c for c in history['experiments'] if c['id']=='b1_full_tracker_pilot_v1')
    fields=dict(obtained='runs/recovery_research_20261027/b1_final_timeout_save/outcome.json: сохранены19из20видео; GPU остановился по прежнему абсолютному сроку, последнее видео не завершено. Независимый аудит19тоже не уложился в свой прежний срок; финального report.json нет. Промежуточный status.events_checked='+str(status['events_checked'])+' не является завершённым актом аудита. Источник: b1_partial_audit/controller_result.json exit15/timed_out=true; поздний фактический конец всехPID подтверждён отдельно, исходный отказ не переписан.',
        interpretation='Ни полная серия20, ни дополнительный аудит всех19не завершены. Это ограничение исполнения и бюджета, не научный отрицательный результат Б1. Нельзя говорить, что готовые19полностью аудированы.',
        decision='Сохранить19выходов/события/метрики и промежуточный прогресс аудита. В новом разрешённом окне продолжать только незавершённые части; полное видео/готовые GPU-выходы не повторять. Зависимые400/411/420остановились по непрошедшей зависимости, без новых данных/обучения/HOTA. До нового разрешения расчёты после07:40не продолжать.',
        cannot_claim='Нет полного HOTA20 и полного аудита19; нет научной приёмки Б1 и результатов матрицы1/3/4. Промежуточные проверки не заменяют финальный отчёт; бюджет не продлён. Старые отказы и поля карточки сохранены в revisions.')
    for field,value in fields.items():
        card.setdefault('revisions',[]).append(dict(date='2026-10-28',commit='',field=field,old=card[field],reason='Истёк прежний бюджет аудита19; финального акта нет'))
        card[field]=value
    card['acceptance'][0]['met']=False;card['acceptance'][0]['evidence']='runs/recovery_research_20261027/b1_final_timeout_save/outcome.json:20не завершены, аудит19не завершён'
    card['evidence'].append(dict(kind='file',ref='runs/recovery_research_20261027/b1_final_timeout_save/outcome.json',claim='Фактический остаток и отсутствие полного научного результата'))
    write(ROOT/'experiments/project_history.json',history)
    journal('b1_audit_timeout','Б1:сохранены19выходов, аудит тоже не завершён к прежнему сроку',
        '*Получено.* `runs/recovery_research_20261027/b1_final_timeout_save/outcome.json`:19из20выходов сохранены, аудит390остановлен по прежнему сроку безфинального report. Промежуточный счётчик'+str(status['events_checked'])+' не является завершённым аудитом. Поздний конец всехPIDпроверен отдельно, оригинальный отрицательныйконтроллерсохранён. '
        '*Вывод.* Научного отрицательного результата нет. *Решение.*400/411/420получили отказы зависимости, обучение/новаяGPUсерияне запускались. Сохранить точный остаток до нового разрешённого окна.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 24. Финальное сохранение:аудит390тоже НЕзавершился\n\n'
            '390дошла до прежнейграницы07:23:09,exit15/timed_out/all_owned_ended=falseсremaining21544;позднийфактическийконецпроверенотдельно late_owned_end_verified.json,оригинальныйотказНЕпереписан. report390НЕТ. Последнийсчётчик533370неполныйакт.400/411/420упалинадепенденции,новыйэкспортНЕисполнялся.410доэтогоотмененаДОзапуска. '
            'GPUвыходов19из20сохранены,аудит19НЕзавершён; нельзя назватьихполностьюаудированными. b1_final_timeout_save/outcome.jsonточныйостаток. ПоследнеевидеоБ1/сведение20/незавершённыйаудиттребуютновогоокна;готовыеGPU19неповторять. '
            'Программа1/3/4planned:входныефункциии5тестовесть,реальныйэкспорт/совмещениекaмеры/обучаемыйпуть/обучение/HOTAнеготовы.430сохраняетпоправкускаталогом/обязательнымипроверками;читатьегоreport/логи,невыдаватьнеполноезелёным. После07:30непубликоватьрасчёты,финально07:39проверитьguard/PIDbirth,07:40автоматизацияостанавливается.\n')
    end=datetime.fromisoformat(read(folder/'execution_deadline.json')['deadline']).timestamp();outcomes=[]
    for i,args in enumerate([['scripts/validate_project_history.py'],['-m','pytest','tests/test_project_history.py','-q']]):
        with (folder/f'check_{i}.log').open('w',encoding='utf-8') as out:
            r=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,timeout=max(1,end-time.time()-5))
        outcomes.append(dict(command=args,exit_code=r.returncode));assert r.returncode==0
    write(folder/'report.json',dict(complete=True,catalog_checks=outcomes,scientific_verdict=None,updated=now()))

if __name__=='__main__':main()
