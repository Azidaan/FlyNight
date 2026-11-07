"""Сохранить точную неполноту Б1 и поставить строгий аудит готовых видео."""
from pathlib import Path
from datetime import datetime,timezone
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching,now
from scripts.b123_safe_journal_20261027 import journal

def main():
    source=RUN/'b1_tracker/continuation_recovery1';folder=RUN/'b1_partial_audit'
    c=read(source/'controller_result.json');assert c['exit_code']==1 and c['all_owned_ended']
    assert not any(matching(k,v) for k,v in c['owned'].items())
    err=(source/'stderr.log').read_text(encoding='utf-8')
    assert "TimeoutError: Original B1 tracker budget" in err
    p=read(source/'protocol.json');videos=[s['video'] for s in p['sources'] if (source/'completed'/(s['video']+'.json')).exists()]
    assert len(videos)==19 and len(p['sources'])==20 and not (source/'report.json').exists()
    paths=[source/'controller_result.json',source/'protocol.json',source/'per_video.json',source/'status.json',source/'stderr.log',source/'resume_verified.json']
    for v in videos:
        paths.append(source/'completed'/(v+'.json'))
        for arm in ['baseline','appearance','zero','C','D','R1','B']:
            paths.extend([source/arm/(v+'.json'),source/'raw'/arm/(v+'.npz'),source/'events'/arm/(v+'.json')])
    saved=dict(complete=False,scientific_verdict=None,requested_videos=20,completed_videos=videos,
        missing_videos=[s['video'] for s in p['sources'] if s['video'] not in videos],
        stop_reason='Истёк прежний абсолютный бюджет GPU; не сбой науки и не результат20',
        last_status=read(source/'status.json'),controller_sha256=sha(source/'controller_result.json'),
        sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},updated=now())
    write(folder/'partial_source.json',saved)
    old_end=datetime.fromisoformat(read(RUN/'b1_tracker_audit/continuation_recovery1/execution_deadline.json')['deadline']).timestamp()
    budget=int(old_end-time.time()-60);assert budget>300
    script='scripts/audit_b1_partial_20261028.py'
    write(folder/'protocol.json',dict(budget_seconds=budget,original_audit_deadline=datetime.fromtimestamp(old_end,timezone.utc).isoformat(),
        parent_experiment='b1_full_tracker_pilot_v1',sha256={script:sha(ROOT/script),
            'scripts/audit_b1_tracker_20261028.py':sha(ROOT/'scripts/audit_b1_tracker_20261028.py')},
        scientific_verdict=None,partial_source_sha256=sha(folder/'partial_source.json')))
    name='390_b1_partial_audit';assert not (RUN/'ready_jobs'/(name+'.json')).exists()
    write(RUN/'ready_jobs'/(name+'.json'),dict(name=name,folder=folder.relative_to(ROOT).as_posix(),script=script,
        script_sha256=sha(ROOT/script),protocol_sha256=sha(folder/'protocol.json')))
    history=read(ROOT/'experiments/project_history.json');card=next(c for c in history['experiments'] if c['id']=='b1_full_tracker_pilot_v1')
    changes=dict(state='inconclusive',done='2026-10-28',
        obtained='runs/recovery_research_20261027/b1_partial_audit/partial_source.json: requested_videos=20, completed_videos содержит19видео; последнее b23f7012-fab06dac не завершено, last_status.frame=855/1205. controller_result.json исходного continuation_recovery1: exit_code=1/all_owned_ended=true; stderr.log: TimeoutError Original B1 tracker budget. Все готовые файлы и прежние отказы сохранены по SHA. Новый независимый аудит19подготовлен, ещё не завершён.',
        interpretation='Серия остановлена по прежнему абсолютному сроку GPU. Полной HOTA20 и научного вывода о Б1 нет. Неполнота не подтверждает и не опровергает пользу.',
        decision='Не продлевать прежний бюджет. Проверить и сохранить19готовых видео в остатке уже выделенного окна аудита, без повторного GPU. При новом разрешённом окне продолжить только последнее видео и сведение/полный аудит. Следующая программа1/3/4 остаётся planned; неполный Б1 не считать отрицательным контролем.',
        cannot_claim='Нет полного20результата/научной приёмки; подмножество19не является заранее выбранным тестом и не может заменить20. Нельзя выдавать возврат к будущему запуску за продление старого окна. Один seed и прежние ограничения сохранены.')
    for field,value in changes.items():
        if field in card and card[field]!=value:
            card.setdefault('revisions',[]).append(dict(date='2026-10-28',commit='',field=field,old=card[field],reason='Сохранён фактический стоп продолжения по абсолютному бюджету,19из20; прежние тексты сохранены'))
        card[field]=value
    card['evidence'].append(dict(kind='file',ref=(folder/'partial_source.json').relative_to(ROOT).as_posix(),claim='Неполнота19из20, причина и контрольные суммы'))
    write(ROOT/'experiments/project_history.json',history)
    journal('b1_partial_preservation','Б1 остановлена по прежнему сроку:19готовых видео, полный итог отсутствует',
        '*Получено.* `runs/recovery_research_20261027/b1_partial_audit/partial_source.json`:19из20полных видео, последнее остановилось на855/1205. Причина — исходный абсолютный срок, stderr TimeoutError Original B1 tracker budget; фактический конец дерева подтверждён. '
        '*Вывод.* Это неполная серия, не научный отрицательный результат. *Решение.* Старый бюджет не продлевать;390строго повторяет готовые результаты наCPU в остатке прежнего окна аудита, без GPU. Карточка inconclusive, критерии научного качества остаются неоценёнными.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 23. 06:40МСК:350остановилась по прежнему абсолютному сроку,19готовы\n\n'
            'Старые350/360код1/allended,380код0сprerequisite_passed=false. Это не научный отрицательный итог. ТекущегоGPUнет;queue_recovery1/guardживы. '
            'Все19полных видео/метрики/события/сырые выходы/первое reused сохранены поSHA в b1_partial_audit/partial_source.json. Последнее видео на855/1205НЕготово. Не продолжатьGPUвстаромокне и не повторять19. '
            '390_b1_partial_audit CPUтолькоостатокПРЕЖНЕГО360абсолютногосрокаминус60с; проверяет все старые строгиеусловия на19, сохраняя оригинальныеотказы. Прямо full_series=false/scientific_verdict=null; никаких утверждений поHOTA20. После390прочитатьreport/controller/PIDend и оформитьитогпроверок. '
            'Затем программа1/3/4:380непроверяладанные из-занеполногоБ1; приоставшемсяокне подготовитьисходникибезповторенияGPU, явноотдельноотнеполногоБ1. Полнуюматрицудо07:40необещать. Сохранение07:30/stop07:39:30/PAUSED07:40,lockнетронут.\n')
    print(dict(completed=len(videos),budget_seconds=budget,queued=name))

if __name__=='__main__':main()
