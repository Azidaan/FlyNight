"""Сохранить идентичность запуска, измеренную цену и продолжение для агента."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,sys,time
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027';FOLDER=RUN/'sam2_query_output_pilot'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    launch=read(FOLDER/'launch.json');process=psutil.Process(launch['pid'])
    assert abs(process.create_time()-launch['created'])<.01 and 'b123_sam2_query_output_pilot_20261027.py' in ' '.join(process.cmdline())
    status=read(FOLDER/'status.json');assert status['state']=='running' and time.time()-datetime.fromisoformat(status['updated']).timestamp()<30
    price=read(FOLDER/'price.json');assert price['proceed']
    guard=read(RUN/'deadline_status.json');assert time.time()-datetime.fromisoformat(guard['updated']).timestamp()<10
    proof=dict(checked=datetime.now(timezone.utc).isoformat(),launch=launch,identity_verified=True,status=status,
        full_series_admitted=True,price=price,deadline_guard_fresh=True,full_pilot_completed=False)
    (FOLDER/'launch_verified.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    audit=RUN/'sam2_query_audit';audit.mkdir()
    source='scripts/b123_audit_query_mask_20261027.py'
    protocol=dict(id='sam2_query_saved_output_audit',budget_seconds=180,author_stop='2026-10-27T07:20:00+03:00',
        action='CPU-only saved-output audit after existing job140; no inference, training, TrackEval or catalogue mutation',
        source_sha256=sha(ROOT/source),pilot_protocol_sha256=sha(FOLDER/'protocol.json'))
    (audit/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    job=dict(title='Независимая сверка сохранённых выходов масочной выдачи',folder=audit.relative_to(ROOT).as_posix(),
        script=source,script_sha256=sha(ROOT/source),protocol_sha256=sha(audit/'protocol.json'))
    (RUN/'ready_jobs/150_sam2_query_audit.json').write_text(json.dumps(job,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from scripts.b123_safe_journal_20261027 import journal
    journal('sam2_query_mask_output_pilot_v1','масочная выдача по запросу допущена к серии по измеренной цене',
        '*Проверено.* PID/создание исполнителя и свежий прогресс совпали; независимый контроллер срока свежий. '
        f'Первое полное видео {price["first_video_seconds"]:.3f}с; консервативная остаточная цена {price["conservative_remaining_seconds"]:.3f}с '
        f'< доступных {price["available_seconds"]:.3f}с, proceed=true. Источники:runs/b123_20261027/sam2_query_output_pilot/price.json иlaunch_verified.json. '
        'Тот же исходный rawXYWH и HOTA первого видео проверены исполнителем точно. Остальные19 продолжаются в той же очереди, готовое видео не повторяется. '
        'HOTA20 ещё нет; цена не доказывает качество. Технические тесты3passed, каталогOK/58passed. После серии готова CPU-сверка150 сохранённых выходов,безповторногоинференса. '
        'Коммит/push блокирует прежний .git/index.lock, он не изменялся. '
        'Независимый общий наблюдатель восстановлен отдельной версией после WinError5 при замене deadline_status.json. '
        'Старый PID25240 завершился, прежние источник/ошибка/статус сохранены; новый PID/создание и хеши вdeadline_guard_recovery1/launch.json. '
        'Временная ошибка записи статуса теперь обрабатывается ограниченным повтором, остановка не зависит от успешности очередной записи. '
        'Сроки07:20/07:29:30/07:30 и бюджет самого прогона не менялись.')
    with (RUN/'HANDOFF.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## '+datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M')+' МСК — масочная выдача по запросу прошла цену\n\n'
            'Плотный sam2_output_pilot120 остановился по цене:283.687с/первое видео,10900.106>6910.297с,код1/all_owned_ended. '
            'cost_stop_audit.json passed;карточкаinconclusive,неполнаяHOTA20 и не отрицательный вывод о масках. Старый бюджет не продлевать. '
            'Парныйsam2_precision_profile130готов:FP32 3.578с,BF16 3.281с,ускорение1.090521;21общаямаска,IoU0.975798..0.998087,рамкиизменились. '
            'Только9%ускорения недостаточно для плотного пилота,полноеBF16видеоНЕзапускалось. ДваBF16служебныхфайлаоставлены,нонепубликоватьихготовуюступень. '
            'Теперь140_sam2_query_output_pilot:новыйотдельныйFP32вариант,нейроннаямаскаобновляетсятолькопозапросамвыдачи,всеисходныекадры/реальныеназначениянаблюдаются. '
            'Запросы—фиксированныевременаизстаройalignment;меткиобъектовмодулюнедоступны. Это режимвыдачипозапросу,некачество непрерывного30Гцмасочного выхода. '
            f'Первыйполныйпроход {price["first_video_seconds"]:.3f}с,цена {price["conservative_remaining_seconds"]:.3f}<{price["available_seconds"]:.3f},proceed=true. '
            'PID+birth/status/guardпровереныlaunch_verified.json. Серия20продолжаетсябезповтораготового,порогинеподбирать. '
            'После140локальнаяочередьготоваисполнить150_sam2_query_audit:независимые20baselineJSON/метрики/append-only/uniqueIDs/все3pairedCI/хеши;безинференса. '
            'Еслинетполногоreport,аудитсохранитpassed=false иостаток,ненаучныйотрицательныйисход. '
            'Послеreport+контроллеров140/150allendedагентфиксируетсодержательныйитогкарточки/журнала/проверок/однойфундаментальнойзадачи4. '
            'ГлавныеБ1–Б3непровереныHOTA;собственнаяSAM2ветвьНЕLTRиНЕSGAR. BUSCAcompleted_negativeужеаудирован,неповторять. '
            'Новыхтехническихузловграфанесоздавать,файлынеудалять/неперемещать,гипотезыпобайтно. КаталогOK/58passed,3тестамасочногоходаpassed. '
            'ОбщийguardPID25240умер04:33из-заWinError5атомарнойзаменыстатуса;перезапущенотдельныйdeadline_guard_recovery1,PID+birthвlaunch.json. '
            'Исходныеисточник/логисохранены,сроктотже,старыеownedPIDунаследованы;ограниченныйповторзаписинепрерываетзащитусрока. '
            'Контроллерсамогопилотавсёвремяоставалсяработающимсо своимдедлайном06:36:38МСК. '
            'ОдинGPU,07:20сохранение/07:29:30стоп/07:30автоматизациюприостановить,.git/index.lockнетронут.\n')
    print(json.dumps(proof,ensure_ascii=False))
if __name__=='__main__':main()
