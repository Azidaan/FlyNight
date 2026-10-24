"""Record and publish the bounded resource-aware recovery, excluding live logs."""
from pathlib import Path
import json
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
W=ROOT/'runs/z13_weekend_20261024'
REC=ROOT/'runs/z13_val20_matched_20261024/common_authorized_recovery1'
OLD=REC.parent/'common_authorized'


def record_checks():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts import z13_common_authorized_recovery1_20261024 as R
    from scripts.z13_val20_common_20261024 import journal
    p=read(REC/'protocol.json');verify_sources(p);ps=WindowsProcesses()
    q=read(REC/'queue.json');assert ps.creation(q['pid'])==q['birth']
    assert p['save_at']=='2026-10-25T06:00:07.166100+03:00'
    assert p['hard_stop']=='2026-10-25T06:00:27.166100+03:00'
    console=read(REC/'console_recovery.json');assert console['console_window_handle']!=0
    # Binding only: no GPU inference or repeated completed work.
    bound=R.bind();assert bound.REC==REC and bound.S.STAGE==REC
    assert bound.S.__file__==str(ROOT/'scripts/z13_common_authorized_recovery1_20261024.py')
    status=read(REC/'status.json')
    write(W/'common_authorized_recovery1_start_verified.json',dict(at=now().isoformat(),
        queue=q,queue_identity_verified=True,console=console,protocol_sha256=sha(REC/'protocol.json'),
        current_status=status,original_deadline_unchanged=True,
        frozen_scientific_sources_verified=True,dispatch_binding_verified=True,
        no_completed_gpu_jobs_repeated=True))
    journal('проверена ресурсная очередь восстановления MASA',
        'runs/z13_weekend_20261024/common_authorized_recovery1_start_verified.json:очередьPID/'
        'время создания проверены,118готовых заданий сохранены. Ресурсная очередь не вычисляет '
        'GPU-признаки и не завершает чужие процессы. Отдельное окно наблюдения открыто '
        'черезPowerShell7,PID'+str(console['console_pid'])+',при запуске ненулевойWindowHandle. '
        'Первое окно черезWindowsPowerShell5 не запустилось:его парсер неверно прочитал '
        'русскийUTF8скрипт безBOM;подтверждены синтаксические ошибки декодирования. '
        'Замороженный скрипт и уже работающая очередь не менялись/не перезапускались. '
        'Результат запуска окна:common_authorized_recovery1/console_recovery.json. '
        'Научных новых итогов нет;контроллер прерванной стадии и её коды сохранены.')
    handoff=W/'HANDOFF.md'
    header=('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
        'ТЕКУЩАЯ стадия common_authorized_recovery1, scripts/z13_common_authorized_recovery1_20261024.py. '
        'ОчередьPID'+str(q['pid'])+'/birth'+str(q['birth'])+' работает без GPU;'
        'проверять queue.json/queue_result.json/status.json и фактическую идентичность. '
        'На записи ожидала завершения WarThunder/освобождения ресурсов:нетaces,>=8192МиБ '
        'свободно,загрузка<=30% непрерывно30с. Это служебные условия запуска,не научные пороги. '
        'Чужую игру/процессы не завершать. Интервал ожидания15с локально,без обращений к модели.\n\n'
        'Старая common_authorized остановлена15:28:01exit1/all_owned_ended:MASA четвёртого '
        'b230132b-ff8f2719 не обновлялась после500кадров;прежний600с сторож завершил своё дерево. '
        'Причина зависания не установлена. GPUсейчас тяжело занята и есть WarThunder,но '
        'причинность этим не доказана. Все118маркеров/хешей и конец старых PID/birth '
        'проверены,старые логи/ошибки/исходники сохранены. Нет готовых/временных6FlyMASA '
        'на прерванном видео;его единственный повтор будет свежим процессом. '
        'Готовые сигнатуры/3полныхMASAролика/4nativeрезультаты не повторяются.\n\n'
        'Очередь сама запускает независимый контроллер и --monitor после освобождения '
        'ресурсов,только если цена ещё помещается:первоначальные910,516с×17×2+900='
        +str(p['service_recovery_after_masa_stall']['conservative_remaining_seconds'])+
        'с. Включён один частичный ролик консервативно;нулевое время пропуска готового '
        'не используется для новой цены. Если при ожидании остатка недостаточно,'
        'queue_result state=not_started и status=paused,118заданий остаются,без вывода '
        'об эффективности. Исходный срок сохранения25.10 06:00:07.166100МСК и '
        'жёсткого конца06:00:27.166100 НЕПРОДЛЁН;общее26.10окно его не отменяет.\n\n'
        'При запуске смотреть controller.json/completion.json,поPID+birth. После exit0 '
        'и полного конца своего дерева монитор --record проверит231маркер/первичные '
        'строки/20общихHOTA/парные интервалы,запишет карточку сrevisions,журнал, '
        'проверки каталога/тесты/граф. Итог REC/report.json,verified_summary.json,'
        'results.md,finish_recorded.json. Ошибки W/val20_common_authorized_recovery1_failure.json '
        'и _monitor_failure.json. Старый W/val20_common_authorized_failure.json '
        'остаётся историей,не новой ошибкой. При повторе такого же зависания не '
        'перезапускать ещё раз без новой диагностики.\n\n'
        'КонсольPowerShell7 PID'+str(console['console_pid'])+' открыта отдельно;'
        'console_recovery.json. Старый launcher.ps1 использовалPS5,первое окно '
        'не запустилось из-заUTF8безBOM. Очередь при этом успешно стартовала;'
        'повторно launcher не запускать. Исходный консольный скрипт запущен через '
        'PowerShell7 без редактирования,видимость подтверждена ненулевымhandle.\n\n'
        'Авторское разрешение0,0001px по координатам действует;число строк/кадры/ID/'
        'классы/уверенности точны,научные веса/пороги/данные прежние. '
        'Новые отрицательные descriptor/loss-версии не продвигать;не открывать75,'
        'не загружать. Архитектурного изменения нет,чужие57графправок/.gitignore '
        'сохранить. Проверки/публикация:common_authorized_recovery1_start_checks.json '
        'и common_authorized_recovery1_start_publication.json.\n\n'
        'Ниже прежние передачи сохранены.\n\n')
    handoff.write_text(header+handoff.read_text(encoding='utf-8'),encoding='utf-8')
    from evaluation.project_history.index import fingerprint
    before=fingerprint(ROOT)
    for label,args in [('catalog',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','tests/test_z13_coordinate_replay_guard.py','tests/test_z13_status_read_retry.py','-q'])]:
        result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf-8'},capture_output=True)
        (W/('common_authorized_recovery1_start_'+label+'.log')).write_bytes(result.stdout)
        (W/('common_authorized_recovery1_start_'+label+'.stderr.log')).write_bytes(result.stderr)
        assert result.returncode==0,(label,result.returncode)
        print(label+' passed',flush=True)
    after=fingerprint(ROOT);assert before==after
    write(W/'common_authorized_recovery1_start_checks.json',dict(at=now().isoformat(),catalog_exit=0,tests_exit=0,fingerprint_before=before,fingerprint_after=after))


def publish():
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    protected=foreign_unchanged();checks=read(W/'common_authorized_recovery1_start_checks.json')
    assert checks['catalog_exit']==checks['tests_exit']==0
    assert '66 passed' in (W/'common_authorized_recovery1_start_tests.log').read_text(encoding='utf-8')
    assert read(W/'common_authorized_recovery1_start_vault.json')['official_generator_completed']
    names=['experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md',
        'scripts/z13_common_authorized_recovery1_20261024.py',
        'scripts/z13_common_recovery1_console_20261024.ps1',
        'scripts/z13_launch_common_recovery1_20261024.ps1',
        'scripts/z13_record_common_recovery1_start_20261024.py']
    for stage in (OLD,REC):
        for name in ('protocol.json','controller.json','controller_result.json','completion.json','previous_end_audit.json','price.json','child_cleanup.json','queue.json','console_recovery.json','queue_result.json'):
            if (stage/name).is_file():names.append((stage/name).relative_to(ROOT).as_posix())
        names += [p.relative_to(ROOT).as_posix() for p in (stage/'jobs').glob('*.json')]
    for name in ('HANDOFF.md','val20_common_authorized_failure.json','common_authorized_recovery1_start_verified.json','common_authorized_recovery1_start_checks.json','common_authorized_recovery1_start_vault.json'):
        names.append((W/name).relative_to(ROOT).as_posix())
    deleted=set()
    for state,name in graph_rows():
        if name not in protected:
            if (ROOT/name).is_file():names.append(name)
            elif 'D' in state:names.append(name);deleted.add(name)
    names=sorted(set(names));assert not set(names)&protected and '.gitignore' not in names
    assert all((ROOT/name).is_file() or name in deleted for name in names)
    assert not (ROOT/'.git/index.lock').exists()
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split(chr(0)))-{''}
    assert staged<=set(names)
    for start in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[start:start+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Preserve118completed jobs and queue bounded MASA recovery after GPU availability','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    print(json.dumps(dict(commit=head,files=len(names))),flush=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True);foreign_unchanged()
    write(W/'common_authorized_recovery1_start_publication.json',dict(at=now().isoformat(),commit=head,push_completed=True,paths=names))


if __name__=='__main__':
    if sys.argv[1:]==['--record-checks']:record_checks()
    elif sys.argv[1:]==['--publish']:publish()
    else:raise RuntimeError('Choose explicit record/checks or publication mode')
