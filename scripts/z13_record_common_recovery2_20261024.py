"""Record the diagnostic-backed continuation and publish only owned metadata."""
from pathlib import Path
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
W=ROOT/'runs/z13_weekend_20261024'
BASE=ROOT/'runs/z13_val20_matched_20261024'
REC=BASE/'common_authorized_recovery2';DIAG=BASE/'signature_stall_diagnostic'
PREFIX='common_authorized_recovery2_start'


def record_checks():
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_val20_common_20261024 import journal
    p=read(REC/'protocol.json');verify_sources(p);ps=WindowsProcesses()
    c=read(REC/'controller.json');m=read(REC/'completion.json');launch=read(REC/'launch.json')
    assert ps.creation(c['pid'])==c['birth'] and ps.creation(m['pid'])==m['birth']
    assert c['save_at']==p['save_at']=='2026-10-25T06:00:07.166100+03:00'
    assert c['hard_stop']==p['hard_stop']=='2026-10-25T06:00:27.166100+03:00'
    assert launch['console_window_handle']!=0
    write(W/(PREFIX+'_verified.json'),dict(at=now().isoformat(),controller=c,monitor=m,
        console=launch,diagnostic=read(DIAG/'report.json'),
        scientific_sources_verified=True,original_deadline_unchanged=True))
    journal('проверено продолжение после инструментального повтора R1',
        'common_authorized_recovery2/controller.json иcompletion.json:контроллерPID'+
        str(c['pid'])+' инаблюдательPID'+str(m['pid'])+' проверены по времени создания. '
        'Отдельная консольPowerShell7 PID'+str(launch['console_pid'])+' открыта с '
        'ненулевымWindowHandle. Все замороженные источники/веса/данные проверены;'
        'источникcommon_authorized_recovery2_start_verified.json. '
        'Готовое включая инструментальный R1неповторяется;новые исходыHOTA ещё не получены.')
    handoff=W/'HANDOFF.md'
    header=('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
        'ТЕКУЩИЙ расчёт common_authorized_recovery2, '
        'scripts/z13_common_authorized_recovery2_20261024.py. Контроллер'+
        str(c['pid'])+'/birth'+str(c['birth'])+',монитор'+str(m['pid'])+'/birth'+
        str(m['birth'])+'. Видимая консольPowerShell7 PID'+str(launch['console_pid'])+
        ' проверена,launch.json. Предыдущая common_authorized_recovery1 не текущая. '
        'Сохранение25.10 06:00:07.166100МСК/полное завершение06:00:27.166100 '
        'НЕПРОДЛЕНЫ;общее окно26.10 не отменяет меньший срок.\n\n'
        'Предыдущий запуск остановился18:57:02после600с без прогресса на '
        'centerR1/b23adb0d-72704b27,последние950/1204кадра. Все егоPID/birth '
        'окончены;180готовых заданий/12полныхвидео проверены. Это повторный '
        'случай отсутствия прогресса,поэтому слепого повторного запуска не было. '
        'signature_stall_diagnostic выполнен в отдельном600сокне внутри прежнего '
        'срока:тот же builder,только иной каталогвывода,снимки Python/нативного '
        'стека каждые45с. Полные1204кадра/6504строки готовы за122,547с с '
        'наблюдением;exit0/all_owned_ended. Снимки показывают нормальную обработку '
        'изображения/NumPy в разные моменты;прежнее зависание не воспроизведено,'
        'его причина НЕУСТАНОВЛЕНА. Не выдавать старый CUDA-стек другого задания '
        'за доказательство этой причины. Диагностический NPZ проверен и сохранён '
        'как законченный результат отсутствовавшего задания;его не повторять.\n\n'
        'До нового старта181маркер/хеш сохранены. Остались8видео включая один '
        'частичный;цена по первоначальным910,516с:2×8×910,516+900=15468,256с,'
        'ниже оставшегося исходного окна. Нулевое время пропуска готового не '
        'используется как новая цена. ПередGPU проверяются прежние ресурсные '
        'условия отсутствияaces,8192МиБсвободно,<=30%загрузки в течение30с;'
        'если ресурсное ожидание исчерпает исходный бюджет,не продлевать.\n\n'
        'Служебное отличиеrecovery2:при120/240/360/480с без прежнего обновления '
        'собирается стек только точного собственного childPID/birth,без локальных '
        'переменных. Файлы REC/stacks/. Прежний600с сторож/явное завершение своего '
        'дерева неизменны. При следующем сбое исследовать эти снимки;'
        'автоматического повторения упавшего задания нет. Все научные настройки/'
        'данные/веса/порогисигнала и разрешённый координатный допуск прежние.\n\n'
        'Монитор этого же файла --monitor после полного exit0/конца дерева '
        'сам --record:231маркер/первичные строки/20общихHOTA/все парные интервалы,'
        'карточка сrevisions/журнал/каталог/тесты/граф. Итог REC/report.json,'
        'verified_summary.json,results.md,finish_recorded.json. Ошибки '
        'W/val20_common_authorized_recovery2_failure.json и _monitor_failure.json. '
        'Старые recovery1_failure/common_authorized_failure и их журналы не '
        'новые ошибки и не переписаны. Новых выводов о пользе FlyVis до полного '
        'сравнения нет;не продвигать отрицательные descriptor/loss-версии,'
        'не открывать75,не загружать. Архитектурного изменения нет.\n\n'
        'Проверки/публикация:'+PREFIX+'_checks.json и '+PREFIX+'_publication.json. '
        'Чужие57графправок/.gitignore сохранять.\n\n'
        'Ниже прежние передачи сохранены.\n\n')
    handoff.write_text(header+handoff.read_text(encoding='utf-8'),encoding='utf-8')
    from evaluation.project_history.index import fingerprint
    before=fingerprint(ROOT)
    for label,args in [('catalog',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','tests/test_z13_coordinate_replay_guard.py','tests/test_z13_status_read_retry.py','-q'])]:
        result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf-8'},capture_output=True)
        (W/(PREFIX+'_'+label+'.log')).write_bytes(result.stdout)
        (W/(PREFIX+'_'+label+'.stderr.log')).write_bytes(result.stderr)
        assert result.returncode==0,(label,result.returncode)
        print(label+' passed',flush=True)
    after=fingerprint(ROOT);assert before==after
    write(W/(PREFIX+'_checks.json'),dict(at=now().isoformat(),catalog_exit=0,tests_exit=0,fingerprint_before=before,fingerprint_after=after))


def publish():
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    protected=foreign_unchanged();checks=read(W/(PREFIX+'_checks.json'))
    assert checks['catalog_exit']==checks['tests_exit']==0
    assert '66 passed' in (W/(PREFIX+'_tests.log')).read_text(encoding='utf-8')
    assert read(W/(PREFIX+'_vault.json'))['official_generator_completed']
    names=['experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md',
        'scripts/z13_signature_stall_diagnostic_20261024.py',
        'scripts/z13_common_authorized_recovery2_20261024.py',
        'scripts/z13_record_common_recovery2_20261024.py']
    for stage in (BASE/'common_authorized_recovery1',DIAG,REC):
        for name in ('protocol.json','controller.json','controller_result.json','completion.json','price.json','child_cleanup.json','launch.json','report.json','job_result.json','stack_samples.json','actual_job.json'):
            if (stage/name).is_file():names.append((stage/name).relative_to(ROOT).as_posix())
        names += [p.relative_to(ROOT).as_posix() for p in (stage/'jobs').glob('*.json')]
    for name in ('HANDOFF.md','val20_common_authorized_recovery1_failure.json',PREFIX+'_verified.json',PREFIX+'_checks.json',PREFIX+'_vault.json'):
        names.append((W/name).relative_to(ROOT).as_posix())
    deleted=set()
    for state,name in graph_rows():
        if name not in protected:
            if (ROOT/name).is_file():names.append(name)
            elif 'D' in state:names.append(name);deleted.add(name)
    names=sorted(set(names));assert not set(names)&protected and '.gitignore' not in names
    assert all((ROOT/name).is_file() or name in deleted for name in names)
    assert not (ROOT/'.git/index.lock').exists()
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split(chr(0)))-{''};assert staged<=set(names)
    for start in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[start:start+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Diagnose unfinished R1 and retain181jobs with stack capture before future timeouts','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();print(head,flush=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True);foreign_unchanged()
    write(W/(PREFIX+'_publication.json'),dict(at=now().isoformat(),commit=head,push_completed=True,paths=names))


if __name__=='__main__':
    if sys.argv[1:]==['--record-checks']:record_checks()
    elif sys.argv[1:]==['--publish']:publish()
    else:raise RuntimeError('Choose explicit mode')
