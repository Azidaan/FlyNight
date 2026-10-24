"""Record the authorized restart, validate fixed inputs, and publish exact files."""
from pathlib import Path
import inspect
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now
W = ROOT / 'runs/z13_weekend_20261024'
REC = ROOT / 'runs/z13_val20_matched_20261024/common_authorized'


def verify_live():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    ps = WindowsProcesses()
    controller = read(REC / 'controller.json')
    monitor = read(REC / 'completion.json')
    assert ps.creation(controller['pid']) == controller['birth']
    assert ps.creation(controller['worker_pid']) == controller['worker_birth']
    assert ps.creation(monitor['pid']) == monitor['birth']
    protocol = read(REC / 'protocol.json')
    verify_sources(protocol)
    assert read(W / 'val20_replay_authorized.json')['status'] == 'approved'
    assert protocol['save_at'] == controller['save_at'] == '2026-10-25T06:00:07.166100+03:00'
    assert protocol['hard_stop'] == controller['hard_stop'] == '2026-10-25T06:00:27.166100+03:00'
    return controller, monitor


def record():
    from scripts import z13_val20_common_authorized_20261024 as R
    from scripts import z13_val20_common_finish_20261024 as F
    controller, monitor = verify_live()
    # Compile the delayed evaluation/record adapters now, without executing them.
    source = inspect.getsource(R.evaluate)
    source = source.replace("namespace['evaluate']()", "return namespace['evaluate']")
    ns = dict(vars(R)); exec(compile(source, __file__, 'exec'), ns)
    assert callable(ns['evaluate']())
    source = inspect.getsource(R.record)
    source = source.replace("namespace['record']()", "return namespace['record']")
    ns = dict(vars(R)); exec(compile(source, __file__, 'exec'), ns)
    assert callable(ns['record']())
    launch = json.loads((REC / 'launch.json').read_text(encoding='utf-8-sig'))
    assert launch['console_window_handle'] != 0
    write(W / 'common_authorized_start_verified.json', dict(at=now().isoformat(),
        controller=controller, monitor=monitor, visible_console_at_launch=launch,
        protocol_sha256=sha(REC / 'protocol.json'), frozen_sources_verified=True,
        delayed_adapters_compiled=True, original_deadline_unchanged=True,
        scientific_settings_unchanged=True,
        launch_script_sha256=sha(ROOT / 'scripts/z13_launch_common_authorized_20261024.ps1')))
    handoff = W / 'HANDOFF.md'
    prefix = ('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
        'АВТОР ЯВНО РАЗРЕШИЛ служебный допуск:«Разрешаю этот служебный допуск». '
        'val20_replay_authorized.json и pending_author_decision.json/status=approved. '
        'Прежний pending сохранён в .before_approval.json. '
        'Только абсолютная разница координат<=0,0001px;число строк/кадры/ID/классы/'
        'уверенности точны. Старый непройденный побитовый контроль и его ошибка '
        'не переписаны;в карточке он исторический met=false.\n\n'
        'ТЕКУЩИЙ расчёт common_authorized, scripts/z13_val20_common_authorized_20261024.py. '
        'Контроллер '+str(controller['pid'])+'/birth'+str(controller['birth'])+
        ';workerlauncher '+str(controller['worker_pid'])+'/birth'+str(controller['worker_birth'])+
        ';монитор '+str(monitor['pid'])+'/birth'+str(monitor['birth'])+'. '
        'Видимая консоль '+str(launch['console_pid'])+' с ненулевым окном проверена при запуске. '
        'Исходный86400с срок от24.10 06:00 НЕПРОДЛЁН:сохранение '+controller['save_at']+
        ',конец дерева '+controller['hard_stop']+'. Общий авторский срок26.10 07:40 '
        'не отменяет этот меньший срок.\n\n'
        'До запуска проверены86готовых заданий:80native сигнатур,5прежних точных '
        'native/MASAконтролей,ранее полученный legacyD теперь проверен разрешённым '
        'координатным критерием без GPU-повтора. Осталось5контролей+6вариантов×20видео '
        '+20MASA. 4native/MASA/их готовые оценки используются без перезаписи. '
        'В каждом новом MASA-маркере хеши всех10вариантов;досчитываются только6. '
        'Всего231маркер. Первый новый полный ролик:2×фактическое время6недостающих '
        'сигнатур+MASA×19+900с должно поместиться в старый остаток;иначе остановить '
        'без продления и без вывода об эффективности.\n\n'
        'Монитор этого же файла --monitor после полного exit0/конца дерева '
        'сам --record,проверяет231маркер/первичные строки/20общих HOTA/все парные '
        'интервалы,карточку сrevisions,каталог/тесты/граф. '
        'Итог common_authorized/report.json,verified_summary.json,results.md,'
        'finish_recorded.json. Ошибки W/val20_common_authorized_failure.json '
        'или val20_common_authorized_monitor_failure.json;старые common_failure '
        'не новая ошибка. Сторож600с,ограниченные повторы чтения status и '
        'явная регистрацияchildPID/birth сохранены.\n\n'
        'Новые варианты descriptor/loss,непрошедшие критерии,не используются. '
        'Это оценка сохранённых версий,не новое обучение. Не менять веса/данные/'
        'пороги;не открывать75 и не загружать. D-C не изолирует одно ядро,'
        'C имеет обученный считыватель,контроли B/R1 неравны по ёмкости. '
        '20val не нетронутый финальный тест;старые7 и20 в разных таблицах. '
        'Чужие57графправок/.gitignore сохранять. Архитектурного изменения нет.\n\n'
        'Старт/проверки/публикация:common_authorized_start_verified.json,'
        'common_authorized_start_checks.json,common_authorized_start_publication.json.\n\n'
        'Ниже прежние передачи сохранены.\n\n')
    handoff.write_text(prefix+handoff.read_text(encoding='utf-8'), encoding='utf-8')
    print('Live identities/frozen inputs/delayed adapters verified;handoff recorded.', flush=True)


def checks():
    from evaluation.project_history.index import fingerprint
    before = fingerprint(ROOT)
    for label, args in (
        ('catalog', ['scripts/validate_project_history.py']),
        ('tests', ['-m', 'pytest', 'tests/test_project_history.py',
            'tests/test_z13_coordinate_replay_guard.py', 'tests/test_z13_status_read_retry.py', '-q'])):
        result = subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), *args], cwd=ROOT,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'PYTHONIOENCODING':'utf-8'}, capture_output=True)
        (W / ('common_authorized_start_'+label+'.log')).write_bytes(result.stdout)
        (W / ('common_authorized_start_'+label+'.stderr.log')).write_bytes(result.stderr)
        assert result.returncode == 0, (label, result.returncode)
        print(label+' passed', flush=True)
    after = fingerprint(ROOT)
    assert before == after
    write(W / 'common_authorized_start_checks.json', dict(at=now().isoformat(),
        catalog_exit=0, tests_exit=0, fingerprint_before=before, fingerprint_after=after))


def publish():
    evidence = read(W / 'common_authorized_start_checks.json')
    assert evidence['catalog_exit'] == evidence['tests_exit'] == 0
    assert '66 passed' in (W / 'common_authorized_start_tests.log').read_text(encoding='utf-8')
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged, graph_rows
    protected = foreign_unchanged()
    assert read(W / 'common_authorized_start_vault.json')['official_generator_completed']
    names = ['experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md',
        'scripts/z13_coordinate_replay_guard.py','tests/test_z13_coordinate_replay_guard.py',
        'scripts/z13_val20_common_authorized_20261024.py',
        'scripts/z13_launch_common_authorized_20261024.ps1',
        'scripts/z13_record_common_authorized_start_20261024.py']
    for name in ('protocol.json','controller.json','completion.json','launch.json'):
        names.append((REC / name).relative_to(ROOT).as_posix())
    names += [p.relative_to(ROOT).as_posix() for p in (REC / 'jobs').glob('*.json')]
    for name in ('HANDOFF.md','val20_replay_authorized.json',
        'val20_replay_pending_author_decision.json',
        'val20_replay_pending_author_decision.before_approval.json',
        'common_authorized_start_verified.json','common_authorized_start_checks.json',
        'common_authorized_start_vault.json'):
        names.append((W / name).relative_to(ROOT).as_posix())
    deleted = set()
    for state, name in graph_rows():
        if name not in protected:
            if (ROOT / name).is_file():
                names.append(name)
            elif 'D' in state:
                names.append(name)
                deleted.add(name)
    names = sorted(set(names))
    assert not set(names) & protected and '.gitignore' not in names
    assert all((ROOT / name).is_file() or name in deleted for name in names)
    assert not (ROOT / '.git/index.lock').exists()
    staged = set(subprocess.check_output(['git','diff','--cached','--name-only','-z'], cwd=ROOT).decode().split(chr(0))) - {''}
    assert staged <= set(names)
    for start in range(0,len(names),25):
        subprocess.run(['git','add','-f','--',*names[start:start+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m',
        'Resume matched FlyVis evaluation with explicitly authorized coordinate replay guard',
        '-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    print(json.dumps(dict(commit=head,files=len(names))),flush=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged()
    write(W / 'common_authorized_start_publication.json',dict(at=now().isoformat(),commit=head,push_completed=True,paths=names))


if __name__ == '__main__':
    if sys.argv[1:] == ['--record-checks']:
        record();checks()
    elif sys.argv[1:] == ['--publish']:
        publish()
    else:
        raise RuntimeError('Choose --record-checks or --publish')
