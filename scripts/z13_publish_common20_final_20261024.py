"""Audit, explain and publish the completed matched20 result without rerunning video."""
from pathlib import Path
import json
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now
W = ROOT / 'runs/z13_weekend_20261024'
STAGE = ROOT / 'runs/z13_val20_matched_20261024/common_authorized_recovery2'
PREFIX = 'common20_final'


def audit():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged
    from scripts.z13_val20_common_20261024 import journal
    assert not (W / (PREFIX+'_audit.json')).exists()
    p = read(STAGE/'protocol.json'); verify_sources(p)
    e = read(STAGE/'controller_result.json')
    assert e['exit_code'] == 0 and e['worker_ended'] and e['all_owned_ended'] and not e['remaining']
    identities = dict(e['owned'])
    for name in ('controller.json', 'completion.json'):
        item = read(STAGE/name); identities[str(item['pid'])] = item['birth']
    ps = WindowsProcesses()
    assert all(ps.creation(int(pid)) != birth for pid, birth in identities.items())
    r = read(STAGE/'report.json'); digest = sha(STAGE/'report.json')
    assert len(r['videos']) == 20 and r['videos'] == p['videos'] and r['all_conditions_common']
    assert len(r['pooled']) == 20 and not r['independent_final_test']
    assert read(STAGE/'verified_summary.json')['report_sha256'] == digest == read(STAGE/'finish_recorded.json')['report_sha256']
    jobs = list((STAGE/'jobs').glob('*.json')); assert len(jobs) == 231
    checked = {}
    for marker in jobs:
        for name, expected in read(marker)['outputs'].items():
            path = (ROOT/name).resolve(); path.relative_to(ROOT)
            assert sha(path) == expected, name
            checked[name] = expected
    for name, expected in r['output_sha256'].items():
        assert sha(ROOT/name) == expected, name
    pv = {arm: {v: row['HOTA'] for v, row in rows.items()} for arm, rows in r['per_video'].items()}
    for name, expected in r['paired'].items():
        a, b = name.split('-'); assert paired(pv[a], pv[b]) == expected and expected['n'] == 20
    protected = foreign_unchanged()
    write(W/(PREFIX+'_audit.json'), dict(at=now().isoformat(), report_sha256=digest,
        scientific_sources_verified=len(p['sha256']), verified_jobs=231,
        verified_job_outputs=len(checked), verified_report_outputs=len(r['output_sha256']),
        paired_intervals_recomputed=len(r['paired']), checked_process_identities=len(identities),
        all_identities_ended=True, original_monitor_recomputed_pooled_hota=20,
        protected_foreign_edits=len(protected), no_video_recalculation=True))
    path = ROOT/'experiments/project_history.json'; cat = read(path)
    c = next(x for x in cat['experiments'] if x['id'] == 'z13_val20_common_v1')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    interpretation = ('Полная оценка не подтвердила пользу добавления D к XYWH+MASA: '
        'прежняя D−XYWH средняя по видео −0,00806456,95%ДИ[−0,05454689;0,02558647];'
        'центрированная D−XYWH −0,00990808[−0,02941177;0,00023406]. '
        'Отдельные положительные сравнения: прежняя D−C +0,12010830[0,06161887;0,18657257],17/20;'
        'центрированная D−R1 +0,10853790[0,00447228;0,22112094],13/20. '
        'Источник '+str((STAGE/'report.json').relative_to(ROOT).as_posix())+' → paired. '
        'Эти вторичные сравнения не спасают отсутствие пользы стенду;D−C не выделяет ядро,'
        'поскольку различаются обученные считыватели/политики. Положительный статус карточки означает полноту сравнения.')
    if c['interpretation'] != interpretation:
        c.setdefault('revisions', []).append(dict(date=now().strftime('%Y-%m-%d'), commit=head,
            field='interpretation', old=c['interpretation'], reason='Явная интерпретация проверенных первичных интервалов после полного итога.'))
        c['interpretation'] = interpretation
    source = (STAGE/'report.json').relative_to(ROOT).as_posix()
    for name, title in (
        ('legacyD_masa-legacyC_masa', 'Прежняя D лучше системы со штатным ядром C;пользы сверх стенда не подтверждено'),
        ('centerD_masa-centerR1_masa', 'Центрированная D лучше временного входа R1;пользы сверх стенда не подтверждено')):
        assert r['paired'][name]['ci95'][0] > 0
        ident = 'z13_val20_common_v1_'+name
        if not any(x['id'] == ident for x in c['confirmed_results']):
            c['confirmed_results'].append(dict(id=ident, title=title, category='tracker', metric='HOTA',
                unit='points', comparison=name, subset='Фиксированные20ранее использованныхval',
                conditions=p['settings'], limitations=p['limitations']+' '+p['controls']+
                ' Вторичное системное сравнение;преимущество D над стендом без мухи не подтверждено.',
                evidence=dict(source=source, sha256=digest, locator=['paired', name])))
    write(path, cat)
    journal('проверен полный итог20val:польза D стенду не подтверждена',
        '*Получено.* '+interpretation+'\n\n*Проверка.* '+str((W/(PREFIX+'_audit.json')).relative_to(ROOT).as_posix())+
        ':231маркер,400выходов отчёта и все парные интервалы проверены;все зарегистрированные PID/birth закончены. '
        'Исходный монитор уже независимо пересчитал20общих HOTA. Видео повторно не обрабатывались.\n\n'
        '*Решение.* Сохранить полный исход без подбора на20val. Два вторичных положительных сравнения '
        'отразить отдельно с ограничениями. Прежняя D к общей базе +0,00607238%,центрированная −0,00867605%;'
        'это проценты общей HOTA,а не средние разности по видео. Доля потолка неизвестна.')
    handoff = W/'HANDOFF.md'
    text = ('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
        'common_authorized_recovery2 ЗАВЕРШЁН22:17:42,exit0/worker_ended/all_owned_ended. '
        '231задание и400выходов отчёта проверены,все парные интервалы повторены. '
        'Итог report.json SHA256='+digest+'. Монитор сохранил карточку/журнал и прошёл58тестов/каталог/граф. '
        'Дополнительная интерпретация и два ограниченных положительных системных контраста внесены отдельной записью/revisions.\n\n'
        +interpretation+'\n\nНе возобновлять завершённый расчёт. Не подбирать параметры на20val,'
        'не открывать75,не продвигать проваленные descriptor/loss-версии. Следующее действие выбирать '
        'на train по действующему плану;данные20val служат результатом проверки,а не выбором порога. '
        'Архитектурных узлов этот аудит не меняет. Проверки/публикация:'+PREFIX+'_checks.json,'+PREFIX+'_publication.json.\n\n'
        'Ниже прежние передачи сохранены.\n\n')
    handoff.write_text(text+handoff.read_text(encoding='utf-8'), encoding='utf-8')
    print('231jobs,400outputs,allpaired and ended identities verified', flush=True)


def checks():
    from evaluation.project_history.index import fingerprint
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged, graph_rows
    foreign = foreign_unchanged(); before = fingerprint(ROOT)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'PYTHONIOENCODING':'utf-8'}
    for label, args in [('catalog',['scripts/validate_project_history.py']),
        ('tests',['-m','pytest','tests/test_project_history.py','-q'])]:
        result = subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args], cwd=ROOT, env=env, capture_output=True)
        (W/(PREFIX+'_'+label+'.log')).write_bytes(result.stdout)
        (W/(PREFIX+'_'+label+'.stderr.log')).write_bytes(result.stderr)
        assert result.returncode == 0, (label,result.returncode)
        print(label+' passed', flush=True)
    assert fingerprint(ROOT) == before
    receipt = W/(PREFIX+'_owned_graph.json')
    write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file() and (ROOT/n).suffix=='.md'})
    result = subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py',
        '--report',str(W/(PREFIX+'_vault.json')),'--refresh-owned',str(receipt)],cwd=ROOT,env=env,capture_output=True)
    (W/(PREFIX+'_vault.log')).write_bytes(result.stdout)
    (W/(PREFIX+'_vault.stderr.log')).write_bytes(result.stderr)
    assert result.returncode == 0
    foreign_unchanged()
    write(W/(PREFIX+'_checks.json'),dict(at=now().isoformat(),catalog_exit=0,tests_exit=0,
        input_fingerprint_before=before,input_fingerprint_after=fingerprint(ROOT),vault_completed=True))


def publish():
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged, graph_rows
    foreign = foreign_unchanged(); check = read(W/(PREFIX+'_checks.json'))
    assert check['catalog_exit'] == check['tests_exit'] == 0 and check['vault_completed']
    assert read(W/(PREFIX+'_audit.json'))['report_sha256'] == sha(STAGE/'report.json')
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip() == 'main'
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip() == 'https://gitlab.com/aziteam-group/drbdiss.git'
    names = ['experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md',Path(__file__).relative_to(ROOT).as_posix()]
    for name in ('protocol.json','controller.json','controller_result.json','completion.json','report.json',
                 'verified_summary.json','results.md','finish_recorded.json'):
        names.append((STAGE/name).relative_to(ROOT).as_posix())
    names += [x.relative_to(ROOT).as_posix() for x in (STAGE/'jobs').glob('*.json')]
    for name in ('HANDOFF.md',PREFIX+'_audit.json',PREFIX+'_checks.json',PREFIX+'_owned_graph.json',PREFIX+'_vault.json',
                 'val20_common_authorized_recovery2_verified.json','val20_common_authorized_recovery2_vault.json','val20_common_authorized_recovery2_complete.json'):
        names.append((W/name).relative_to(ROOT).as_posix())
    for state,name in graph_rows():
        if name not in foreign and ((ROOT/name).is_file() or 'D' in state): names.append(name)
    names = sorted(set(names)); assert not set(names)&foreign and '.gitignore' not in names
    staged = set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split(chr(0)))-{''}
    assert staged <= set(names)
    for start in range(0,len(names),25):
        subprocess.run(['git','add','-f','--',*names[start:start+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Verify full matched20 FlyVis result and separate secondary gains from absent stand benefit',
        '-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged(); write(W/(PREFIX+'_publication.json'),dict(at=now().isoformat(),commit=head,push_completed=True,paths=names))
    print(head,flush=True)


if __name__ == '__main__':
    {'--audit':audit,'--checks':checks,'--publish':publish}[sys.argv[1]]()
