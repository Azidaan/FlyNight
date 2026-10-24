"""Exact-file publication and protected graph update for a bounded readout pilot."""
from pathlib import Path
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
W=ROOT/'runs/z13_weekend_20261024'
RUN=ROOT/'runs/z13_unperturbed_readout_20261024'
GATE=ROOT/'runs/z13_detection_candidate_gate_20261024'


def startup():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_unperturbed_readout_20261024 import journal,IDENT
    p=read(RUN/'protocol.json');c=read(RUN/'controller.json');m=read(RUN/'completion.json');launch=read(RUN/'launch.json')
    ps=WindowsProcesses();assert ps.creation(m['pid'])==m['birth']
    assert ps.creation(c['pid'])==c['birth'] or read(RUN/'controller_result.json')['all_owned_ended']
    assert launch['console_window_handle']!=0
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(x for x in cat['experiments'] if x['id']==IDENT);card['state']='running'
    gate=next(x for x in cat['experiments'] if x['id']=='z13_detection_candidate_gate_v1')
    if not any(x['field']=='summary' for x in gate.get('revisions',[])):
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
        gate.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field='summary',
            old='Диагностическая CPU-проверка до600с зарегистрирована до расчёта.',reason='Сохранена исходная постановка перед итоговым summary;начальное значение изprepare.'))
    write(cp,cat)
    journal('проверен старт согласования режима обучающих вырезок',
        'КонтроллерPID'+str(c['pid'])+'/birth'+str(c['birth'])+',наблюдательPID'+str(m['pid'])+
        ',отдельная консольPowerShell7 PID'+str(launch['console_pid'])+' с ненулевым окном проверены. '
        'CPU1800с,до'+c['save_at']+' сохранение,до'+c['hard_stop']+' конец своего дерева. '
        'Детекторный oracle-гейт завершёнexit0/all_owned_ended за76с,не принят: '
        'runs/z13_detection_candidate_gate_20261024/report.json: +1,94954573п.п.[−7,14810586;9,94414292],'
        '26видео. Это неопределённость,не доказательство бесполезности детекций. Дорогого извлечения нет. '
        'Источник нового запуска runs/z13_unperturbed_readout_20261024/protocol.json;готовое не повторяется.')
    handoff=W/'HANDOFF.md';header=('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
        'common20 завершён/проверен и опубликован e8d47b7;нового выигрыша FlyVis над стендом не подтверждено. '
        'Подробности common20_final_audit.json/common_authorized_recovery2/report.json.\n\n'
        'detection_candidate_gate завершён76с,exit0/all_owned_ended;3детекции oracle против3пиковD '
        '+1,94954573п.п.[−7,14810586;9,94414292],inconclusive. Не продвигать эту реализацию к извлечению. '
        'Это оракул на train26,не HOTA и не эффект мухи.\n\n'
        'ТЕКУЩИЙ новый CPU-пилот z13_unperturbed_readout_20261024. Контроллер'+str(c['pid'])+
        '/birth'+str(c['birth'])+',монитор'+str(m['pid'])+'/birth'+str(m['birth'])+
        ',видимая PowerShell7консоль'+str(launch['console_pid'])+'. Сроки '+c['save_at']+' / '+c['hard_stop']+
        ' НЕ ПРОДЛЕВАТЬ. 36моделей,безGPU/обученияядра/загрузок/dev/val20подбора/75. '
        'Учится тот же считыватель на исходных unperturbedэпизодах против смешанной выборки равного размера по видео/kind,'
        'те же1150шагов/145950показов/3сида. Все C/D/R1/геометрия-контроли. Критерии вprotocol доисходов,'
        'основная оценка на unperturbedчасти26train,не независимый перенос. '
        'Известно различие аугментации,вред пока гипотеза.\n\n'
        'Монитор --monitor ждёт controller_result exit0/конец дерева И публикацию старта,затем '
        '--record:36моделей/строки/интервалы/источники,карточка сrevisions/журнал,каталог/тесты/сохраняющийграф,'
        'точный коммит/push. Готовые переходы локально,без вызовов модели. Смотриstatus.json,price.json,'
        'reproduction.json,report.json,finish_recorded.json,monitor_failure.json иreadout_sampling_finish_publication.json. '
        'При неуспехе не продвигать к HOTA;следующий механизм выбирать отдельно. Чужие57графправок/.gitignore сохранить.\n\n'
        'Ниже прежние передачи сохранены.\n\n')
    handoff.write_text(header+handoff.read_text(encoding='utf-8'),encoding='utf-8')


def checks_and_publish(mode):
    assert mode in ('start','finish');prefix='readout_sampling_'+mode
    assert not (W/(prefix+'_publication.json')).exists()
    if mode=='finish':
        r=read(RUN/'report.json');handoff=W/'HANDOFF.md'
        header=('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'
            'z13_unperturbed_readout_20261024 завершён exit0/worker_ended/all_owned_ended. '
            '36моделей и3сида сохранены,первоисточники/строки/интервалы проверены;'
            'finish_recorded.json. primary_pass='+str(r['primary_pass'])+'. '
            'Основное сравнение confirmed-yield cleanD-mixedD: '+str(r['paired_confirmed_yield']['clean_D_activity-mixed_D_activity'])+'. '
            'Источник runs/z13_unperturbed_readout_20261024/report.json;это не HOTA,'
            'не подтверждение причинности аугментации и не гарантия точности на неизвестных. '
            'Приfalse не продвигать к дорогому переносу;следующий узел выбирает агент по действующему плану. '
            'Приtrue нужна отдельная зарегистрированная фиксированная проверка применения,не автоматический выбор порога. '
            'Данные20val не использовать дляподбора;75неоткрывать. '
            'Проверки/публикация readout_sampling_finish_checks.json/readout_sampling_finish_publication.json. '
            'Прежнийдетекторныйгейт inconclusive сохранён;common20 окончен/опубликован. '
            'Чужие57правокграфа/.gitignore сохранять.\n\nНиже прежние передачи сохранены.\n\n')
        handoff.write_text(header+handoff.read_text(encoding='utf-8'),encoding='utf-8')
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    from evaluation.project_history.index import fingerprint
    foreign=foreign_unchanged();before=fingerprint(ROOT)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf-8'}
    for label,args in [('catalog',['scripts/validate_project_history.py']),('tests',['-m','pytest',
        'tests/test_project_history.py','tests/test_z13_detection_candidate_gate.py','tests/test_z13_unperturbed_readout.py','-q'])]:
        r=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env=env,capture_output=True)
        (W/(prefix+'_'+label+'.log')).write_bytes(r.stdout);(W/(prefix+'_'+label+'.stderr.log')).write_bytes(r.stderr)
        assert r.returncode==0,(label,r.returncode)
        print(label+' passed',flush=True)
    assert fingerprint(ROOT)==before
    receipt=W/(prefix+'_owned_graph.json')
    write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file() and (ROOT/n).suffix=='.md'})
    r=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(W/(prefix+'_vault.json')),
        '--refresh-owned',str(receipt)],cwd=ROOT,env=env,capture_output=True)
    (W/(prefix+'_vault.log')).write_bytes(r.stdout);(W/(prefix+'_vault.stderr.log')).write_bytes(r.stderr)
    assert r.returncode==0
    foreign_unchanged();write(W/(prefix+'_checks.json'),dict(at=now().isoformat(),catalog_exit=0,tests_exit=0,
        fingerprint_before=before,fingerprint_after=fingerprint(ROOT),vault_completed=True))
    names=['experiments/project_history.json','experiments/metrics_glossary.json','dissertation/ЕДИНЫЙ_ПЛАН.md',
        'scripts/z13_detection_candidate_gate_20261024.py','scripts/z13_detection_candidate_gate_console.ps1','tests/test_z13_detection_candidate_gate.py',
        'scripts/z13_unperturbed_readout_20261024.py','scripts/z13_unperturbed_readout_console.ps1','tests/test_z13_unperturbed_readout.py',
        'scripts/z13_publish_readout_sampling_20261024.py']
    for stage in (GATE,RUN):
        for name in ('protocol.json','controller.json','controller_result.json','completion.json','launch.json',
            'report.json','finish_recorded.json','sampling.json','price.json','reproduction.json','cache_manifest.json','seed_metrics.json'):
            if (stage/name).is_file():names.append((stage/name).relative_to(ROOT).as_posix())
        if mode=='finish':
            names += [x.relative_to(ROOT).as_posix() for pattern in ('*/seed*/training.json','*/seed*/done.json') for x in stage.glob(pattern)]
    # Full per-episode predictions and binary models stay local; primary paired
    # tables,per-video exact counts,source hashes and reproduction are published.
    for name in ('HANDOFF.md',prefix+'_checks.json',prefix+'_owned_graph.json',prefix+'_vault.json'):
        names.append((W/name).relative_to(ROOT).as_posix())
    for state,name in graph_rows():
        if name not in foreign and ((ROOT/name).is_file() or 'D' in state):names.append(name)
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()=='main'
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split(chr(0)))-{''}
    assert staged<=set(names),'Unrelated staged files'
    for start in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[start:start+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    title='Record inconclusive detection gate and start matched unperturbed readout sampling' if mode=='start' else 'Record matched unperturbed readout sampling with complete controls and frozen sources'
    subprocess.run(['git','commit','--quiet','-m',title,'-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(W/(prefix+'_publication.json'),dict(at=now().isoformat(),commit=head,push_completed=True,paths=names))
    print(head,flush=True)


if __name__=='__main__':
    if sys.argv[1]=='--start':startup();checks_and_publish('start')
    else:raise ValueError(sys.argv[1])
