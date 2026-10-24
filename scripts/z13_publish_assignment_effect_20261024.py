"""Publish scoped assignment-cache metadata after green checks, preserving edits."""
from pathlib import Path
import os,subprocess,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
W=ROOT/'runs/z13_weekend_20261024';RUN=ROOT/'runs/z13_assignment_effect_20261024'
OLD=ROOT/'runs/z13_unperturbed_readout_20261024'


def handoff(mode):
    from scripts.z13_assignment_effect_20261024 import journal,IDENT
    if mode=='start':
        from scripts.z13_hota_deadline_guard import WindowsProcesses
        c=read(RUN/'controller.json');m=read(RUN/'completion.json');launch=read(RUN/'launch.json');ps=WindowsProcesses()
        assert ps.creation(m['pid'])==m['birth']
        assert ps.creation(c['pid'])==c['birth'] or read(RUN/'controller_result.json')['all_owned_ended']
        assert launch['console_window_handle']!=0
        path=ROOT/'experiments/project_history.json';cat=read(path);card=next(x for x in cat['experiments'] if x['id']==IDENT);card['state']='running';write(path,cat)
        detail=('ТЕКУЩИЙ CPU-этап z13_assignment_effect_20261024:контроллер'+str(c['pid'])+'/birth'+str(c['birth'])+
            ',монитор'+str(m['pid'])+'/birth'+str(m['birth'])+',видимая консольPowerShell7 '+str(launch['console_pid'])+
            '. Исходные сроки '+c['save_at']+' / '+c['hard_stop']+' НЕ ПРОДЛЕВАТЬ. '
            '130trainвидео,безвидео/GPU/ядра/обучения/новойHOTA/dev/val20подбора/75. '
            'Сначала первые2fit20 — контроль точного исходного выхода/прогнозов/идентичностей и цены. '
            'Потом полный кэш:одна строка стоимости от3кандидатов C/D/R1 в полном неизменном сопоставлении,'
            'с точным восстановлением исходного состояния. GTтолько оценивает сохранённые назначения,'
            'интерполяции нет. Смотриstatus,price,videos/*.json,controller_result,monitor_failure. '
            'Послеexit0полного конца дерева монитор --record/проверки/граф/точныйpush;'
            'ошибка не превращается в научный результат. При пройденном оракульном запасе отдельная '
            'регистрация обучения по последствиям;готового обучающего исполнителя ещё нет. '
            'Нельзя назвать этот кэш улучшением HOTA.')
        journal('проверен запуск точного кэша последствий',detail)
    else:
        r=read(RUN/'report.json');detail=('z13_assignment_effect_20261024 ПОЛНОСТЬЮ ЗАВЕРШЁН,exit0/конец дерева. '
            'Источникreport.json/finish_recorded.json,130точных повторов. primary_pass='+str(r['primary_pass'])+
            ';oracle D='+json.dumps(r['oracle_beneficial_fraction']['D'])+'. Это диагностический запас,не HOTA. '
            'Следующее действие — отдельный протокол равного обучения считывателя по фактическим последствиям '
            'только если пройден критерий;не выбирать порог/сид по26. При неуспехе не продвигать эту реализацию. '
            'Данные20val и75 для подбора не использовать.')
    text=('# Актуально — '+now().strftime('%d.%m.%Y %H:%M')+' MSK\n\n'+detail+'\n\n'
        'Собственная ошибочная постановка unperturbed_readout ОСТАНОВЛЕНА23:11:43МСК,exit1/all_owned_ended,12моделей сохранены. '
        'eligibility_audit.json/stopped_recorded.json:исходныйархив19798эпизодов/9888со сдвигом,но '
        'кэш кандидатов УЖЕ исключил все ep_pert=True. Два выбранных режима содержат одинаковые20778известных строк;'
        'это не научный отрицательный результат и не сбойресурса. Ошибка агента явно записана,revisions сохранены. '
        'Не возобновлять и не продвигать12моделей. Источники/логи/первоначальныйexit1непереписаны. '
        'Детекторныйгейт inconclusive сохранён,долгоевычислениеэтойверсии не запускать. '
        'common20 завершён/проверен/опубликован e8d47b7:добавка Dкстендунегарантирована. '
        'Проверки/публикация assignment_effect_'+mode+'_checks.json/_publication.json. '
        'Чужие57графправок/.gitignore сохранить;архитектурной вехи принятого узла нет.\n\n'
        'Ниже прежние передачи сохранены.\n\n')
    path=W/'HANDOFF.md';path.write_text(text+path.read_text(encoding='utf-8'),encoding='utf-8')


def publish(mode):
    from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
    from evaluation.project_history.index import fingerprint
    prefix='assignment_effect_'+mode;assert not (W/(prefix+'_publication.json')).exists()
    foreign=foreign_unchanged();before=fingerprint(ROOT);env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf-8'}
    for label,args in [('catalog',['scripts/validate_project_history.py']),('tests',['-m','pytest',
        'tests/test_project_history.py','tests/test_z13_detection_candidate_gate.py','tests/test_z13_unperturbed_readout.py','tests/test_z13_assignment_effect.py','-q'])]:
        result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env=env,capture_output=True)
        (W/(prefix+'_'+label+'.log')).write_bytes(result.stdout);(W/(prefix+'_'+label+'.stderr.log')).write_bytes(result.stderr)
        assert result.returncode==0,(label,result.returncode)
        print(label+' passed',flush=True)
    assert before==fingerprint(ROOT)
    receipt=W/(prefix+'_owned_graph.json');write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file() and (ROOT/n).suffix=='.md'})
    result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(W/(prefix+'_vault.json')),'--refresh-owned',str(receipt)],cwd=ROOT,env=env,capture_output=True)
    (W/(prefix+'_vault.log')).write_bytes(result.stdout);(W/(prefix+'_vault.stderr.log')).write_bytes(result.stderr);assert result.returncode==0
    foreign_unchanged();write(W/(prefix+'_checks.json'),dict(at=now().isoformat(),catalog_exit=0,tests_exit=0,fingerprint_before=before,fingerprint_after=fingerprint(ROOT),vault_completed=True))
    names=['experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','scripts/z13_assignment_effect_20261024.py',
        'scripts/z13_assignment_effect_console.ps1','scripts/z13_publish_assignment_effect_20261024.py','tests/test_z13_assignment_effect.py']
    for stage in (OLD,RUN):
        for name in ('protocol.json','controller.json','controller_result.json','completion.json','launch.json','report.json',
                     'eligibility_audit.json','stopped_recorded.json','monitor_failure.json','sampling.json','price.json','finish_recorded.json'):
            if (stage/name).is_file():names.append((stage/name).relative_to(ROOT).as_posix())
    if mode=='finish':names += [x.relative_to(ROOT).as_posix() for x in (RUN/'videos').glob('*.json')]
    for name in ('HANDOFF.md',prefix+'_checks.json',prefix+'_owned_graph.json',prefix+'_vault.json'):
        names.append((W/name).relative_to(ROOT).as_posix())
    for state,name in graph_rows():
        if name not in foreign and ((ROOT/name).is_file() or 'D' in state):names.append(name)
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()=='main'
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split(chr(0)))-{''};assert staged<=set(names)
    for i in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    title='Correct invalid crop-sampling premise and start exact assignment-effect supervision' if mode=='start' else 'Record complete assignment-effect supervision with exact baseline replay'
    subprocess.run(['git','commit','--quiet','-m',title,'-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(W/(prefix+'_publication.json'),dict(at=now().isoformat(),commit=head,push_completed=True,paths=names));print(head,flush=True)


def finish():handoff('finish');publish('finish')


if __name__=='__main__':
    assert sys.argv[1]=='--start';handoff('start');publish('start')
