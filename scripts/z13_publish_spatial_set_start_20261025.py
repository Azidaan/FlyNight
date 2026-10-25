"""Save and publish the registered spatial-set experiment, preserving foreign edits."""
from pathlib import Path
import os,sys,subprocess,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,now,sha
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_spatial_set_readout_20261025 import RUN,IDENT,journal,handoff
BASE=RUN
for version in ('recovery2','recovery1'):
    if (BASE/version/'protocol.json').exists():RUN=BASE/version;break


def record():
    assert not (RUN/'start_recorded.json').exists();p=read(RUN/'protocol.json');verify_sources(p)
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    ps=WindowsProcesses();c=read(RUN/'controller.json');f=read(RUN/'features/controller.json')
    assert ps.creation(c['pid'])==c['birth'] and ps.creation(f['pid'])==f['birth']
    # An initial phase status is service metadata only; never overwrite a worker update.
    path=RUN/'features/status.json'
    if not path.exists():
        try:
            with path.open('x',encoding='utf8') as out:json.dump(dict(state='running',stage='Инициализация входов; число групп ещё не определено',updated=f['started']),out,ensure_ascii=False,indent=1)
        except FileExistsError:pass
    cat=read(ROOT/'experiments/project_history.json');card=next(x for x in cat['experiments'] if x['id']==IDENT);card['state']='running';write(ROOT/'experiments/project_history.json',cat)
    journal('запуск GPU получения пространственных входов','*Сделано.* По просьбе автора «Давай гпу» запущена скрытая очередь '+IDENT+'. Независимый GPU-контроллер PID '+str(f['pid'])+', время создания '+str(f['birth'])+'; предел сохранения '+f['save_at']+', остановка '+f['hard_stop']+'. Общее дерево сохраняется до '+c['save_at']+' и заканчивается до '+c['hard_stop']+'.\n\n*Проверка.* PID и время создания контроллеров совпали. Первичные файлы: '+RUN.relative_to(ROOT).as_posix()+'/controller.json и features/controller.json. Процент GPU-этапа до получения первого статуса не определён. Следующая CPU-ступень допускается только после завершения и проверок GPU. Научного результата пока нет. Архитектурную веху объединить с итогом нового узла, источники не пересчитывать ради схемы.')
    handoff('ТЕКУЩИЙ '+IDENT+' в '+RUN.relative_to(ROOT).as_posix()+', запуск '+c['started']+', root '+str(c['pid'])+'/birth'+str(c['birth'])+', features '+str(f['pid'])+'/birth'+str(f['birth'])+'. Проверять реальные дочерние PID/CPU и свежие chunk_*.json, не одно наличие процесса. Features save '+f['save_at']+' hard '+f['hard_stop']+'; общий save '+c['save_at']+' hard '+c['hard_stop']+'. features→CPUtraining10800с автоматически только при проверках/price gate. Root status не заменяет features/status или training/status. Результат требует независимого разбора всех строк, хешей и парных интервалов; карточка/revisions/журнал/тесты/preserving graph/коммит/push. Последний969d93c7опубликован. Всё скрыто,57чужихправок/.gitignore сохранить; старые бюджеты не продлены. Новая пространственная схема пока не имеет итога.')
    write(RUN/'start_recorded.json',dict(at=now().isoformat(),root=c,features=f,identities_verified=True))


def checks():
    assert not (RUN/'start_checks.json').exists();foreign_unchanged()
    from evaluation.project_history.index import fingerprint
    before=fingerprint(ROOT);results={}
    for label,args in [('validator',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','tests/test_z13_spatial_set_readout.py','-q'])]:
        c=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf8'},capture_output=True)
        (RUN/('start_'+label+'.stdout.log')).write_bytes(c.stdout);(RUN/('start_'+label+'.stderr.log')).write_bytes(c.stderr)
        results[label]=dict(exit_code=c.returncode,tail=c.stdout.decode('utf8',errors='replace')[-1200:]);assert c.returncode==0,results[label];print(label+' passed',flush=True)
    assert before==fingerprint(ROOT)
    c=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'start_vault.json')],cwd=ROOT,capture_output=True)
    (RUN/'start_vault.stdout.log').write_bytes(c.stdout);(RUN/'start_vault.stderr.log').write_bytes(c.stderr);assert c.returncode==0
    foreign_unchanged();write(RUN/'start_checks.json',dict(at=now().isoformat(),inputs_unchanged=True,results=results))


def publish():
    assert read(RUN/'start_checks.json')['inputs_unchanged'] and not (RUN/'start_publication.json').exists();verify_sources(read(RUN/'protocol.json'))
    foreign=foreign_unchanged();vault=RUN/'start_vault_refreshed.json'
    if not vault.exists():vault=RUN/'start_vault.json'
    protected={n.replace(chr(92),'/') for n in read(vault)['protected_existing_edits']}
    git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True,encoding='utf8').strip()
    assert not (ROOT/'.git/index.lock').exists() and not git('diff','--cached','--name-only')
    assert git('branch','--show-current')=='main' and git('remote','get-url','origin')=='https://gitlab.com/aziteam-group/drbdiss.git'
    names=['tracking/z13_spatial_set_readout.py','tests/test_z13_spatial_set_readout.py','scripts/z13_spatial_set_readout_20261025.py','scripts/z13_spatial_set_recovery1_20261025.py','scripts/z13_spatial_set_recovery2_20261025.py',Path(__file__).relative_to(ROOT).as_posix(),'experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [f.relative_to(ROOT).as_posix() for f in BASE.glob('*.json') if f.name!='status.json']
    names += [f.relative_to(ROOT).as_posix() for f in (BASE/'features').glob('*.json') if f.name!='status.json']
    for parent in (BASE/'recovery1',):
        names += [f.relative_to(ROOT).as_posix() for f in parent.glob('*.json') if f.name!='status.json']
        names += [f.relative_to(ROOT).as_posix() for f in (parent/'features').glob('*.json') if f.name!='status.json']
        names += [f.relative_to(ROOT).as_posix() for f in (parent/'features').glob('*/*/chunk_*.json')]
    for path in [*RUN.glob('*.json'),RUN/'features/protocol.json',RUN/'features/controller.json']:
        if path.is_file() and path.name not in ('status.json','start_publication.json','report.json','controller_result.json'):names.append(path.relative_to(ROOT).as_posix())
    names += [n for _,n in graph_rows() if n not in protected and (ROOT/n).is_file()]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    for i in range(0,len(names),20):subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+20]],cwd=ROOT,capture_output=True,check=True)
    staged=set(git('diff','--cached','--name-only','-z').split('\0'))-{''};assert staged and staged<=set(names)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Register bounded matched spatial column readout and start frozen GPU extraction','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True)
    commit=git('rev-parse','HEAD');subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'start_publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names));print(commit,flush=True)


def refresh_owned():
    assert read(RUN/'start_checks.json')['inputs_unchanged'];foreign_unchanged()
    prior=read(BASE/'recovery1/start_vault.json')['protected_existing_edits']
    protected={n.replace(chr(92),'/') for n in prior}
    owned={n:sha(ROOT/n) for _,n in graph_rows() if n not in protected and (ROOT/n).is_file() and (ROOT/n).suffix=='.md'}
    receipt=RUN/'start_owned_graph.json';assert not receipt.exists();write(receipt,owned)
    p=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'start_vault_refreshed.json'),'--refresh-owned',str(receipt)],cwd=ROOT,capture_output=True)
    (RUN/'start_refresh.stdout.log').write_bytes(p.stdout);(RUN/'start_refresh.stderr.log').write_bytes(p.stderr);assert p.returncode==0;foreign_unchanged()


if __name__=='__main__':
    if '--record' in sys.argv:record()
    elif '--checks' in sys.argv:checks()
    elif '--refresh-owned' in sys.argv:refresh_owned()
    elif '--publish' in sys.argv:publish()
