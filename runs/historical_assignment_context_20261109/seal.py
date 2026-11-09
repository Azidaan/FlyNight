"""Archive owned process receipts and exact sources after completed stages."""
import audit as a
from audit import *
import psutil,shutil
F=ROOT/'runs/candidate_relative_alignment_20261109';FOLDERS=[DATA,BASE,F,OUT]
assert read(OUT/'final_checks.json')['passed'] and read(OUT/'final_graph.json')['passed'];assert sha(ROOT/'experiments/project_history.json')==read(OUT/'final_graph.json')['catalog_sha256']
guards=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks';receipts=[]
for folder in guards.iterdir():
    report=folder/'memory_guard.json'
    if not report.exists():continue
    r=read(report);command=r.get('command',[])
    if not any(any(str(parent).lower() in str(arg).lower() for parent in FOLDERS) for arg in command):continue
    if any(str(arg).lower()==str(Path(__file__)).lower() for arg in command):continue
    for pid,birth in r.get('owned',{}).items():
        try:assert psutil.Process(int(pid)).create_time()!=birth
        except psutil.NoSuchProcess:pass
    dest=OUT/'execution_logs'/folder.name;dest.mkdir(parents=True,exist_ok=False)
    for name in ['memory_guard.json','memory_guard.worker.log','supervisor.json']:
        if (folder/name).exists():shutil.copyfile(folder/name,dest/name)
    receipts.append(dict(token=folder.name,returncode=r['returncode'],reason=r.get('reason'),command=command,owned_ended=True))
assert receipts and all(r['reason'] is None for r in receipts)
failed=[r for r in receipts if r['returncode']!=0];assert len(failed)==1 and 'register' in failed[0]['command'] and failed[0]['token']=='ee53341564c74d699ee5823ed4f6b755'
s.verify(read(DATA/'protocol.json'));s.verify(read(BASE/'protocol.json'));s.verify(read(BASE/'maps_complete.json'));s.verify(read(BASE/'attribution_protocol.json'));s.verify(read(F/'protocol.json'));s.verify(read(OUT/'protocol.json'))
write('execution_receipt.json',dict(passed=True,all_previous_owned_processes_ended=True,stages=receipts,failed_registration_preserved=failed,goal_complete=False,no_new_id_assignment=True))
sources={p.relative_to(ROOT).as_posix():sha(p) for folder in FOLDERS for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['seal.log','seal.err','execution_manifest.json']}
write('execution_manifest.json',dict(passed=True,sources=sources,open_files_excluded=['seal.log','seal.err'],scientific_goal_not_completed=True));print('Sealed',len(sources),'files and',len(receipts),'owned stages',flush=True)
