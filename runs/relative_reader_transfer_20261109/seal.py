"""Seal both relative-reader studies after green catalogue and graph checks."""
import study as t
from study import *
import psutil,shutil
assert read(OUT/'final_checks.json')['passed'];assert read(OUT/'final_graph.json')['passed'];assert sha(ROOT/'experiments/project_history.json')==read(OUT/'final_graph.json')['catalog_sha256'];folders=[REL,OUT];receipts=[]
guards=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks'
for folder in guards.iterdir():
    report=folder/'memory_guard.json'
    if not report.exists():continue
    r=read(report);command=r.get('command',[])
    if not any(any(str(parent).lower() in str(arg).lower() for parent in folders) for arg in command):continue
    if any(str(arg).lower()==str(Path(__file__)).lower() for arg in command):continue
    for pid,birth in r.get('owned',{}).items():
        try:assert psutil.Process(int(pid)).create_time()!=birth
        except psutil.NoSuchProcess:pass
    dest=OUT/'execution_logs'/folder.name;dest.mkdir(parents=True,exist_ok=False)
    for name in ['memory_guard.json','memory_guard.worker.log','supervisor.json']:
        if (folder/name).exists():shutil.copyfile(folder/name,dest/name)
    receipts.append(dict(token=folder.name,returncode=r['returncode'],reason=r.get('reason'),command=command,owned_ended=True))
failed={r['token']:r['returncode'] for r in receipts if r['returncode']!=0};assert failed=={'07727181a6e441dc99ee8bda2d1ab586':1,'f718d3deae88493f8fdfeea24d51991e':15,'7846390b4cac40cf88f5465402941025':1};assert all(r['reason'] is None for r in receipts)
verify(read(OUT/'protocol.json'));verify(read(OUT/'maps_complete.json'));s.verify(read(REL/'protocol.json'))
write('execution_receipt.json',dict(passed=True,all_previous_owned_processes_ended=True,stages=receipts,failed_metadata_save_and_premature_checks_retained=failed,science_scores_not_repeated=True,goal_complete=False))
sources={p.relative_to(ROOT).as_posix():sha(p) for folder in folders for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['seal.log','seal.err','execution_manifest.json']}
write('execution_manifest.json',dict(passed=True,sources=sources,open_files_excluded=['seal.log','seal.err'],scientific_goal_still_open=True));print('Relative studies sealed',len(sources),len(receipts),flush=True)
