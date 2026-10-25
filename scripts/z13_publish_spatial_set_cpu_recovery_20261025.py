"""Validate and publish service-only RAM recovery evidence, excluding live model files."""
from pathlib import Path
import os,sys,subprocess,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
from scripts.z13_spatial_set_recovery3_20261025 import RUN,SERVICE,source
from scripts.z13_localization_labels_20261023 import verify_sources
RAM_SERVICE=SERVICE
if (RUN/'training/recovery4_gpu/protocol.json').exists():
    from scripts.z13_spatial_set_recovery4_gpu_20261025 import SERVICE,source


def checks():
    assert not (SERVICE/'checks.json').exists();foreign_unchanged();p=read(SERVICE/'protocol.json');verify_sources(p)
    assert read(RUN/'features/report.json')['complete'] and read(RUN/'features/controller_result.json')['all_owned_ended']
    assert hashlib_source()==p['executed_service_sha256']
    # Do not compete with the initial model's resource-admission timing.
    deadline=__import__('datetime').datetime.fromisoformat(p['save_at'])
    price=SERVICE/'price.json' if SERVICE.name=='recovery4_gpu' else RUN/'training/price.json'
    while not price.exists() and not (SERVICE/'controller_result.json').exists():
        if now()>=deadline:break
        time.sleep(2)
    from evaluation.project_history.index import fingerprint
    before=fingerprint(ROOT);results={}
    for label,args in [('validator',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','tests/test_z13_spatial_set_readout.py','-q'])]:
        proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf8'},capture_output=True)
        (SERVICE/(label+'.stdout.log')).write_bytes(proc.stdout);(SERVICE/(label+'.stderr.log')).write_bytes(proc.stderr)
        results[label]=dict(exit_code=proc.returncode,tail=proc.stdout.decode('utf8',errors='replace')[-1400:]);assert proc.returncode==0,results[label];print(label+' passed',flush=True)
    assert before==fingerprint(ROOT)
    args=[str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(SERVICE/'vault.json')]
    if SERVICE!=RAM_SERVICE and (RAM_SERVICE/'vault.json').exists():
        protected={n.replace(chr(92),'/') for n in read(RAM_SERVICE/'vault.json')['protected_existing_edits']}
        owned={n:sha(ROOT/n) for _,n in graph_rows() if n not in protected and (ROOT/n).is_file() and (ROOT/n).suffix=='.md'}
        receipt=SERVICE/'owned_graph.json';assert not receipt.exists();write(receipt,owned);args+=['--refresh-owned',str(receipt)]
    proc=subprocess.run(args,cwd=ROOT,capture_output=True)
    (SERVICE/'vault.stdout.log').write_bytes(proc.stdout);(SERVICE/'vault.stderr.log').write_bytes(proc.stderr);assert proc.returncode==0
    foreign_unchanged();write(SERVICE/'checks.json',dict(at=now().isoformat(),results=results,inputs_unchanged_during_tests=True,feature_phase_complete=True,live_model_results_not_audited=True))


def hashlib_source():
    import hashlib
    return hashlib.sha256(source().encode()).hexdigest()


def publish():
    assert not (SERVICE/'publication.json').exists() and read(SERVICE/'checks.json')['inputs_unchanged_during_tests']
    verify_sources(read(SERVICE/'protocol.json'));foreign=foreign_unchanged()
    protected={n.replace(chr(92),'/') for n in read(SERVICE/'vault.json')['protected_existing_edits']}
    git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True,encoding='utf8').strip()
    assert not (ROOT/'.git/index.lock').exists() and not git('diff','--cached','--name-only')
    assert git('branch','--show-current')=='main' and git('remote','get-url','origin')=='https://gitlab.com/aziteam-group/drbdiss.git'
    names=['scripts/z13_spatial_set_recovery3_20261025.py','scripts/z13_spatial_set_recovery4_gpu_20261025.py',Path(__file__).relative_to(ROOT).as_posix(),'experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    for f in [*SERVICE.glob('*.json'),RUN/'controller_result.json',RUN/'features/report.json',RUN/'features/controller_result.json',RUN/'training/protocol.json',RUN/'training/controller.json',RUN/'training/controller_result.json',RUN/'start_publication.json']:
        if f.is_file() and f.name not in ('publication.json','status.json','failure.json'):names.append(f.relative_to(ROOT).as_posix())
    if SERVICE!=RAM_SERVICE:
        names += [f.relative_to(ROOT).as_posix() for f in RAM_SERVICE.glob('*.json') if f.name not in ('publication.json','status.json','failure.json')]
        names += [(RUN/'training/internal/D_spatial/seed0/fold0'/n).relative_to(ROOT).as_posix() for n in ('done.json','report.json','rows.json')]
    names += [n for _,n in graph_rows() if n not in protected and (ROOT/n).is_file()]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    for i in range(0,len(names),20):subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+20]],cwd=ROOT,capture_output=True,check=True)
    staged=set(git('diff','--cached','--name-only','-z').split('\0'))-{''};assert staged and staged<=set(names)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve spatial features and completed CPU model; continue matched readouts on GPU','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True)
    commit=git('rev-parse','HEAD');subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(SERVICE/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names));print(commit,flush=True)


if __name__=='__main__':
    if '--checks' in sys.argv:checks()
    elif '--publish' in sys.argv:publish()
