"""Validate and publish only the audited spatial-readout result and owned records."""
from pathlib import Path
import os,sys,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
RUN=ROOT/'runs/z13_spatial_set_readout_20261025/recovery2'
OUT=RUN/'training/result_audit'
AUDIT=OUT
if '--deferred' in sys.argv:OUT=AUDIT/'deferred_20261025'
if '--clarified' in sys.argv:OUT=AUDIT/'deferred_clarified_20261025'


def checks():
    assert not (OUT/'checks.json').exists();foreign_unchanged()
    assert read(AUDIT/'audit.json')['internal_report_sha256']==sha(RUN/'training/internal_report.json')
    from evaluation.project_history.index import fingerprint
    before=fingerprint(ROOT);results={}
    for label,args in [('validator',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','tests/test_z13_spatial_set_readout.py','-q'])]:
        proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf8'},capture_output=True)
        (OUT/(label+'.stdout.log')).write_bytes(proc.stdout);(OUT/(label+'.stderr.log')).write_bytes(proc.stderr)
        results[label]=dict(exit_code=proc.returncode,tail=proc.stdout.decode('utf8',errors='replace')[-1200:]);assert proc.returncode==0,results[label]
        print(label+' passed',flush=True)
    assert fingerprint(ROOT)==before
    prior=read(RUN/'training/recovery4_gpu/publication.json')['paths']
    owned={n:sha(ROOT/n) for n in prior if n.startswith('ObsidianFlyVis/') and (ROOT/n).is_file() and (ROOT/n).suffix=='.md'}
    if OUT!=AUDIT:
        protected={n.replace(chr(92),'/') for n in read(AUDIT/'vault.json')['protected_existing_edits']}
        owned.update({n:sha(ROOT/n) for _,n in graph_rows() if n not in protected and (ROOT/n).is_file() and (ROOT/n).suffix=='.md'})
    write(OUT/'owned_graph.json',owned)
    proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(OUT/'vault.json'),'--refresh-owned',str(OUT/'owned_graph.json')],cwd=ROOT,capture_output=True)
    (OUT/'vault.stdout.log').write_bytes(proc.stdout);(OUT/'vault.stderr.log').write_bytes(proc.stderr);assert proc.returncode==0,proc.stderr.decode('utf8',errors='replace')
    foreign_unchanged();write(OUT/'checks.json',dict(at=now().isoformat(),results=results,inputs_unchanged_during_tests=True))


def publish():
    assert not (OUT/'publication.json').exists() and read(OUT/'checks.json')['inputs_unchanged_during_tests']
    assert read(OUT/'architecture_visual_review.json')['reviewed']
    assert read(ROOT/'dissertation/architecture/validation.json')['diagram_sheets']==1
    foreign=foreign_unchanged();protected={n.replace(chr(92),'/') for n in read(OUT/'vault.json')['protected_existing_edits']}
    git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True,encoding='utf8').strip()
    assert not (ROOT/'.git/index.lock').exists() and not git('diff','--cached','--name-only')
    assert git('branch','--show-current')=='main' and git('remote','get-url','origin')=='https://gitlab.com/aziteam-group/drbdiss.git'
    names=['scripts/z13_audit_spatial_set_result_20261025.py','scripts/z13_record_spatial_set_result_20261025.py',Path(__file__).relative_to(ROOT).as_posix(),'scripts/build_flyvis_architecture.py','scripts/flyvis_architecture_layout.py','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names+=['scripts/z13_defer_spatial_set_20261025.py','scripts/z13_clarify_spatial_deferral_20261025.py','scripts/z13_recover_deferred_architecture_20261025.py','dissertation/architecture/Пространственное_считывание_отложено_2026-10-25.md']
    if OUT!=AUDIT:names += [f.relative_to(ROOT).as_posix() for f in AUDIT.glob('*.json') if f.name!='publication.json']
    if '--clarified' in sys.argv:names += [f.relative_to(ROOT).as_posix() for f in (AUDIT/'deferred_20261025').glob('*.json') if f.name!='publication.json']
    names += [f.relative_to(ROOT).as_posix() for f in OUT.glob('*.json') if f.name!='publication.json']
    names += [(RUN/'training'/f).relative_to(ROOT).as_posix() for f in ['internal_report.json','status.json','recovery4_gpu/controller_result.json','recovery4_gpu/failure.json','recovery4_gpu/publication.json']]
    for f in (RUN/'training/internal').glob('*/seed*/fold*/*.json'):names.append(f.relative_to(ROOT).as_posix())
    for n in ('FlyVis_architecture.drawio','FlyVis_architecture.svg','FlyVis_architecture.png','FlyVis_architecture_preview.png','FlyVis_architecture_detail.png','FlyVis_architecture_sources.json','validation.json'):names.append('dissertation/architecture/'+n)
    previous=set(read(RUN/'training/recovery4_gpu/publication.json')['paths'])
    names += [n for status,n in graph_rows() if n not in protected and ((ROOT/n).is_file() or ('D' in status and n in previous))]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    for i in range(0,len(names),20):subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+20]],cwd=ROOT,capture_output=True,check=True)
    staged=set(git('diff','--cached','--name-only','-z').split('\0'))-{''};assert staged and staged<=set(names)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Record audited negative spatial-readout result and preserve completion failure','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True)
    commit=git('rev-parse','HEAD');subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(OUT/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names));print(commit,flush=True)


if __name__=='__main__':
    if '--checks' in sys.argv:checks()
    elif '--publish' in sys.argv:publish()
