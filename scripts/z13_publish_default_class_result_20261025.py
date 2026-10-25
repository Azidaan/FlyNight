"""Validate and publish only owned evidence for the completed default readout."""
from pathlib import Path
import os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
from scripts.z13_localization_labels_20261023 import verify_sources
RUN=ROOT/'runs/z13_default_class_trial_20261025'
def checks():
    assert not (RUN/'checks.json').exists();foreign_unchanged()
    from evaluation.project_history.index import fingerprint
    before=fingerprint(ROOT);results={}
    for label,args in [('validator',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','tests/test_z13_margin_rule.py','tests/test_z13_default_class.py','-q'])]:
        p=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf-8'},capture_output=True)
        (RUN/(label+'.stdout.log')).write_bytes(p.stdout);(RUN/(label+'.stderr.log')).write_bytes(p.stderr)
        results[label]=dict(exit_code=p.returncode,tail=p.stdout.decode('utf8',errors='replace')[-1600:]);assert p.returncode==0,results[label]
        print(label+' passed',flush=True)
    assert before==fingerprint(ROOT)
    p=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'vault_final.json')],cwd=ROOT,capture_output=True)
    (RUN/'vault.stdout.log').write_bytes(p.stdout);(RUN/'vault.stderr.log').write_bytes(p.stderr);assert p.returncode==0
    foreign_unchanged();write(RUN/'checks.json',dict(at=now().isoformat(),results=results,inputs_unchanged_during_tests=True))
def publish():
    assert not (RUN/'publication.json').exists()
    assert read(RUN/'checks.json')['inputs_unchanged_during_tests'] and read(RUN/'architecture_visual_review.json')['reviewed']
    assert read(ROOT/'dissertation/architecture/validation.json')['diagram_sheets']==1
    assert read(RUN/'independent_audit.json')['report_sha256']==sha(RUN/'internal_report.json')
    verify_sources(read(RUN/'protocol.json'));foreign=foreign_unchanged()
    protected={n.replace(chr(92),'/') for n in read(RUN/'vault_final.json')['protected_existing_edits']}
    git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True,encoding='utf8').strip()
    assert not (ROOT/'.git/index.lock').exists() and not git('diff','--cached','--name-only')
    assert git('branch','--show-current')=='main' and git('remote','get-url','origin')=='https://gitlab.com/aziteam-group/drbdiss.git'
    names=['scripts/z13_record_default_class_result_20261025.py',Path(__file__).relative_to(ROOT).as_posix(),'scripts/build_flyvis_architecture.py','scripts/flyvis_architecture_layout.py','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [f.relative_to(ROOT).as_posix() for f in RUN.glob('*.json') if f.name!='publication.json']
    for pattern in ('*/report.json','*/done.json','*/*/rows.json'):
        names += [f.relative_to(ROOT).as_posix() for f in (RUN/'internal').glob('*/*/seed*/'+pattern)]
    for name in ('FlyVis_architecture.drawio','FlyVis_architecture.svg','FlyVis_architecture.png','FlyVis_architecture_preview.png','FlyVis_architecture_detail.png','FlyVis_architecture_sources.json','validation.json'):
        names.append('dissertation/architecture/'+name)
    previous=set(read(ROOT/'runs/z13_margin_rule_audit_20261025/publication.json')['paths'])
    names += [n for status,n in graph_rows() if n not in protected and ((ROOT/n).is_file() or ('D' in status and n in previous))]
    names=sorted(set(names));assert not(set(names)&foreign) and '.gitignore' not in names
    for i in range(0,len(names),20):subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+20]],cwd=ROOT,check=True,capture_output=True)
    staged=set(git('diff','--cached','--name-only','-z').split('\0'))-{''};assert staged and staged<=set(names)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Record failed matched explicit-default readout and preserve 120-model evidence','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True)
    commit=git('rev-parse','HEAD');subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names));print(commit)
if __name__=='__main__':
    if '--checks' in sys.argv:checks()
    elif '--publish' in sys.argv:publish()
