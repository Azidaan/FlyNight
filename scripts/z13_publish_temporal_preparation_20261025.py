"""Validate preparation evidence, preserve foreign edits, publish exact owned files."""
from pathlib import Path
import os,sys,subprocess,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
from scripts.z13_localization_labels_20261023 import verify_sources
GATE=ROOT/'runs/z13_temporal_observation_gate_20261025'
FEATURES=ROOT/'runs/z13_temporal_region_features_20261025'
OUT=GATE/'publication_checks'


def checks():
    assert not (OUT/'checks.json').exists();foreign_unchanged()
    p=read(FEATURES/'protocol.json');verify_sources(p)
    # Never compete with the initial cost measurement on the CPU.
    while not (FEATURES/'price.json').exists() and not (FEATURES/'controller_result.json').exists():
        if now()>=__import__('datetime').datetime.fromisoformat(p['save_at']):break
        time.sleep(2)
    from evaluation.project_history.index import fingerprint
    before=fingerprint(ROOT);results={}
    for label,args in [('validator',['scripts/validate_project_history.py']),('tests',['-m','pytest','tests/test_project_history.py','-q'])]:
        proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONIOENCODING':'utf8'},capture_output=True)
        (OUT/(label+'.stdout.log')).parent.mkdir(parents=True,exist_ok=True)
        (OUT/(label+'.stdout.log')).write_bytes(proc.stdout);(OUT/(label+'.stderr.log')).write_bytes(proc.stderr)
        results[label]=dict(exit_code=proc.returncode,tail=proc.stdout.decode('utf8',errors='replace')[-1200:]);assert proc.returncode==0,results[label];print(label+' passed',flush=True)
    assert fingerprint(ROOT)==before
    previous=ROOT/'runs/z13_spatial_set_readout_20261025/recovery2/training/result_audit/deferred_clarified_20261025/vault.json'
    protected={n.replace(chr(92),'/') for n in read(previous)['protected_existing_edits']}
    owned={n:sha(ROOT/n) for _,n in graph_rows() if n not in protected and (ROOT/n).is_file() and (ROOT/n).suffix=='.md'}
    write(OUT/'owned_graph.json',owned)
    proc=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(OUT/'vault.json'),'--refresh-owned',str(OUT/'owned_graph.json')],cwd=ROOT,capture_output=True)
    (OUT/'vault.stdout.log').write_bytes(proc.stdout);(OUT/'vault.stderr.log').write_bytes(proc.stderr);assert proc.returncode==0
    foreign_unchanged();write(OUT/'checks.json',dict(at=now().isoformat(),results=results,catalogue_stable_during_checks=True))


def architecture():
    from scripts import build_flyvis_architecture as gen
    from scripts.flyvis_bottlenecks_layout import diagram_digest
    arch=ROOT/'dissertation/architecture';before=read(arch/'validation.json');first=diagram_digest(__import__('xml.etree.ElementTree',fromlist=['parse']).parse(arch/'FlyVis_architecture.drawio').getroot().find('diagram'))
    out=OUT/'architecture';assert not out.exists();gen.OUT=out;gen.main()
    import xml.etree.ElementTree as ET
    assert diagram_digest(ET.parse(out/'FlyVis_architecture.drawio').getroot().find('diagram'))==first
    assert sha(out/'FlyVis_architecture.png')==before['outputs']['FlyVis_architecture.png']
    import shutil
    for f in out.iterdir():
        if f.is_file():
            temp=arch/(f.name+'.temporal.next');shutil.copy2(f,temp);os.replace(temp,arch/f.name)
    write(OUT/'architecture_checks.json',dict(at=now().isoformat(),two_sheets=True,first_sheet_unchanged=True,new_data_readiness_source_verified=True))


def publish():
    assert read(OUT/'checks.json')['catalogue_stable_during_checks'] and read(OUT/'architecture_visual_review.json')['reviewed'] and not (OUT/'publication.json').exists()
    foreign=foreign_unchanged();git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT,text=True,encoding='utf8').strip()
    assert git('branch','--show-current')=='main' and git('remote','get-url','origin')=='https://gitlab.com/aziteam-group/drbdiss.git'
    assert not (ROOT/'.git/index.lock').exists() and not git('diff','--cached','--name-only')
    names=['scripts/z13_temporal_observation_gate_20261025.py','scripts/z13_record_temporal_observation_gate_20261025.py','scripts/z13_temporal_region_features_20261025.py','scripts/z13_after_temporal_features_20261025.py',Path(__file__).relative_to(ROOT).as_posix(),'scripts/flyvis_bottlenecks_layout.py','experiments/project_history.json','dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [f.relative_to(ROOT).as_posix() for f in GATE.glob('*.json')]
    names += [f.relative_to(ROOT).as_posix() for f in OUT.glob('*.json') if f.name!='publication.json']
    names += [f.relative_to(ROOT).as_posix() for f in FEATURES.glob('*.json') if f.name not in ('status.json','failure.json','report.json')]
    for n in ('FlyVis_architecture.drawio','FlyVis_architecture.svg','FlyVis_architecture.png','FlyVis_architecture_preview.png','FlyVis_architecture_detail.png','FlyVis_architecture_sources.json','validation.json','FlyVis_bottlenecks.png','FlyVis_bottlenecks_preview.png','FlyVis_bottlenecks.svg','FlyVis_bottlenecks_sources.json'):names.append('dissertation/architecture/'+n)
    protected={n.replace(chr(92),'/') for n in read(OUT/'vault.json')['protected_existing_edits']}
    previous=set(read(ROOT/'runs/z13_spatial_set_readout_20261025/recovery2/training/result_audit/deferred_clarified_20261025/publication.json')['paths'])
    names += [n for status,n in graph_rows() if n not in protected and ((ROOT/n).is_file() or ('D' in status and n in previous))]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    for i in range(0,len(names),20):subprocess.run(['git','-c','core.autocrlf=input','add','-f','--',*names[i:i+20]],cwd=ROOT,check=True,capture_output=True)
    staged=set(git('diff','--cached','--name-only','-z').split('\0'))-{''};assert staged and staged<=set(names)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Verify expanded past-observation inputs and prepare matched temporal features','-m','Co-authored-by: Codex <noreply@openai.com>'],cwd=ROOT,check=True)
    commit=git('rev-parse','HEAD');subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(OUT/'publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names));print(commit,flush=True)


if __name__=='__main__':
    if '--checks' in sys.argv:checks()
    elif '--architecture' in sys.argv:architecture()
    elif '--publish' in sys.argv:publish()
