"""Audit service-only source changes and publish the verified recovery launch."""
from pathlib import Path
import ast
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_region_assignment_recovery2_20261025 as service
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_localization_labels_20261023 import verify_sources
from scripts.z13_publish_native20_final_20261024 import foreign_unchanged,graph_rows
RUN=service.RUN;BASE=service.BASE


def main():
    assert not (RUN/'start_publication.json').exists()
    p=read(RUN/'protocol.json');verify_sources(p)
    original=(ROOT/'scripts/z13_region_assignment_20261025.py').read_text(encoding='utf-8')
    recovered=service.source()
    def functions(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(original),functions(recovered)
    unchanged=['train_worker','load_split','predict','policy_rows','queries_and_crops','label_worker']
    assert all(a[k]==b[k] for k in unchanged)
    normalized=recovered.replace("queries,_=queries_and_crops();os.environ['CUDA_VISIBLE_DEVICES']='0';byuid=","queries,_=queries_and_crops();byuid=")
    normalized=normalized.replace("read(ROOT/'runs/z13_region_assignment_20261025/labels/report.json')","read(RUN/'labels/report.json')").replace("path=ROOT/'runs/z13_region_assignment_20261025/labels/videos'/(v+'.json')","path=RUN/'labels/videos'/(v+'.json')")
    assert functions(normalized)['feature_worker']==a['feature_worker']
    old=read(BASE/'features/protocol.json');new=read(RUN/'features/protocol.json')
    assert old['save_at']==new['save_at'] and old['hard_stop']==new['hard_stop'] and old['budget_seconds']==new['budget_seconds']
    assert p['pipeline_hard_stop']==read(BASE/'protocol.json')['pipeline_hard_stop']
    assert read(BASE/'device_service_check.json')['passed'] and read(RUN/'features/price.json')['proceed']
    assert read(BASE/'roi_preflight.json')['all_choices_equal_independent_gate']
    write(RUN/'service_audit.json',dict(at=now().isoformat(),unchanged_scientific_functions=unchanged,feature_function_identical_except_device_restore_and_immutable_label_path=True,original_feature_deadline_preserved=True,global_deadline_preserved=True,completed_label_rows_reused=37296,source_sha256=sha(Path(__file__))))
    with (RUN/'service_validation.log').open('w',encoding='utf-8') as f:subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/validate_project_history.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    foreign=foreign_unchanged();receipt=RUN/'start_owned_graph.json';write(receipt,{n:sha(ROOT/n) for _,n in graph_rows() if (ROOT/n).is_file() and n not in foreign})
    subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'scripts/z13_build_vault_preserving_edits.py','--report',str(RUN/'vault_start.json'),'--refresh-owned',str(receipt)],cwd=ROOT,check=True,capture_output=True)
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_region_assignment_recovery1_20261025.py','scripts/z13_region_assignment_recovery1_console.ps1',
        'scripts/z13_region_assignment_recovery2_20261025.py','scripts/z13_region_assignment_recovery2_console.ps1','scripts/z13_region_assignment_device_check_20261025.py',
        'dissertation/ЕДИНЫЙ_ПЛАН.md','runs/z13_weekend_20261024/HANDOFF.md']
    names += [(BASE/n).relative_to(ROOT).as_posix() for n in ('device_service_check.json','pipeline_failure.json','labels/report.json','labels/controller_result.json','features/protocol.json','features/controller_result.json','features/status.json','recovery1/protocol.json')]
    names += [(RUN/n).relative_to(ROOT).as_posix() for n in ('protocol.json','launch.json','pipeline.json','features/protocol.json','features/controller.json','features/launch.json','features/price.json','labels/reused.json','labels/report.json','labels/controller_result.json','service_audit.json','start_owned_graph.json','vault_start.json')]
    names += [n for _,n in graph_rows() if n not in foreign and (ROOT/n).is_file()]
    names=sorted(set(names));assert not set(names)&foreign and '.gitignore' not in names
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0'))-{''};assert staged<=set(names)
    assert not (ROOT/'.git/index.lock').exists()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip()=='main'
    assert subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT).decode().strip()=='https://gitlab.com/aziteam-group/drbdiss.git'
    for i in range(0,len(names),25):subprocess.run(['git','add','-f','--',*names[i:i+25]],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','--quiet','-m','Restore CUDA visibility after CPU metadata import without extending budgets','-m','Co-authored-by: Codex <codex@openai.com>'],cwd=ROOT,check=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
    foreign_unchanged();write(RUN/'start_publication.json',dict(at=now().isoformat(),commit=commit,push_completed=True,paths=names))


if __name__=='__main__':main()
